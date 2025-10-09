# Kanji Handwriting Recognition
***Skills***_: Machine Learning, Neural Networks, Python (NumPy, Pandas, scikit-learn, TensorFlow, Matplotlib)_

## Overview
Many historical manuscripts, letters, and books remain inaccessible to the general Japanese public as the ancient Kuzushiji script has fazed out of popularity. Furthermore, each of its character can be written in many ways, leaving some texts intelligible to all but experts linguists. As such, I developed a machine learning pipeline for recognizing handwritten Kuzishiji characters and translate them into modern Japanese (Romaji). It automatically compares different sklearn models () and hyperparameters () and choose the best performing option for recognition.


```python
# SETUP
## Import libraries
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import openml
import time
import math
import gdown
from tqdm.auto import tqdm
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from IPython import display
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input

## Check sklearn version
from packaging import version
import sklearn
sklearn_version = sklearn.__version__
if version.parse(sklearn_version) < version.parse("1.6.0"):
    print("scikit-learn is outdated (currently: {}, required: 1.6.0 or higher). Please update by running 'pip install -U scikit-learn' in your terminal".format(sklearn_version))
else:
    print("scikit-learn version is OK (currently: {})".format(sklearn_version))

## Display plots directly in notebook
%matplotlib inline

## Hide convergence warning
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils._testing import ignore_warnings
warnings.simplefilter(action="ignore", category=ConvergenceWarning)


```

    scikit-learn version is OK (currently: 1.6.1)
    


```python
# LOAD DATA
# Load Devanagari data from OpenML
data = openml.datasets.get_dataset("Kuzushiji-MNIST", download_data=True)
Kuzushiji_X, Kuzushiji_y, _, _ = data.get_data(target=data.default_target_attribute)

# Scaling to [0..1] as Logistic Regression can be sensitive to scale
Kuzushiji_X_scaled = Kuzushiji_X / 255.0

# Split the data, using 25% for training and 25% for testing (in the interest of time/resources)
X, X_eval, y, y_eval = train_test_split(Kuzushiji_X_scaled, Kuzushiji_y, stratify=Kuzushiji_y, train_size=0.25, test_size=0.25, random_state=0)

# Sorted label list
labels = sorted([label for label in set(Kuzushiji_y)])

# Map classes to Kuzushiji characters
Kuzushiji_classes = {0:"o", 1: "ki", 2: "su", 3: "tsu", 4: "na", 5: "ha",
                     6: "ma", 7: "ya", 8: "re", 9: "wo"}
y_classes = np.array([Kuzushiji_classes[int(yi)] for yi in y])
```

### Data Exploration



```python
def plot_class_examples(images, labels, num_per_class=7, class_names=None, title=""):
    """
    Plot each class in its own subplot, arranging the subplots in 2 columns.
    Each example is a separate square image with its own border.
    """
    classes = sorted(labels.unique())
    n_classes = len(classes)
    n_cols = 2
    n_rows = math.ceil(n_classes / n_cols)
    row_len = num_per_class
    n_img_rows = math.ceil(num_per_class / row_len)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(num_per_class + 1, n_rows + 1))
    axes = np.atleast_2d(axes).flatten()
    
    for i, c in enumerate(classes):
        ax = axes[i]
        idx = labels[labels == c].index[:num_per_class]
        class_images = images.loc[idx]
        
        ax.set_aspect('equal')
        ax.set_xlim(0, row_len*28)
        ax.set_ylim(0, n_img_rows*28)
        
        for j, (_, row) in enumerate(class_images.iterrows()):
            r = j // row_len
            c_ = j % row_len
            img = row.values.reshape(28,28)
            ax.imshow(img, cmap=plt.cm.Blues,
                      extent=(c_*28, (c_+1)*28, (n_img_rows-r-1)*28, (n_img_rows-r)*28))
            rect = patches.Rectangle((c_*28, (n_img_rows-r-1)*28), 28, 28,
                                     linewidth=0.5, edgecolor='black', facecolor='none')
            ax.add_patch(rect)
        
        ax.set_xticks([])
        ax.set_yticks([])
        if class_names:
            ax.set_title(class_names[int(labels[idx[0]])], fontsize=10, pad=5)
    
    # Turn off unused subplots
    for j in range(i+1, len(axes)):
        axes[j].axis('off')
    
    # Add title
    if title:
        fig.suptitle(title, fontsize=14, y=0.9)
    
    # Plot & save figure
    plt.tight_layout(pad=1.0, h_pad=0.5, w_pad=0.5, rect=[0,0,1,0.95])
    plt.savefig("Kuzushiji Script.png", bbox_inches='tight')
    plt.show()

plot_class_examples(X, y, num_per_class=5, class_names=Kuzushiji_classes, title="Kuzushiji Script Examples")
```


      Cell In[96], line 47
        *()
        ^
    SyntaxError: can't use starred expression here
    


## Question 1: Pixels vs Embeddings
Given the high variability in the way characters are written, pixel values may not be the best representation for linear models. We can instead use a (pretrained) deep learning model to extract a better vector representation (an embedding) of the images. Then, we can use these features (instead of the pixel values) as input for your linear model. As we will see later in the course, such representations are robust against changes in the exact position of the letters (translation invariance) and can identify small and larger patterns (from lines and curves to entire letters), which helps when characters have similar sub-curves but aren't exactly the same.

<img src="https://qdrant.tech/articles_data/what-are-embeddings/How-Embeddings-Work.jpg" width=800 />

Below we provide code to create such embeddings, using a MobileNet architecture pretrained on ImageNet. You don't need to fully understand this code yet, but you'll be expected to understand and write code like this by the end of the course. To save unnecessary compute (and issues with GPUs) we have already included the embedded data in your assignment repo, and will load it now.

**Optional**: you can always rerun the code out of interest (uncomment the lines below). We do recommend using a GPU for this (in Colab, change your Runtime to a GPU Runtime, T4 is fine). If you're already familiar with deep learning architectures, we recommend reading [the MobileNet v2 paper](https://arxiv.org/abs/1801.04381).


```python
# Create image embeddings using MobileNetV2
def create_embedding(X, batch_size=32):
    X = X*255.0 # MobileNetV2 was likely trained on ImageNet data in the standard [0,255] scale
                # When using pretrained embeddings, always make the input data as similar as possible
    if isinstance(X, pd.DataFrame):
        X = X.to_numpy() # Convert the input DataFrame to a numpy array
    n_batches = int(np.ceil(X.shape[0] / batch_size)) # Determine the total number of batches
    all_embeddings = [] # Stores batch embeddings

    # Initialize MobileNetV2, only keep the embedding layers, use the smallest available input shape
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(96, 96, 3), pooling='avg')

    # Process data in batches (to avoid memory issues)
    for i in tqdm(range(n_batches)):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        X_batch = X[start_idx:end_idx]

        X_reshaped = X_batch.reshape((-1, 28, 28)) # Convert our pixel vectors to 28x28 images and add a channel dimension
        X_reshaped = np.stack((X_reshaped,)*3, axis=-1) # Convert to 3 channels (required for MobileNet) by duplicating the grayscale data
        X_resized = tf.image.resize(X_reshaped, [96,96]).numpy() # Upscale images to 96x96 to fit smallest supported MobileNetV2 resolution
        X_preprocessed = preprocess_input(X_resized) # Preprocess images according to MobileNetV2 requirements

        # Generate embeddings for the batch by passing our data through the pretrained network
        batch_embeddings = base_model.predict(X_preprocessed, batch_size=batch_size, verbose=0)

        # Collect the embeddings
        all_embeddings.append(batch_embeddings)

    embeddings = np.vstack(all_embeddings) # Concatenate all batch embeddings
    return embeddings
```


```python
# Loading the embedded data from disc
X_embedded = np.load("X_emb_train.npy")
X_eval_embedded = np.load("X_emb_test.npy")
```

We can visualize the embeddings using [t-SNE](https://lvdmaaten.github.io/tsne/), a dimensionality reduction method that helps to visualize the original high-dimensional embedding into a 2D scatterplot. From this we can see that the images of the same character are clustered together, which suggests that the embedding may be useful to distinguish them.

**Optional** You can run the code yourself by commenting out the call below. You can also try other values for the 'perplexity' parameter. We commented it out (and show the plot) because computing t-SNE is quite expensive.

<img src="tsne.png" width=700 />


```python
def plot_tsne():
  # Perform t-SNE
  tsne = TSNE(n_components=2, perplexity=30, random_state=42)
  embeddings_2d = tsne.fit_transform(X_embedded)

  # Plot the t-SNE embedding
  plt.figure(figsize=(10, 8))
  y_numeric = y.astype(int)
  unique_labels = np.unique(y_numeric)
  cmap = plt.get_cmap('tab10')
  for label in unique_labels:
      indices = y_numeric == label
      plt.scatter(
          embeddings_2d[indices, 0], embeddings_2d[indices, 1],
          color=cmap(label), label=Kuzushiji_classes[label], alpha=0.7
      )
  plt.xlabel("t-SNE Component 1")
  plt.ylabel("t-SNE Component 2")
  plt.title("t-SNE Visualization of Character Embeddings")
  plt.legend(title="Class Labels", loc="best")
  plt.show()

plot_tsne()
```


    
![png](README_files/README_9_0.png)
    


# Questions
Please solve the questions below. Notes:
* The questions can be done in any order, although subquestions (e.g. 1.1 and 1.2) build on each other.
* Every sub-question counts for 1 point and there are 10 points in total. During final grading, partial points are possible.
* Some questions are multiple choice. Indication all options will give you 0 points (but you can't get negative points on a question).
* Your code, output plots and the multiple-choice questions will all be graded. Make sure that everything (all outputs and all plots) are in your notebook when you submit it.

### Question 1.1: Model tuning (1 point)
Implement a method `evaluate_LR` that evaluates a Logistic Regression model trained with a given regularization constant (C), and the Limited-memory Broyden–Fletcher–Goldfarb–Shanno optimizer. It should return the train and test score of a 5-fold stratified cross-validation using the accuracy metric. You don't need to do any additional preprocessing of the data at this point. It is already scaled.

**Important:** Only use the variables passed as parameters (in this case, `X`, `y`, and `C`). Do not add additional parameters to the function, and do not use global variables in the function implementation (except for imported modules and functions, which you are free to use). These guidelines also apply to any future questions in this and future assignments.

Next, you need to find the optimal amount of regularization for this dataset. To do this, implement a method `plot_curve` that plots the training and test scores of `evaluate_LR` on a stratified subsample of size `train_size` on the Devanagari dataset, for C values ranging from at least 1e-4 (10<sup>-4</sup>) to 1e4 (10<sup>4</sup>) on a log base 10 scale, with at least 9 values. Use `random_state=0` for reproducibility. You should use the plotting function `plot_live` defined above (carefully read what it does - it will save you time!).

Tune C on a 25% subsample of the data to `evaluate_LR`. Using a subsample won't give you optimal performance, but when tuning hyperparameters (which is expensive) you can often get a pretty good estimate of the best hyperparameters by using a smaller sample. When you have a rough idea of where the good hyperparameters lie, you can evaluate these on a larger sample. This is also called multi-fidelty optimization. For now, using a 25% subsample of the training set is fine to optimize the C parameter is fine.

*note:* Throughout the notebook, you will find cells that call functions on the `validation` module. You can run these cells to perform a _very_ basic test on your provided answer. If validation fails, that means there is a technical error with the provided answer (e.g., the function signature was changed). Passing validation does *not* necessarily mean your answer is correct, and is *not* a replacement for performing your own sanity checks. The `validation.py` file needs to be in the same folder as this notebook. If you are running this is Colab, you need to upload it to your working directory.


```python
# Implement. Do not change the name or signature of these functions.
def evaluate_LR(X, y, C):
    """ Evaluate a Logistic Regression model with cross-validation on the provided image data.
    Keyword arguments:
    X -- the data for training and testing
    y -- the correct output values
    C -- the regularization constant

    Returns: a dictionary with the mean train and test score, e.g. {"train": 0.9, "test": 0.95}
    """
    model = sklearn.linear_model.LogisticRegression(C=C, solver='lbfgs')
    cross_val = sklearn.model_selection.cross_validate(model, X, y, return_train_score=True, cv = 5, scoring = 'accuracy')
    train_mean = np.mean(cross_val["train_score"]);
    test_mean = np.mean(cross_val["test_score"]);
    return {"train": train_mean, "test": test_mean}

def plot_curve(X,y,train_size):
    """ Plots the train and test accuracy of logistic regression on a
    subsample of the given data for different amounts of regularization.
    X          -- the data for training and testing
    y          -- the correct labels
    train_size -- the proportion of the data used for training and testing, between 0.0 and 1.0.

    Returns: a plot as described above, with C on the x-axis and accuracy on
    the y-axis.
    """
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 1 - train_size, stratify=y, random_state=0)
    C = np.logspace(-4, 4, 9);
    evaluations = [];
    for c in C:
        evaluations.append(evaluate_LR(X_train, y_train, c))
    df = pd.DataFrame(evaluations)
    
    line_train, = plt.plot(C, df["train"], label="train")
    line_test, = plt.plot(C, df["test"], label="test")
    plt.legend(handles=[line_train, line_test])
    plt.xlabel("C")
    plt.ylabel("accuracy")
    plt.xscale("log")
    plt.show()


plot_curve(X,y,0.25)
```


    
![png](README_files/README_14_0.png)
    


### Question 1.2: Interpretation (1 point)
Interpret the graph. Indicate which of the following answers are correct. Run additional tests if needed.

- 'A': Setting C=1e-4 causes the model to underfit
- 'B': Setting C=1e-4 causes the model to overfit
- 'C': Setting C=1e4 causes the model to underfit
- 'D': Setting C=1e4 causes the model to overfit
- 'E': Any value for C higher than 10 is fine (close to optimal)
- 'F': None of these is correct

Enter the correct letter(s) in value `q_1_2` in the code. They may be separated by commas (e.g. `'A,B,C'`) if multiple answers are correct.




```python
# Fill in the correct answer. Don't change the name of the variable
q_1_2 = 'A,D'
```

### Question 1.3: Evaluation on held-out data (1 point)
Implement the `evaluate_test` function that train a Logistic Regression model with C=0.1 on the given training set, and correctly evaluates it on the evaluation set.

*NOTE: This is the first exercise in which you should use the evaluation set.*


```python
def evaluate_test(X_train, y_train, X_eval, y_eval):
    """ Evaluate a Logistic Regression model
    X_train -- the training data
    y_train -- the training labels
    X_eval  -- the evaluation (test) data
    y_eval  -- the evaluation (test) labels

    Returns: the evaluation score (accuracy) of the optimal model trained on pixel data
    """
    C=0.1
    model = sklearn.linear_model.LogisticRegression(C=C)
    fit = model.fit(X_train, y_train)
    y_pred = fit.predict(X_eval)
    score = sklearn.metrics.accuracy_score(y_eval, y_pred)
    return score

# Calling the functions. Don't change these calls
pixel_lr_score = evaluate_test(X, y, X_eval, y_eval)
embedding_lr_score = evaluate_test(X_embedded, y, X_eval_embedded, y_eval)
```


```python
print("Pixel representation score:",pixel_lr_score)
print("Embedding representation score:",embedding_lr_score)
```

    Pixel representation score: 0.7929142857142857
    Embedding representation score: 0.9120571428571429
    

Interpret the results. Indicate which of the following answers are correct.

- 'A': The pixel representation works much better than the deep learning embedding. This is because the deep learning embedding was pretrained on images from the internet and that's very different than the characters here.
- 'B': The pixel representation works much better since it's a lot more concise representation of the data.
- 'C': The deep learning embedding representation works much better since it represents general components of images (e.g. lines and curves) and complex combinations thereof.
- 'D': The deep learning embedding representation works much better since it was pretrained on ancient Japanese characters (hence, it's cheating).
- 'E': None of these is correct

Enter the correct letter(s) in value `q_1_3` in the code. They may be separated by commas (e.g. `'A,B,C'`) if multiple answers are correct.


```python
# Fill in the correct answer. Don't change the name of the variable
q_1_3 = 'C'
```

### Question 1.4: Few-shot learning
Few-shot learning is a branch of machine learning where we need to learn from very few examples.
Implement the method `few_shot_learning`, which is identical to `evaluate_test`, only it uses only 1% of the training data, about 17 images per character (and all of the evaluation data).

*NOTE: This is the last exercise in which you should use the evaluation set.*


```python
def few_shot_learning(X_train, y_train, X_eval, y_eval):
    """ Evaluate a Logistic Regression model
    X_train -- the training data
    y_train -- the training labels
    X_eval  -- the evaluation (test) data
    y_eval  -- the evaluation (test) labels

    Returns: the evaluation score (accuracy) of the optimal model trained on pixel data
    """
    C=0.1
    X_train_few, _, y_train_few, _ = train_test_split(X_train, y_train, train_size = 0.01, stratify = y_train, random_state=0)
    
    model = sklearn.linear_model.LogisticRegression(C=C)
    fit = model.fit(X_train_few, y_train_few)
    y_pred = fit.predict(X_eval)
    score = sklearn.metrics.accuracy_score(y_eval, y_pred)
    return score

# Calling the functions. Don't change these calls
pixel_few_shot = few_shot_learning(X, y, X_eval, y_eval)
embedding_few_shot = few_shot_learning(X_embedded, y, X_eval_embedded, y_eval)
```


```python
print("Pixel representation score:",pixel_few_shot)
print("Embedding representation score:",embedding_few_shot)
```

    Pixel representation score: 0.6313714285714286
    Embedding representation score: 0.6853714285714285
    

Interpret the results. Indicate which of the following answers are correct.

- 'A': The pixel representation now work much better than the deep learning embedding. The embeddings is too high-dimensional so you need more data to train the linear classifier well.
- 'B': The deep learning embedding representation is still equally good as before. This is because everything we need to know is in the embedding and we can learn from very few examples.
- 'C': Both models perform worse now. The embedding is too general, and we still need sufficient training data for the classifier (logistic regression) to allow it to learn a mapping from the embeddings to the character classes.
- 'D': None of these is correct

Enter the correct letter(s) in value `q_1_4` in the code. They may be separated by commas (e.g. `'A,B,C'`) if multiple answers are correct.


```python
# Fill in the correct answer. Don't change the name of the variable
q_1_4 = 'C'
```

### Question 2.1: Model inspection (1 point)
Implement a function `plot_character_coefficients` that trains 3 logistic regression models on the pixel representation (not the embeddings), one trained with C=1e-4, one with C=1e-1, and one with C=1e4. For all models, use Stochastic Average Gradient Descent with at least 100 iterations. Also evaluate each model by using 75% of the data for training and the rest for testing.

To interpret whether the model has learned something useful, we will now plot the coefficients of these three models. Remember that Logistic Regression is a binary classifier, and you can assume (see the note below) that a one-vs-rest-like approach is used for multi-class problems, hence the n-th set of coefficients in the model belong to the submodel that separates the n-th class from the rest. To plot these coefficients, you'll to reshape them into a 28x28 matrix and then plot that (e.g. use `imshow` as in previous examples, and use an intuitive [colormap](https://matplotlib.org/stable/users/explain/colors/colormaps.html) to represent low and high values, like 'plasma').

Your function is given a character `character`. First, plot an example of this character (any will do). Then, plot the coefficients of the model that separates the character from the other characters. Repeat 3 times for the different C values. It is also given a `penalty` parameter which allows to set the regularizer (e.g. L2 loss).

Hence, you should return four plots: one character plot and three coeficients plots, and they should appear next to each other (not below each other). Add the C-value and test accuracy to the title of the three coefficient plots. Evaluate the test accuracy with a simple 75-25 train-test split).

Run `plot_character_coefficients` for character `tsu`.

Note: Scikit-learn actually uses [a more sophisticated approach](https://scikit-learn.org/stable/auto_examples/linear_model/plot_logistic_multinomial.html#sphx-glr-auto-examples-linear-model-plot-logistic-multinomial-py) here than simple one-vs-all. It uses the fact that Logistic Regression predicts probabilities, and hence the probabilities of each class are taken into account (in a softmax function). It will still produce one model per class.


```python
# Implement. Do not change the name or signature of this function.
def plot_character_coefficients(X, y, character, penalty):
    """ Plots 28x28 heatmaps showing the coefficients of three Logistic
    Regression models, each with different amounts of regularization values.
    X -- the data for training and testing
    y -- the correct labels
    character -- the character to plot
    penalty -- the penalty to use, e.g. 'l2'

    Returns: 4 plots, as described above.
    """
#     print("code_started")
    key = str([key for key, val in Kuzushiji_classes.items() if val == character][0])
    index = y_class(key, 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, stratify=y, random_state=0)
    C = [1e-4, 1e-1, 1e4]
    fig, axes = plt.subplots(1,len(C)+1,figsize=(16,4))
    for i, ax in enumerate(axes):
        if(i == 0):
            image = X.loc[index]
            label = y.loc[index]
            ax.imshow(image.values.reshape(28, 28), cmap=plt.cm.Blues)
            ax.set_xlabel(character)
            ax.set_xticks(()), ax.set_yticks(())
        else:
            model = sklearn.linear_model.LogisticRegression(solver='saga', max_iter=200, penalty=penalty, C=C[i-1], random_state=0)
            fit = model.fit(X_train, y_train)
            y_pred = fit.predict(X_test)
            score = sklearn.metrics.accuracy_score(y_test, y_pred)
            heatmap = ax.imshow(model.coef_[int(key)].reshape(28,28), cmap='plasma')
            ax.set_xlabel("C = {:.0e}, acc = {:.3f}".format(C[i-1], score))
            ax.set_xticks(()), ax.set_yticks(())
            cax = fig.add_axes([(i+1)/(5), 0.9, 0.1, 0.05])
            fig.colorbar(heatmap, cax=cax, orientation='horizontal')
            
            # time per model
#             %timeit fit = model.fit(X_train, y_train)
            
            # fraction of unused pixels (<10% of max weight)
#             values = model.coef_[int(key)].reshape(28,28)
#             values = [value for val in values for value in val]
#             values_abs = [abs(value) for value in values]
#             cutoff = 0.1 * (max(values))
#             print(sum(i < cutoff for i in values_abs) / (28*28))
    plt.show()
#     print("code_ended")

plot_character_coefficients(X, y, 'tsu', 'l2')
```


    
![png](README_files/README_28_0.png)
    


### Question 2.2: Sparse models (1 point)
Plot the coefficient again, but this time use a sparse logistic regression model. Keep everything else the same as in the previous question.


```python
# Implement. Do not change the name or signature of this function.
def plot_sparse_character_coefficients(X, y, character):
#     print("code_started")
    key = str([key for key, val in Kuzushiji_classes.items() if val == character][0])
    index = y_class(key, 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, stratify=y, random_state=0)
    C = [1e-4, 1e-1, 1e4]
    fig, axes = plt.subplots(1,len(C)+1,figsize=(16,4))
    for i, ax in enumerate(axes):
        if(i == 0):
            image = X.loc[index]
            label = y.loc[index]
            ax.imshow(image.values.reshape(28, 28), cmap=plt.cm.Blues)
            ax.set_xlabel(character)
            ax.set_xticks(()), ax.set_yticks(())
        else:
            model = sklearn.linear_model.LogisticRegression(solver='saga', max_iter=1, penalty='l2', C=C[i-1], random_state=0)
            fit = model.fit(X_train, y_train)
            y_pred = fit.predict(X_test)
            score = sklearn.metrics.accuracy_score(y_test, y_pred)
            heatmap = ax.imshow(model.coef_[int(key)].reshape(28,28), cmap='plasma')
            ax.set_xlabel("C = {:.0e}, acc = {:.3f}".format(C[i-1], score))
            ax.set_xticks(()), ax.set_yticks(())
            cax = fig.add_axes([(i+1)/(5), 0.9, 0.1, 0.05])
            fig.colorbar(heatmap, cax=cax, orientation='horizontal')
            
            # time per model
#             %timeit fit = model.fit(X_train, y_train)
            
            # fraction of ignored pixels (<10% of max weight)
#             values = model.coef_[int(key)].reshape(28,28)
#             values = [value for val in values for value in val]
#             values_abs = [abs(value) for value in values]
#             cutoff = 0.1 * (max(values))
#             print(max(values))
#             print(sum(i < cutoff for i in values_abs) / (28*28))
#     print("code_ended")

plot_sparse_character_coefficients(X, y, 'tsu')
```


    
![png](README_files/README_30_0.png)
    


### Question 2.3: Interpretation (1 points)
Interpret the graphs. Indicate which of the following answers are correct.

- 'A': For C=1e-4, the dense model is performing well. It clearly captures the shape of the character 'tsu' in its weights. Hence, it's paying attention to the right pixels.
- 'B': For C=1e-4, the dense model is underfitting. It can only capture one way of writing the character.
- 'C': For C=1e-4, the dense model is overfitting to a particular way of writing character 'tsu'.
- 'D': For C=0.1, the dense model is well-balanced, paying attention to different pixels to capture different ways of writing the character 'tsu'.
- 'E': For C=0.1, the weight are entirely random, hence it's clearly overfitting.
- 'F': For C=1e4, the dense model is well-balanced, paying attention to different pixels to capture different ways of writing the character 'tsu'.
- 'G': For C=1e4, the weight for the dense model are quite random and the model is likely overfitting.
- 'H': The sparse model with C=0.1 is a lot faster to train than the dense model with C=0.1.
- 'I': The sparse model with C=1e-4 ignores about half of the pixels but can still accurately predict the characters.
- 'J': The sparse model with C=0.1 ignores about half of the pixels but can still accurately predict the characters.
- 'K': The sparse model with C=1e4 ignores about half of the pixels but can still accurately predict the characters.
- 'L': The sparse model with C=1e4 isn't very sparse at all. It has about the same weights as the dense model
- 'M': Models with very high C values (more penalized) take much longer to train than models very low C values (more relaxed).
- 'N': Models with very high C values (more penalized) are much faster to train than models very low C values (more relaxed).
- 'O': None of these is correct

Enter the correct letter(s) in value `q_2_3` in the code. They may be separated by commas (e.g. `'A,B,C'`) if multiple answers are correct.


```python
# Fill in the correct answer(s). Don't change the name of the variable
q_2_3 = 'B,D,G'
```

### Question 3.1: Learning curve analysis (1 points)

Implement a method `learning_curve` that trains a Logistic Regression model trained using Stochastic Gradient Descent (SGD) on mini-batches of data and returns the training and validation accuracies after every batch. The curve that plots the model's scores after every new batch of training data is called a learning curve. Your implementation must also allow cycling multiple times over the training datasets (one pass over the training data is called an _epoch_).

The method must first make a (single) stratified 80-20 split of the input data (with `random_state=0`), where 80% forms the training data and the other 20% the validation data. It will then cycle `n_epoch` times over the training data. Every cycle, it shuffles the data, ensuring every epoch has a different shuffle of the data, and then creates batches of training data, where each batch consists of `batch_size` training examples. It must then _incrementally_ train the Logistic Regression model on every batch, i.e. continue training the model instead of re-initializing it, and after every batch it must compute the accuracy on the entire training and validation set. After `n_epoch` passes have been computed, it returns `train_accuracies` and `val_accuracies`, both vectors with the training and validation scores after each batch, respectively.

You'll have to use the [SGD Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html#sklearn.linear_model.SGDClassifier) to be able to do this in sklearn. Carefully consider how to use it so that it is equivalent to a Logistic Regression model, and that the regularization parameter is called `alpha`. This is an incremental learner: you can simply use `partial_fit` instead of the default `fit` to train the model incrementally. [See here](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html#sklearn.linear_model.SGDClassifier.partial_fit), and note that it requires the list of classes when first called. `SGDClassifier` also takes a hyperparameter `alpha` representing the regularization strength. For SGD you can use the `optimal` learning rate. This performs a kind of learning rate decay, starting with a larger learning rate and gradually making it smaller. With every batch, do a single optimization iteration. Use `random_state=0`.

To test the method and plot the result, we provide the simple `plot_learning_curves` method below. You must run the method on the entire (training) dataset, i.e. `X` and `y`. Use alpha=0.01 and run for 5 epochs.

Note: You'll have to implement most of this yourself. There is no handy method in sklearn that does this for you the way described here. However, we provide example code in `learning_curve_example` to show how to divide the data in batches and train SGD incrementally. Note that this code is very incomplete so you'll need to adapt it.




```python
# Implement. Do not change the name or signature of this function.
def learning_curve(X, y, alpha=0.01, n_epochs=1, batch_size=100):
    """ Trains a Logistic Regression model incrementally using stochastic gradient descent and returns the learning curves.
    X -- the data for training and testing
    y -- the correct labels
    alpha -- the regularization hyperparameter
    n_epochs -- the number of epochs
    batch_size -- the batch size
    Returns: Two arrays, with the training accuracies and validation accuracies, respectively, in that order.
    """
    model = sklearn.linear_model.SGDClassifier(alpha=alpha, max_iter=1, random_state=0, learning_rate='optimal', loss='log_loss')
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.8, stratify=y, random_state=0)
    classes = np.unique(y)
    train_scores, val_scores = [], []
    
    for i in range(n_epochs):
        X_batch, y_batch = sklearn.utils.shuffle(X_train, y_train, n_samples=batch_size, random_state=0)
        model.partial_fit(X_batch, y_batch, classes=classes)
        train_scores.append(sklearn.metrics.accuracy_score(y_train, model.predict(X_train)))
        val_scores.append(sklearn.metrics.accuracy_score(y_val, model.predict(X_val)))
    
    return train_scores, val_scores

def plot_learning_curves(train_accuracies, val_accuracies, title='Learning Curve Across Epochs'):
    """ Plots the given learning curves. """
    if len(train_accuracies) > 0:
        plt.figure(figsize=(10, 6))
        plt.plot(train_accuracies, label='Training Accuracy')
        plt.plot(val_accuracies, label='Validation Accuracy')
        plt.xlabel('Number of Batches')
        plt.ylabel('Accuracy')
        plt.title(title)
        plt.grid(True)
        plt.legend()
        plt.show()

plot_learning_curves(*learning_curve(X, y, alpha=0.01, n_epochs=5), title="Learning curve on pixel data, 5 epochs, and alpha=0.01")
plot_learning_curves(*learning_curve(X_embedded, y, alpha=0.01, n_epochs=5), title="Learning curve on embedded data, 5 epochs, and alpha=0.01")
```


    
![png](README_files/README_34_0.png)
    



    
![png](README_files/README_34_1.png)
    



```python

```

### Question 3.2: Interpretation (1 points)

Interpret the graphs. Indicate which of the following answers are correct.  
*Note*: Feel free to run additional experiments if you feel that you need them.

- 'A': The SGD approach performs much better than the Logistic Regression model used in Question 1, given the same training data.
- 'B': The SGD approach performs about the same as the Logistic Regression model used in Question 1, given the same training data (if there is a difference it is less than 5% accuracy).
- 'C': The learning curve on the embedded data is steeper (learns faster). E.g. after 50 batches it has already surpassed the learning curve of the pixel data.
- 'D': The learning curve on the embedded data matches the one on the pixel data. Only in the last epoch does it surpass the learning curve of the pixel data.
- 'E': The learning curves for the validation accuracy go down. That means that both models are overfitting.
- 'F': Both learning curves flatten out after a few epochs. It's not worth training for much longer.
- 'G': None of these are correct


```python
# Fill in the correct answer. Don't change the name of the variable
q_3_2 = 'B,C'
```

## Question 4 (1 points)

Archeologists found an ancient text in the mountain cave temples of Shodoshima, which is partially decifered into the following sentence:
***
THE **<word 1>** LOOKS BEAUTIFUL OVER THE **<word 2>**
***

They need your help to uncover the meaning of the missing characters and expand our knowledge on early Japanese literature.

They sent us pictures of the missing characters in the form of 28x28 numpy arrays (because why not). The code below imports them into the notebook. Both words consist of two characters as shown below. The first two characters form the first word and the last two form the second.


```python
# !pip install gdown # uncomment to install the downloader
url = 'https://drive.google.com/uc?id=1CcnG1a6feMd7n8rOph_nSBf29ltuJ9no'
output = 'mystery_characters.npy'
gdown.download(url, output, quiet=False)
temple_data = pd.DataFrame(np.load('mystery_characters.npy'))
plot_examples(temple_data[0:2], None, row_length=2, title="Word 1")
plot_examples(temple_data[2:], None, row_length=2, title="Word 2")
```

    Downloading...
    From: https://drive.google.com/uc?id=1CcnG1a6feMd7n8rOph_nSBf29ltuJ9no
    To: c:\Users\shoot\OneDrive - TU Eindhoven\Data Science\Portfolio\Machine Learning - Python -\mystery_characters.npy
    100%|██████████| 25.2k/25.2k [00:00<?, ?B/s]
    


```python
plot_examples(temple_data[0:2], None, row_length=2, title="Word 1")
plot_examples(temple_data[2:], None, row_length=2, title="Word 2")
```


    
![png](README_files/README_40_0.png)
    



    
![png](README_files/README_40_1.png)
    


Use the best model that you trained on the embeddings in question 1 to classify these images. Implement a small method `predict_characters` that prints the right classes (e.g. 'tsu') given the character values. Note that you may have to embed the new characters in order to do so (this will be fast since it's only a few characters).

Once you have translated them into modern Japanese, translate them to English and type the two words in the variables q_4_word_1 and q_4_word_2.

Hint: You can use ChatGPT or another chatbot to translate for you if you don't know Japanese. Enter the words without spaces between the characters. There may be multiple meanings for a word, you can pick the one that fits the sentence best. If you're unsure, check the image with the Japanese text in the beginning of this assignment. It may give you a hint :).


```python
# Implement. Do not change the name or signature of this function.
def predict_characters(X, y, X_test):
    """ Print the class names for all the images in X.
    X -- the data for training and testing
    y -- the correct labels
    X_test -- the new input images as 1D arrays

    Returns: an array with the classified characters
    """
#     print("code_started")
    model = sklearn.linear_model.LogisticRegression(C=0.01, solver='lbfgs')
    model.fit(X, y)
    prediction = model.predict(create_embedding(X_test))
    characters = [Kuzushiji_classes[int(i)] for i in prediction]
#     print("code_ended")
    return characters

characters = predict_characters(X_embedded, y, temple_data)
```


      0%|          | 0/1 [00:00<?, ?it/s]



```python
print("The sentence is : The '{}' looks beautiful over the '{}'.".format("".join(characters[0:2]),"".join(characters[2:4])))
```

    The sentence is : The 'tsuki' looks beautiful over the 'yama'.
    


```python
# Fill in the correct meaning
q_4_word_1 = "moon"
q_4_word_2 = "mountain"
```
