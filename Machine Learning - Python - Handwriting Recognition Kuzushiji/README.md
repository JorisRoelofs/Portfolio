# Kuzishiji Handwriting Recognition
***Skills***_: Machine Learning, Python (Scikit-Learn, TensorFlow, Matplotlib, NumPy, Pandas)_

## Problem Overview
For over a thousand years Kuzushiji script dominated Japanese writing, until it was removed from school curriculum in 1900. As a result, vast collections of historical manuscripts, letters, and books remain inaccessible to the modern Japanese public. Digitization efforts are ongoing, but face a significant challenge: each character can be written in hundreds of stylistic variations depending on region, historical period, and individual handwriting. **This leaves many texts incomprehensible to all but expert linguists, and machine models.**

## Solution
I developed a machine learning pipeline capable of recognizing commonly used handrwritten Kuzishiji characters. Key features include:
- **Bayesian hyperparameter optimization**: Using Optuna, the pipeline efficiently determines the best hyperparameter values for each model.
- **Multimodel evaluation**: Five classical machine learning models (Scikit-Learn) and three deep learning models (TensorFlow) are automatically trained, graphed and compared.
- **Dual data representations**: Both raw pixel data and high-dimensional vector embeddings are supported, allowing each model to use the data representation it suits most.

## Key Results
- Achieved **>90% validation accuracy** with 7 out of 8 models through automatic hyperparameter tuning and vector embeddings.
- Achieved **95.1% validation accuracy** using a Convolutional Neural Network, trained to recognize nearly 7.000 variations of each character.
- Visualized **confusion matrices**, confirming balanced class performance in all models.

## Skills Demonstrated
- **Machine Learning (Scikit-Learn)**:
    - Logistic Regression
    - k-Nearest Neighbor
    - Decision Tree
    - Random Forest
    - Gradient Boosting
- **Deep Learning (TensorFlow)**:
    - Stochastig Gradient Descent
    - Multilayer Perceptron
    - Convolutional Neural Network
- **Hyperparameter Autotuning (Optuna)**:
    - Bayesian Optimization
- **Data Processing**
    - Pandas, Numpy
    - Vector embeddings
- **Data Visualization**
    - Bar charts
    - Learning curves
    - Confusion matrices
    - Decision weight heatmaps
    - t-SNE dimensionality collapse

## Data Source
- **Kuzushiji-MNIST**: 70.0000 handwritten images (28x28 grayscale) of the 10 most common Kuzushiji characters in a variety of handwritings, from [Kaggle](https://www.kaggle.com/datasets/anokas/kuzushiji).

## Data Exploration
<p align="center">
  <img src="Visuals\Visuals_7_0.png">
</p>

*Figure 1. Each Kuzushiji character features a vast variety of representations.*

### Pixel Weight Heatmaps
<p align="center">
  <img src="Visuals/Visuals_19_1.png">
</p>

*Figure 2. Left: Examples of the character 'tsu'. Right: Pixel weight heatmaps for 'tsu', from three Logistic Regression models with different regularization values.*


Pixel models are highly dependent on the data being normalized on centering, rotation, and scaling. As the dataset is normalized and noiseless it is likely to perform well, but small variations might still lead to vastly different classifications. As shown in Figure 2, weights for a Logistic Regression model differ substantially between pixels, with large differences in median and mean weight. This suggests that using vector features derived from the pixel data (e.g. shapes, curviness, slanting) as vector features is still likely to increase robustness for many models. As such, vector embeddings were created of the pixel data, and embedding support was implemented into the machine learning pipeline for each model but the Convolutional Neural Network (CNN), which is designed specifically for spatial data such as pixel images.

### t-SNE Dimensional Collapse
<p align="center">
  <img src="Visuals\Visuals_11_0.png">
</p>

*Figure 3. 2D representation of the 1280 vector dimensions from the embedding data representation.*

Figure 3 shows that high-dimensional embeddings form clear, distinguishable clusters for each class, albeit with some overlap. Each of the models was tested with both embeddings and pixel data as input.

## Results
| Model | Data | Eval | Val | Hyperparameters |
|--------|----------------|-------------|-------------|-----------------|
| Convolutional Neural Network | Pixel | **95.1%** | 95.4% | learning_rate = 0.001, l2_reg = 0.0005, n_epochs = 100, batch_size = 128 |
| Nearest Neighbors | Pixel | **93.8%** | 93.7% | n_neighbors = 1 |
| Nearest Neighbors | Embed | 90.2% | 90.5% | n_neighbors = 7 |
| Multilayer Perceptron | Embed | **92.6%** | 92.8% | learning_rate = 0.0001, l2_reg = 0.0001, n_epochs = 95, batch_size = 32 |
| Multilayer Perceptron | Pixel | 92.1% | 92.3% | learning_rate = 0.005, l2_reg = 1e-05, n_epochs = 75, batch_size = 160 |
| Histogram Gradient Boosting | Pixel | **91.7%** | 91.9% | max_depth = 16, learning_rate = 0.1, l2_regularization = 0.05 |
| Histogram Gradient Boosting | Embed | 90.5% | 90.7% | max_depth = 11, learning_rate = 0.1, l2_regularization = 0.01 |
| Stochastic Gradient Descent | Embed | **91.4%** | 91.4% | learning_rate = 0.0005, l2_reg = 0, n_epochs = 30, batch_size = 64 |
| Stochastic Gradient Descent | Pixel | 78.8% | 79.5% | learning_rate = 0.001, l2_reg = 0.0005, n_epochs = 85, batch_size = 192 |
| Logistic Regression | Embed | **91.1%** | 90.9% | C = 0.05 |
| Logistic Regression | Pixel | 79.2% | 79.3% | C = 0.05 |
| Random Forest | Pixel | **90.5%** | 90.7% | max_depth = None, n_estimators = 170 |
| Random Forest | Embed | 86.0% | 86.6% | max_depth = None, n_estimators = 190 |
| Decision Tree | Pixel | **67.2%** | 67.6% | max_depth = 15 |
| Decision Tree | Embed | 58.0% | 58.4% | max_depth = 19 |

*Table 1. Most performative model of each model type and data representation, sorted by evaluation performance.*

As shown in Table 1, the Convolutional Neural Network (CNN) achieved the highest validation and evaluation accuracies. As CNN's are specifically designed to work with spatial data this does not come as a surprise, as it is able to make the most of the spatial pixel information. Unfortunately, for the same reason it is the only model that could not naturally handle embedded data. The details of its results are shown below, with graphs of the other seven models available in the 'Handwriting Recognition Kuzushiji.ipynb' file or 'Visuals' folder.


<p align="center">
  <img src="Visuals/Visuals_27_376.png">
</p>

*Figure 4. Learning curve of the Convolutional Neural Network, trained on a pixel representation of the data.*

Figure 4 shows that the CNN model with optimized hyperparameters already outperforms most models in its first few epochs, after which it slowly keeps ascending until it reaches the maximum epoch count (due to long trainig times). Its 100% training accuracy suggest that it learns detailed structural patterns. The occasional downwards spikes are likely caused by batch variability, given that the validation drop mimics the test drop and that validation and evaluation scores are high. As such, per-class validation accuracy is plotted below to determine if there are any class imbalances.

<p align="center">
  <img src="Visuals/Visuals_27_374.png">
</p>

*Figure 5. Bar chart showing overall and per-class validation accuracy of the Convolutional Neural Network, trained on a pixel representation of the data.*

Figure 5 shows minor but not insubstantial class impalances, with classes ranging between 92.2% and 97.4% validation accuracy. The confusion matrix below sheds light on why the accuracy of 'na' is lower than other classes.


<p align="center">
  <img src="/Visuals/Visuals_27_373.png">
</p>

*Figure 6. Confusion matrix showing per-class classification of the Convolutional Neural Network, trained on a pixel representation of the data.*

As shown in Figure 6, 'na' is occasionally misclassified as 'o', and vice versa. This pattern appears across other models as well and is understandable, since some variants of 'o' resemble the left portion of certain 'na' representations. Fortunately, these misclassifications remain rare, occurring in only 2.5% of 'na' cases and 1.6% of O cases. Nevertheless, they likely explain the spikes in the CNN learning curve, as a batch with nine 'na'-'o' misclassifications can already drop performance by the 7% shown.