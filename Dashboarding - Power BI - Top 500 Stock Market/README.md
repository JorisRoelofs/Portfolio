# Kuzishiji Handwriting Recognition
***Skills***_: Dashboarding, Power BI, Finance

## Problem Overview
DEGIRO is commonly used in the Netherlands for buying and selling stocks. Nevertheless, its dashboard has is severly limited in what metrics it shows. It only shows an overview of the current prices and daily changes, lacking more advanced and insightful metrics such as volatility or MACD (moving average convergence/divergence). Furthermore, it does not provide comparisons between different indsutries or regions, and news articles are not fitlered by selection. As such, it difficult to compare stocks on DEGIRO's dashboard and thus make wise investment choices.

## Solution
I designed a more advanced dashboard in Power BI as a personal study . It expands on DEGIRO's dashboard by showing the same data as well as aditional metrics and filters. Key features include:

## Key Features
- **Advanced Metrics**: Calculates volatility, MACD (moving average convergence/divergence), RSI (relative strength index), 90-day trend line, and sales volume peaks within the specified period and filters.
- **Expanded Filtering**: Stock can be filtered by sector, industry, country of origin, and stock market, with each category providing average trendlines to compare stocks against. Furthermore, news articles are automatically filtered on the selection to learn the most relevant news.
- **Easy Sorting**: Stocks can be sorted on all metrics to determine e.g. the least and most volatile stocks within a category.
- **Currency Conversion**: Switch the graph and tables between € and $, as well as daily % changes to better compare volatility between stocks.

## Data Sources
- **Top 500 Stock Market**: 5 years of daily Yahoo Finance stock market info of the top 500 companies by market gap (602,693 rows), from [Kaggle](https://www.kaggle.com/datasets/iveeaten3223times/massive-yahoo-finance-dataset/data)
- **Company Info**: company registration details of 159,514 equity stocks, from [Finance Database](https://github.com/JerBouma/FinanceDatabase?utm_source=chatgpt.com)
- **US➔EU Conversion**: currency conversion history (01-01-1991 ➔ 14-10-2025), from [European Central Bank](https://data.ecb.europa.eu/data/datasets/EXR/EXR.D.USD.EUR.SP00.A)
- **Financial News**: , from [HuggingFace](https://huggingface.co/datasets/Zihan1004/FNSPID)

## Dashboard
### Market Overview
<p align="center">
  <img src="Visuals/Stock Market Dashboard - Overview.pdf">
</p>
*Figure 1: Dasboard showing all top 500 stocks in a sortable table and a graph with their average trendline.*

### Help Function
<p align="center">
  <img src="Visuals/Stock Market Dashboard - Overview Help.pdf">
</p>
*Figure 2: Pressing the help button explains what all buttons do (in white) and what each table and graph shows (in light grey).*

### Sector Filtering
<p align="center">
  <img src="Visuals/Stock Market Dashboard - Sector.pdf">
</p>
*Figure 3: Top 500 stock market dashboard, with tables showing metrics and news for each fossil fuel companies and the graph showing the average stock prices in the fossil fuel indsutry.*

### Overview
<p align="center">
  <img src="Visuals/Stock Market Dashboard - Single Stock.pdf.pdf">
</p>
*Figure 4: Top 500 stock market dashboard, filtered on a single stock.*