# stock-predictor

Using various machine learning techniques to classify good companies vs bad ones

1. Statement of problem to be solved.

The problem to be solved is Quantitative Value Investing. For a long time, value investors have been using ratios to determine the value of companies.
These ratios were then compared with other companies to determine which would be the best investment and which would overperform and underperform a single index.
However analysing those ratios can be a long and arduous process. As a group we decided to make this process faster and determine how do those ratios look
when a company overperforms and when a company underperforms and train our model to recognise them.

2. Loss function and performance measure.

The loss function is the standard loss function of the svm, which is going to be the main driver of this research

The train data is going to be from the 2017 fiscal year as well as the status (overperformance/underperformance).

We will use f1 score as the performance measure.

3. Data sources.

All data sources are going to be retrieved from Bloomberg. The companies that are going to be selected are the companies
that exist on the NYSE and NASDAQ indices.

4. Feature engineering.

The ratios that are going to be selected are the most widely used for the value investing analysis as well as ratios that
have the highest availability from bloomberg for the selected companies.

5. Naive method.

Random selection.
OLS

6. Machine learning methods
Regression
Support Vector Machines
