# Deep Reinforcement Learning Robot Advisor for Portfolio Management
This project aims at developing a robot advisor to grow wealth with ETF investment through continuous learning and rational decision-making. As an ordinary employee without a large amount of assets and professional investment knowledge, I hope it helps manage my portfolio on its own. Modern Portfolio Theory which some robot advisor providers adopt has limitations in real life application. For example the assumptions that normally distributed asset returns, same information available to all investors or investors holding same view of expected return, are not true in the complex reality. It is an attempt to let the robot advisor learn itself by reinforcement learning. It takes advantage of reinforcement learning model which makes consistent decision in order to optimize the target, the total return or final portfolio value, by observing and learning gradually from real word situation with minimal predefined assumptions.

## Result and Discussion
### Portfolio Performance Summary
*** *Updated with model trained and tested on Oct 30, 2021*  ***

Back-test Period: From 2015-01-02 to	2020-12-31

&nbsp; | Details 
--- | --- 
**Annual return** | 17.836%
**Cumulative returns** | 167.534%
**Annual volatility** | 21.585%
**Sharpe ratio** |	0.87
**Sortino ratio** |	1.21
**Alpha** |	0.04
**Beta** |	1.08

In back-testing, the portfolio value grows 2.7 times in 6 years with annual return 18% and volatility 22%, which performed better than the benchmark index (SPY) but also riskier. The Sharpe ratio is 0.87 on average, which is smaller than 1, and fluctuated from -2 to 4 during the back-test period while the Sortino ratio is 1.21, which is more than 1, and means the downside risk-adjusted return is acceptable. It performed the best in 2019 and worst in 2018 and fluctuates in 2020. Though experiencing a large drop in 2020, it recovered faster than the benchmark and had the largest growth afterwards. The aims is to grow wealth through continuous learning and rational decision-making so looking at the allocation along the time shows more about how it made decisions.

![alt text][cumula_ret]
![alt text][pyfolio_tear_sheet_rolling]

More details can be found in [Pyfolio generated tear sheet](https://github.com/kftam1994/Robo_Advisor/blob/main/images/pyfolio_tear_sheet.png)

### Portfolio's Allocation
The model chose different composition of ETF after observing different portfolio performance to cater for the market condition. During the back-test period, XLK and XLB are the major ETF chosen and occupied a large proportion of allocation.
In 2016 and 2019, the model chose XLK most of the time. In 2015 and 2017, the model chose on average one fifth to one third of weights for XLB and the rest XLK. In 2018 and 2020, the weights for XLB and XLK were fluctuating and the weight for XLB rose to half of weights in Jan 2018 and drop to less than 5% in some months of 2020.
But it is only an observation of how the model had performed. Therefore, Integrated Gradients is also performed to describe which feature the model was focusing when it made the decision.

![alt text][port_weights]

![alt text][roll_vol_vs_port_maj_weight]

### Model Interpretation-Integrated Gradients
Integrated Gradients is an axiomatic attribution method adopted to interpret the model and understand which features lead to each decision and which factors the model focus on. From a baseline input, usually all zero, it generates a linear interpolation between the baseline and the original input step-by-step, computes the gradients of the model output with respect to each input feature in the neural network and computes the numerical approximation for the integral of those gradients by Gauss-Legendre quadrature rule.

Apart from prices, the model made decision to choose XLK and XLB due to the economic condition and price momentum and volatility of the XLK and XLB.

#### XLK:
In 2015, the model focused on average true range, unemployment rate and monetary base. It also looked at relative strength indicator during mid 2015. In 2016, the only focused feature was the relative strength indicator. During 2017 and 2018, M2 money supply, monetary base and unemployment rate were the target features. In 2019 and afterwards, it returned to average true range and relative strength indicator.

![alt text][XLK_overall]

From the distribution of aggregated integrated gradients values for each feature of the [previous day/1st day in the window of data](#note-about-focusing-on-analyzing-the-1st-day-in-the-window-of-data), high price, moving average of prices and average true range, and economic indicators including unemployment rate, job opening, industrial production, monetary base were the major features driving the decision for choosing XLK.

![alt text][XLK_features_1]

#### XLB:
Similar to XLK, for 2015, the model also looked at average true range and relative strength indicator. The unemployment rate and monetary base were considered too. From 2016 to 2020, relative strength indicator and average true range sometimes were the main feature the model focused. It took 10-year treasury into consideration in 2018.

A note is that the sign of integrated gradients values of relative strength indicator and average true range for XLK (negative on average) and that for XLB (positive on average) were reversed.

![alt text][XLB_overall]

From the distribution of aggregated integrated gradients values for each feature, it is similar to XLK that high price, moving average of prices and average true range, and economic indicators including unemployment rate, job opening, industrial production, monetary base were the major features driving the decision for choosing XLB.

![alt text][XLK_features_1]

It may indicate that the model considered the long term growth signified by the economic indicators during stable and growth market condition and paid close attention to price volatility and momentum during volatile market. 2017 and 2018 were years of relatively higher GDP growth and Federal Funds Rate was rising. 2015 and after 2019 were also period of fluctuating stock market due to QE, trade war, Brexit and then CoVID-19 pandemic. In addition, the different of sign of integrated gradients values for XLK and XLB may explain the switches to XLB during uncertainties in market and XLK for benefiting from fastest growth in stable market.

#### Note about Focusing on Analyzing the 1st day in the Window of Data
The reason for focusing on the 1st day only out of the whole window of 32-day data is that the aggregated Integrated Gradients values are the most significant for the 1st, 2nd and 3rd day in the window, which means the model placed more emphasis on the features in these days compared to that in other days. Although it does not mean data in other window days were not important, they were less significant when it comes to affecting the decision of the model. Therefore, the above interpretation is mainly on the first day of the whole window.

![alt text][Integrated_Gradients_each_Window]

## Methodology
Referencing to two papers, A Deep Reinforcement Learning Framework for the Financial Portfolio Management Problem ([arXiv:1706.10059](https://arxiv.org/abs/1706.10059) and [Github](https://github.com/ZhengyaoJiang/PGPortfolio)) and Adversarial Deep Reinforcement Learning in Portfolio Management ([arXiv:1808.09940](https://arxiv.org/abs/1808.09940) and [Github](https://github.com/liangzp/Reinforcement-learning-in-portfolio-management-)), this project applies the following methodologies to manage a portfolio of ETF investment.

•	Ensemble of Identical Independent Evaluators

•	Portfolio-Vector Memory

•	Online Stochastic Batch Learning

•	Adversarial Learning

•	Risk Penalty

and the following are the differences in implementation with the papers.

1.	Disable OSBL in training

The parameter of geometric distribution is so small that it is similar to drawing batches from a uniform distribution

2.	Extra Features

Apart from High, Low, Close, Open prices history, macroeconomic indicators and technical analysis indicators are also considered as features to describe context.

3.	Roll Train

During backtesting, roll training is implemented after each new day of observation with a relatively small number of steps/epochs compared to full training

4.	Different network structure & Pytorch Implementation

The neural network structure is different from the papers. The feature maps are increased and then reduced, compared to reduction and then increase in the paper, to firstly generate variety of interpretation for each different feature and then summarized the knowledge. Batch normalization is added to increase stability of training. A fully connected layer is added before softmax function to learn the relationship between cash and portfolio assets before outputting prediction of weights.
![alt text][nn_structure]


## Data and Features
I.	Stock Data

Stock data are daily historical price data retrieved from Yahoo Finance (through yahoofinancials package). The period is from 1990 to 2020 though only 2005 to 2020 are used in training and back-testing. Stock data include the high, low, open, close and volume. Beta and alpha are calculated by regressing return against benchmark return (SPY).

II.	Stock Measurement of Performance and Technical Analysis Indicators

Stock Measurement of Performance and Technical analysis indicators are calculated from daily historical price data. 
Stock Measurement of Performance includes Alpha and Beta relative to SPY, which is a proxy of market performance, for 2-year and 5-year period to provide information about return and volatility compared to market.
Technical analysis data are included to provide additional information about price trends and patterns, and potentially about the future movements. It includes 

1.	Moving Average of 50-day and 200-day windows

3.	Exponential Moving Average (EMA) for 50-day and 200-day windows

5.	Relative Strength Indicator (RSI) for 14-day period

7.	Average True Range (ATR) for 14-day period

III.	Macroeconomic Indicators

Macroeconomic indicators are retrieved from Federal Reserve Economic Data in the St. Louis FED through Pandas FredReader. They are included to provide a background of current Macroeconomic situation in the market. Most of them describe the US market but one describes the global economic growth (GEPUCURRENT). The Macroeconomic indicators include

US Treasury

1.	DGS1MO: 1-Month Treasury Constant Maturity Rate

3.	DGS3MO: 3-Month Treasury Constant Maturity Rate

5.	DGS1: 1-Year Treasury Constant Maturity Rate

7.	DGS3: 3-Year Treasury Constant Maturity Rate

9.	DGS10: 10-Year Treasury Constant Maturity Rate

11.	T10Y2Y: 10-Year Treasury Constant Maturity Minus 2-Year Treasury Constant Maturity

US Inflation

7.	CPALTT01USM657N: Consumer Price Index: Total All Items for the United States

9.	DLTIIT: Treasury Inflation-Indexed Long-Term Average Yield

11.	T5YIFR: 5-Year Forward Inflation Expectation Rate

13.	T10YIE: 10-Year Breakeven Inflation Rate

15.	MICH: University of Michigan: Inflation Expectation

Money Supply

12.	WM1NS: M1 Money Stock

14.	WM2NS: M2 Money Stock

16.	BOGMBASE: Monetary Base

Employment

15.	UNRATE: US Unemployment Rate

17.	JTSJOL: Job Openings: Total Nonfarm

19.	IC4WSA: 4-Week Moving Average of Initial Claims

Production and Consumption

18.	INDPRO: Industrial Production: Total Index

20.	MRTSSM44X72USS: Retail Sales: Retail Trade and Food Services

22.	UMCSENT: University of Michigan: Consumer Sentiment

Market Sentiment

21.	VIXCLS: CBOE Volatility Index: VIX

23.	STLFSI2: St. Louis Fed Financial Stress Index

Commodity Price

23.	DCOILBRENTEU: Crude Oil Prices: Brent - Europe

25.	GOLDAMGBD228NLBM: Gold Fixing Price

Global Growth

25.	GEPUCURRENT: Global Economic Policy Uncertainty Index: Current Price Adjusted GDP (a GDP-weighted average of national EPU indices for 20 countries)

## ETF Pre-selection
32 ETF listed in US are selected to form the portfolio. Reinforcement learning model decides the weights to be allocated to 32 ETF and cash. ETF belonging to different categories, including size, strategy, sector, bond, REITS and regions, are chosen. Apart from SPY, other ETF are chosen as they are those with largest total asset in each category. Most of them were incepted between 1990 and 2005, except leveraged ETF which was firstly issued in 2006.

## Credits & Reference
Yahoo Finance

FRED, Federal Reserve Bank of St. Louis

Z. Jiang, D. Xu, and J. Liang, “A deep reinforcement learning framework for the financial portfolio management problem,” arXiv, arXiv:1706.10059, 2017.

Z. Liang, H. Chen, J. Zhu, K. Jiang, and Y. Li, “Adversarial Deep Reinforcement Learning in Portfolio Management,” arXiv, arXiv:1808.09940, 2018.

## License
License is following Zheng's [Github](https://github.com/ZhengyaoJiang/PGPortfolio) as some codes are re-written based on his.

[nn_structure]: https://github.com/kftam1994/Robo_Advisor/blob/main/images/neural_network_structure.png "neural network structure"
[roll_vol_vs_port_maj_weight]: https://github.com/kftam1994/Robo_Advisor/blob/main/images/roll_vol_vs_port_maj_weight.png "rolling volatility vs portfolio major weights"
[port_weights]: https://github.com/kftam1994/Robo_Advisor/blob/main/images/port_weights.png "portfolio weights"
[cumula_ret]: https://github.com/kftam1994/Robo_Advisor/blob/main/images/cumula_ret.png "portfolio cumulative return"
[pyfolio_tear_sheet_rolling]: https://github.com/kftam1994/Robo_Advisor/blob/main/images/pyfolio_tear_sheet_rolling.png "rolling volatility & sharpe"
[XLK_overall]: https://github.com/kftam1994/Robo_Advisor/blob/main/images/XLK_overall.png "XLK Integrated Gradients"
[XLK_features_1]: https://github.com/kftam1994/Robo_Advisor/blob/main/images/XLK_features_1.png "XLK Integrated Gradients aggregated"
[XLB_overall]: https://github.com/kftam1994/Robo_Advisor/blob/main/images/XLB_overall.png "XLB Integrated Gradients"
[XLB_features_1]: https://github.com/kftam1994/Robo_Advisor/blob/main/images/XLB_features_1.png "XLB Integrated Gradients aggregated"
[Integrated_Gradients_each_Window]: https://github.com/kftam1994/Robo_Advisor/blob/main/images/Integrated_Gradients_each_Window.png "Integrated Gradients each window"
