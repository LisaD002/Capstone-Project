#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Importing Relevant Packages 


# In[208]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import statsmodels.graphics.tsaplots as sgt 
import statsmodels.tsa.stattools as sts
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
from scipy.stats.distributions import chi2
from math import sqrt
import seaborn as sns 
sns.set()


# In[ ]:


#Importing the Data and Pre-processing 


# In[209]:


raw_csv_data= pd.read_csv("C:/Users/lisad/OneDrive/Documents/CIND820/COVID-19_Diagnostic_Laboratory_Testing__PCR_Testing__Time_Series (3).csv")
df_comp = raw_csv_data.copy()
df_comp.date = pd.to_datetime(df_comp.date, dayfirst = True)
df_comp.set_index("date", inplace = True)
df_comp= df_comp.fillna(method= 'ffill')


# In[ ]:


#Examining the Data


# In[210]:


df_comp.head()


# In[211]:


#data- values recorded
# analyze time series in cosecutive chunks of data
# from one data to another 
# dates are used as indexes/indices for time series
# next 8 columns - time series data of state, state_name, region and numerical values for new resutls reported and total results reported. geocoded state - NAN
# new results reported and total results reported - number of daily PCR tests conducted


# In[212]:


#check data types
df_comp.dtypes


# In[213]:


df_comp.shape


# In[214]:


#Check columns
df_comp.columns


# In[215]:


#Check for NA values
df_comp.isnull().sum()


# In[216]:


df_time = df_comp[['new_results_reported', 'total_results_reported']].dropna()
df_time.head()


# In[217]:


df_time.loc['2020-03-01':'2020-12-31'].head(20)


# In[218]:


df_time.loc['2020-03-01':'2020-12-31'].count()


# In[219]:


df_time.loc['2021-01-01': '2021-12-31'].head(20)


# In[220]:


df_time.loc['2021-01-01': '2021-12-31'].count()


# In[221]:


df_time.loc['2022-01-01': '2022-10-27'].head(20)


# In[222]:


df_time.loc['2022-01-01': '2022-10-27'].count()


# In[223]:


df_time.loc['2020-03-01':'2020-12-31'].describe()


# In[224]:


df_time.loc['2021-01-01': '2021-12-31'].describe()


# In[225]:


df_time.loc['2022-01-01': '2022-10-27'].describe()


# In[226]:


df_time.resample('M').mean().head(10)


# In[227]:


df_time.resample('M').median().head(10)


# In[228]:


df_time.resample('M').min().head(10)


# In[229]:


df_time.resample('M').max().head(10)


# In[230]:


df_time.resample('M').std().head(10)


# In[231]:


df_comp.describe()


# In[232]:


df_comp["PCR_test"] = df_comp.new_results_reported


# In[ ]:


#Splittine the Data


# In[233]:


del df_comp["state"]
del df_comp["state_name"]
del df_comp["state_fips"]
del df_comp["fema_region"]
del df_comp["overall_outcome"]
del df_comp["geocoded_state"]
del df_comp["total_results_reported"]


# In[234]:


size = int(len(df_comp)*0.8)
df, df_test = df_comp.iloc[:size], df_comp.iloc[size:]


# In[235]:


# int = ensures that size will be an integer -serves as an approximation of the 80% cutoff point of the dataset
# after determining when the train should end and the test should - use the iloc method
# training set  = "df"
# testing set = " df_test"
# df - assign from beginning (start) up to size value


# In[236]:


df =  df_comp.iloc[:size]


# In[237]:


df_test = df_comp.iloc[size:]


# In[238]:


df.tail()


# In[240]:


df_test.head()


# In[241]:


wn = np.random.normal(loc= df.PCR_test.mean(), scale = df.PCR_test.std(), size = len(df))


# In[242]:


df.describe()


# In[243]:


df_comp.new_results_reported.plot(figsize =(20,5),title = "New Resutls Reported for PCR Testing")
plt.show()


# In[244]:


# White noise in Time series 
# def white nose is a special type of time-series, where the data doesn't follow a pattern
# Assumptions of TS data- patterns found in the past also persist in the future
# no patterns found - unpredictable
# consider patterns of white nose 
# 1. constant mean 
# 2. constant variance
#3. no autocorrelation in any period
# autocorrelation measures how correlated a series is with past versions of itself 
# NO autocorrelation = NO clear relationship between past and present value 
#White noise is sequence of random data, where every value has a time-period associated with it
# white noise behaves sporatically so there is no way to project it into the future 
# in financial modeling it is important to distinguish  whtie noise data from regular time series data
# differentiate the two by comparing their graphs 
# generate white noise data and plot its values
# plot the graph of the S &P closing prices and compare the two 


# In[245]:


df["wn"]= wn


# In[246]:


df.describe()


# In[247]:


df.wn.plot(figsize = (20,5))
plt.title("White Noise in Time Series", size = 24)
plt.show()


# In[248]:


# no clear pattern in data except for most values forming around the mean
# see how many values are within some proximity of the mean of PCR_test 


# In[249]:


df.PCR_test.plot(figsize= (20,5))
df.wn.plot(figsize = (20,5))
plt.title("White Noise vs New Results Reported in PCR testing", size = 24)
plt.ylim(0, 500000)
plt.show()


# In[250]:


sts.adfuller(df.PCR_test)


# In[ ]:


#output of Dickey Fuller test
# 1. test statistic- compare it to certain critical values to determine if we have significant proof of stationarity
# python provides us with the 1, 5 and 10% critical values for the dickey fuller table 
# use any of them as level of significance in our analysis
# t-stat is greater than any of the critical values 
# for all of these levels of significance, we do not find sufficient evidence of stationarity in the dataset 
# 2. p-value -  associated with t-stat
# 40% chance of not rejecting the null - can't confirm that the data is stationary 
# 3. # of lags in the regression when determining t-stat - autocorrelation going back 18 periods ** used to determine proper model 
# 4. number of observations used in the analysis - depends on the number of lags used in the regression - two should add up to the size of the dataset 
# 5. maximized information criteria provided- there is some autocorrelation
# the lower the values, the easier it is to make predictions for the future 


# In[60]:


sts.adfuller(df.wn)


# In[ ]:


# no autocorrelation in white noise - no lags involved in the regression
# a p-value close to 0 and no lags being part of the regression


# In[65]:


import scipy.stats
import pylab


# In[66]:


scipy.stats.probplot(df_comp.new_results_reported, plot= pylab)
pylab.show


# In[266]:


#test, then explore plot
# QQ plot takes all the values a variable can take and arranges them in ascending order
# y axis- New Results Reported
# x axis- theoretical quantile - how many standard deviations away from the mean these values are
# red diagonoal line- what the data points should follow if they are normally distributed
# not normally distributed - more values on 500 mark
# split data into training and test set to use machine learning to forecast the future


# In[68]:


df.PCR_test.dropna()


# In[70]:


sgt.plot_acf(df.PCR_test, lags = 40, zero = False)
plt.title("ACF for New Results Reported", size = 24)
plt.show()


# In[ ]:


# bottom = lags, left - values of ac coefficient
# corr - values between 1 and -1
# thin line - represents the ac from the ts and a lagged copy of itself
# 1st line - ac one time period ago - t-1 etc 
# blue area around the x-axis - significance - the values situated outside are significantly different from 0 - suggests the existence of ac
# blue area expands as lag values increase 
# the greater the distance in time, the more unlikely it is that this ac persists
# e.g. today's prices are more closer to yesterday's prices than prices one month ago - ac coefficient in higher lags is sufficiently greater to be significantly different from 0 
# all the lines are higher than the blue region - coefficients are significant - an indicator of time dependence in the data 
# ac barely diminshes as the lags decrease
# suggests that results from a month back can serve as decent estimators
# tell white noise a part from ts - fundamental in modeling
# determine and plot acf of white noise generated - change sequence of values passing to the arg


# In[624]:


sgt.plot_acf(df.wn, lags = 40, zero = False)
plt.title("ACF WN", size = 24)
plt.show()


# In[625]:


sgt.plot_pacf(df.PCR_test, lags =  40, zero = False, method = ("ols") )
# Order of Least Squares - OLS 
plt.title("PACF New Results Reported", size = 24)
plt.show()


# In[71]:


sgt.plot_pacf(df.wn, lags = 40, zero = False, method = ("ols"))
plt.title("PACF WN", size = 24)
plt.show()


# In[ ]:


#Creating Returns


# In[83]:


df['returns'] = df.PCR_test.pct_change(1)*100
df= df.iloc[1:]


# In[ ]:


#The ARIMA Model


# In[84]:


# Order - p, d, and q 
# P and q reprsent the AR and MA lags respectively
# d order is the integration--> the number of times we need to integrate the timepseries to ensure stationarity 
# No integration:
# ARIMA(0, 0 , q) =  MA)(q)
#ARIMA(p, 0, 0) = AR(p)
# ARIMA (p, 0, q) = ARMA (p,q)
# Integration - accounting for the non- seasonal difference between periods
# AR components= differences between PCR test results
# ARIMA (1, 1, 1)
# lose observations - for any integration we lose a single observation
# no previous period 



# In[ ]:


#LLR Test


# In[111]:


def LLR_Test(mod_1, mod_2, DF = 1):
    L1 = mod_1.fit(start_ar_lags = 20).llf
    L2 = mod_2.fit(start_ar_lags = 20).llf
    LR = (2*(L2-L1))
    p =  chi2.sf(LR, DF).round(3)
    return p 


# In[ ]:


#Creating Returns


# In[91]:


df['returns']= df.PCR_test.pct_change(1)*100


# In[ ]:


#ARIMA(1,1,1)


# In[183]:


import statsmodels.tsa.stattools as sts
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA


# In[184]:


model_ar_1_i_1_ma_1 = sm.tsa.arima.ARIMA(df.PCR_test, order = (1,1,1))
results_ar_1_i_1_ma_1 = model_ar_1_i_1_ma_1.fit()
print(results_ar_1_i_1_ma_1.summary())


# In[ ]:


# 2 coefficients
# integration order has no effect on the number of parameters we need to estimate
# Integration - we a re transforming the underlyuing data while no modellin is performed


# In[ ]:


#Residuals of ARIMA(1,1,1)


# In[185]:


df['res_ar_1_i_1_ma_1'] = results_ar_1_i_1_ma_1.resid
sgt.plot_acf(df.res_ar_1_i_1_ma_1, zero= False, lags = 40)
plt.title("ACF of Residuals for ARIMA(1,1,1)", size = 20)
plt.show()


# In[ ]:


# 3, 21 and 38 lags are highly significant
# incorporating lags into model might significantly improve performance
#Residuals follow the same pattern as ACF – no significant time period to use 
#try and see how the models that contain them perform


# In[ ]:


#Higher- Lag ARIMA Model


# In[186]:


model_ar_1_i_1_ma_2 = sm.tsa.arima.ARIMA(df.PCR_test, order = (1,1,2))
results_ar_1_i_1_ma_2 = model_ar_1_i_1_ma_2.fit()
model_ar_1_i_1_ma_3 = sm.tsa.arima.ARIMA(df.PCR_test, order = (1,1,3))
results_ar_1_i_1_ma_3 = model_ar_1_i_1_ma_3.fit()
model_ar_1_i_1_ma_1 = sm.tsa.arima.ARIMA(df.PCR_test, order = (2,1,1))
results_ar_1_i_1_ma_1 = model_ar_1_i_1_ma_1.fit()
model_ar_1_i_1_ma_1 = sm.tsa.arima.ARIMA(df.PCR_test, order = (3,1,1))
results_ar_1_i_1_ma_1 = model_ar_1_i_1_ma_1.fit()
model_ar_1_i_1_ma_2 = sm.tsa.arima.ARIMA(df.PCR_test, order = (3,1,2))
results_ar_1_i_1_ma_2 = model_ar_1_i_1_ma_2.fit()


# In[103]:


start_ar_lags=  5


# In[251]:


# provide enough starting AR lags for each model to allow for the execution of the fit model 
#.fit(start_ar_lags=  number)
#print the ".llf" and ".aic" store the log-likelihood and the AIC values for each model


# In[98]:


print("ARIMA(1,1,1):  \t LL = " , results_ar_1_i_1_ma_1.llf, "\t AIC = ", results_ar_1_i_1_ma_1.aic)
print("ARIMA(1,1,2):  \t LL = " , results_ar_1_i_1_ma_2.llf, "\t AIC = ", results_ar_1_i_1_ma_2.aic)
print("ARIMA(1,1,3):  \t LL = " ,results_ar_1_i_1_ma_3.llf,  "\t AIC = ",  results_ar_1_i_1_ma_3.aic)
print("ARIMA(2,1,1):  \t LL = " , results_ar_1_i_1_ma_1.llf, "\t AIC = ", results_ar_1_i_1_ma_1.aic)
print("ARIMA(3,1,1):  \t LL = " , results_ar_1_i_1_ma_1.llf, "\t AIC = ", results_ar_1_i_1_ma_1.aic)
print("ARIMA(1,1,2):  \t LL = ", results_ar_1_i_1_ma_1.llf,  "\t AIC = ",  results_ar_1_i_1_ma_1.aic)
    


# In[252]:


# highest log likelihood and the lowest AIC - model 2


# In[ ]:


#Examining the ACF of residuals 


# In[113]:


df['res_ar_1_i_1_ma_2'] = results_ar_1_i_1_ma_2.resid
sgt.plot_acf(df.res_ar_1_i_1_ma_2, zero= False, lags = 40)
plt.title("ACF of Residuals for ARIMA(1,1,2)", size = 20)
plt.show()


# In[114]:


model_ar_5_i_1_ma_1 = sm.tsa.arima.ARIMA(df.PCR_test, order = (5,1,1))
results_ar_5_i_1_ma_1 = model_ar_5_i_1_ma_1.fit()
model_ar_6_i_1_ma_3 = sm.tsa.arima.ARIMA(df.PCR_test, order = (6,1,3))
results_ar_6_i_1_ma_3 = model_ar_6_i_1_ma_3.fit()


# In[117]:


print("ARIMA(1,1,3):  \t  LL = ", results_ar_1_i_1_ma_3.llf, "\t AIC = ", results_ar_1_i_1_ma_3.aic)
print("ARIMA(5,1,1):  \t  LL = " , results_ar_5_i_1_ma_1.llf, "\t AIC = ", results_ar_5_i_1_ma_1.aic)
print("ARIMA(6,1,3):  \t  LL = " , results_ar_6_i_1_ma_3.llf, "\t AIC = ", results_ar_6_i_1_ma_3.aic)


# In[118]:


# AIC (6, 1, 3) is preferred
# The ARIMA(1,1,3) and ARIMA (5,1,1) are nested in the ARIMA (6,1,3)
#ARIMA(1,1,3) - 4 degrees of freedom 
#ARIMA(6,1,3) - 9 degrees of freedom


# In[121]:


df['res_ar_6_i_1_ma_3'] = results_ar_6_i_1_ma_3.resid
sgt.plot_acf(df.res_ar_6_i_1_ma_3, zero= False, lags = 40)
plt.title("ACF of Residuals for ARIMA(6,1,3)", size = 20)
plt.show()


# In[ ]:


#Models with Higher Levels of Integration 


# In[123]:


#captured effects incorporated into the 6th lag without including in model 
# the further back in time we go, the less relevant the values become
# include up to 40 lags into the model - we will have WN residuals
# want the model to predict other time-series data as well
# the model parameters will become too dependent on the data set - lead to overfitting (removing predictive power)
# best estimator for PCR tests - ARIMA(5,1, 1)


# In[124]:


df['delta_new_results_reported'] = df.PCR_test.diff(1)


# In[125]:


# ARIMA(P,1,Q) for new results reported
#fit ARMA(1,1) to the delta new results reported
#ARIMA(1,0,1) is equivalent to an ARMA(1,1)


# In[126]:


model_delta_ar_1_i_1_ma_1 = sm.tsa.arima.ARIMA(df.delta_new_results_reported[1:], order = (1,0,1))
results_delta_ar_1_i_1_ma_1 = model_delta_ar_1_i_1_ma_1.fit()
print(results_delta_ar_1_i_1_ma_1.summary())


# In[157]:


sts.adfuller(df.delta_new_results_reported[1:])


# In[169]:


# Test statistic is 14x greater in absolute value and the critical 1% value
# p-value  is 0.0 - confirmation of stationarity
# no need for additional layers of integration
# fitting ARIMA models with d> 1 is not recommended since the series is already stationary


# In[168]:


#ARIMA models estimate stationary data
# more computationally expensive than regular ARMA models - more layers more background work the program has to create before fitting the data
#transform the data several times
# differentiate the values from zero - harder as values become smaller
# margin of something being significant and insignificant becomes really narrow 
# models failing to converge after 1000s of iterations
# when numbers are too small - numerical instability = 0 - information loss
# the more layers we add, the harder it is to interpret the results
# by integrating stationary data - making it more difficult for the model to estimate the coefficients
# generate every layer of integration one by one and fit the data using ARMA models 
# Data Attrition - lose observations because of the conversion of new results reported and returns and lose more data points for each layer of integration - delta
# the more unnecessary layers we add, the more our model suffers


# In[170]:


# ARIMA(p,d,q)  -> RT
#make sure to integrate the appropriate number of rows from the start of the dataset 
# 1 row because we are using returns
#1 for each degree of freedom


# In[171]:


# Forecasting 
# Time series we expect patterns to persist as we progress through time 
# 1. find the pattern - selecting the correct model 
#2. predict the future


# In[188]:


size = int(len(df_comp)*0.8)
df, df_test = df_comp.iloc[:size], df_comp.iloc[size:]


# In[189]:


df_comp["ret_new_results_reported"] = df_comp.new_results_reported.pct_change(1).mul(100)


# In[190]:


df_comp.ret_new_results_reported = df_comp.ret_new_results_reported


# In[191]:


df_comp["norm_ret_new_results_reported"] = df_comp.ret_new_results_reported.div(df_comp.ret_new_results_reported[1])*100


# In[ ]:


#Fitting a Model


# In[192]:


model_ar = sm.tsa.ARIMA(df.new_results_reported, order =(1, 0, 0))
results_ar = model_ar.fit()


# In[ ]:


#Simple Forecasting


# In[886]:


# Specify a time period 
# the starting point of the forecasted period is the first one we do not have values for 
# the first day after the end of the training set


# In[193]:


df.tail()


# In[194]:


# Create variables that will help us change the periods easily instead of typing them up every time 
# make sure that the start and end dates are business days, otherwise the code will resilt in an error
start_date = "2020-03-01"
end_date = "2021-01-01"


# In[195]:



df_pred = results_ar.predict(start=start_date, end= end_date)


# In[196]:


df_pred[start_date:end_date].plot(figsize = (20, 5), color = "red")
plt.title("Predictions", size = 24)
plt.show()


# In[198]:


# Over the course of the interval actual PCR test results moved cyclically and fluctated up and down and up compared to the stationary predicted values


# In[199]:


df_pred[start_date:end_date].plot(figsize = (20, 5), color = "red")
df_test.new_results_reported[start_date:end_date].plot(color ="blue")
plt.title("Predictions vs. Actual", size = 24)
plt.show()


# In[ ]:


# Constant line at the 0 - model makes no predictions since it assumes all future returns will be 0, or extremely close to it
# coefficients for the past values and values themselves must have low absolute values


# In[ ]:


#Forecasting MA


# In[202]:


end_date = "2021-01-01"

model_ret_ma = sm.tsa.ARIMA(df.new_results_reported[1:], order =(0, 0, 1))
results_ret_ma = model_ret_ma.fit()

df_pred_ar = results_ret_ma.predict(start = start_date, end= end_date)
df_pred_ar[start_date:end_date].plot(figsize = (20,5), color= 'red')
df_test.new_results_reported[start_date:end_date].plot(color= "blue")
plt.title("Predictions cs Actual (Returns)", size = 24)
plt.show()


# In[263]:


# can't make ;ong run predictions if we are relying on error terms 
# cannot autogenerate residuals since we do not have actual values /
#manually create white noise redicuals for the entire period we are forecasting 
#we can recursively create a time series predictions


# In[203]:


df_pred_ma.head(12)


# In[204]:


#error terms, small coefficients


# In[205]:


# Pitfalls of forecasting
# Model -Specific
# Data -dependent 
# models we examined in forecasting were non-integrated 
# picking an incorrect type of model (integrated vs non-integrated) depending in data 
# always forecast stationary returns and create new results reported based on returns
# Integrated Models  ARIMA 
# Lack of visualization 
# we can't plot the integrated predictions against actual new results reported
# delta from testing set is created to compare with integrated predictions with actual values


# In[258]:


from jinja2.utils import markupsafe 
markupsafe.Markup()

