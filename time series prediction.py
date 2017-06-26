import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15, 6
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf, pacf  
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf ,plot_pacf
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller as ADF
from statsmodels.stats.diagnostic import acorr_ljungbox

def test_stationarity(timeseries):
    
    #Determing rolling statistics
    rolmean = pd.rolling_mean(timeseries, window=12)
    rolstd = pd.rolling_std(timeseries, window=12)

    #Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    
    
    #Perform Dickey-Fuller test:
    print ('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)


if __name__ == '__main__':
    
    dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d %H:%M')
    data = pd.read_csv('xiamen_1.csv', parse_dates=True, index_col='time',date_parser=dateparse)
    ts = data['volme']
    #plt.plot(ts)
    print(ADF(ts))
    #plot_acf(ts,lags=20)
    #plot_pacf(ts,lags=20)

    #stationarity
    #plt.figure(4)
    #test_stationarity(ts)





    ts_log = np.log(ts)
    print(u'差分序列的白噪声检验结果为：', acorr_ljungbox(ts_log, lags=6))
    #Smoothing
    #plt.figure(3)
    #moving_avg = pd.rolling_mean(ts_log,12)
    #plt.plot(ts_log)
    #plt.plot(moving_avg, color='red')

    model = ARIMA(ts_log, order=(5, 0, 0))
    print(model.fit().summary2()) 

    print(model.fit().forecast(5))

    plt.figure(8)
    results_ARIMA = model.fit(disp=-1)  


    plt.figure(15)
    predic_dic=model.fit().predict('2016-4-28')
    print(predic_dic)
    #fig, ax = plt.subplots(figsize=(12, 8))
    #ax = ts_log.ix['2016-4-13':'2016-4-27'].plot(ax=ax)
   # fig = model.fit().plot_predict('2016-4-28', dynamic=True, ax=ax, plot_insample=False)
    plt.plot(ts_log)
    plt.plot(predic_dic,color='red')



    plt.figure(13)
    plt.plot(ts_log)
    plt.plot(results_ARIMA.fittedvalues, color='red')
    plt.title('RSS: %.4f'% sum((results_ARIMA.fittedvalues-ts_log)**2))

    plt.show()
