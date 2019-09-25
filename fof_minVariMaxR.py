# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 09:42:32 2017

@author: zhujisong-001
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
#import scipy.optimize as sco
#from scipy.misc import derivative
#
#def partial_derivative(func, var=0, point=[]):
#    args = point[:]
#    def wraps(x):
#        args[var] = x
#        return func(*args)
#    return derivative(wraps, point[var], dx=1e-6)
#
#def riskConcentration(weight, retCov):
#    
#    def portfolioRisk(*weight):
#        w = np.mat(weight)
#        sigma = np.sqrt(w*np.mat(retCov)*w.T)
#        return(sigma)
#    trc = np.zeros(len(weight))
#    for k in range(len(weight)):
#        trc[k] = weight[k]*partial_derivative(portfolioRisk, k, weight)
#    rc = 0
#    for p in range(len(weight)):
#        for q in range(len(weight)):
#            rc += (trc[p] - trc[q])**2
#    return(rc)

indexRets = pd.read_csv('/Users/admin/Desktop/doc/finance/multifactor/data/fof/indexRets.csv')
tradeDate = indexRets['tradeDate']
indexRets['tradeDate'] = pd.to_datetime(indexRets['tradeDate'],format='%Y-%m-%d')
indexRets.set_index('tradeDate', inplace = True)
tradeDate.index = indexRets.index

indexMonthlyRets = indexRets.groupby(pd.TimeGrouper(freq = 'M')).sum()
indexMonthlyRets = np.exp(indexMonthlyRets) - 1 #translate the log return to normal return

nAsset = len(indexRets.columns)
weight = np.array([1/nAsset]*nAsset)
print(weight)

sess = tf.InteractiveSession()
tfWeight =  tf.Variable(weight, dtype = tf.float32, name = 'weight')
import pdb;pdb.set_trace()
tfWeight = tf.expand_dims(tfWeight,0)
tfCov = tf.placeholder(dtype = tf.float32, shape = (nAsset, nAsset), name = 'retcov')
tfRateO = tf.placeholder(dtype = tf.float32, shape = (nAsset), name = 'retmean')
tfRate = tf.expand_dims(tfRateO,0)
#portfolioCov = tf.matmul(tf.matmul(tf.expand_dims(tfWeight,0), retCov), tf.expand_dims(tfWeight,-1))
portfolioCov = tf.matmul(tf.matmul(tfWeight, tfCov), tf.transpose(tfWeight), name = 'portcov')
#portfolioSigma = tf.sqrt(portfolioCov, name = 'portsigma')
portfolioRate = tf.multiply(tfWeight, tfRate,name = 'portr')

#(1,n)
#marginalRisk = tf.gradients(portfolioSigma, tfWeight)

#（1，n）
#riskContribution = tf.multiply(tfWeight, marginalRisk[0], name = 'riskcontri')
#riskConcentration = tf.zeros(1, dtype = tf.float32)

#for p in range(nAsset):
    #for q in range(nAsset):
        #riskConcentration += (riskContribution[0, p] - riskContribution[0, q])**2
                             
loss = portfolioCov - tf.minimum(tf.reduce_min(tfWeight),0) + (tf.reduce_sum(tfWeight) - 1)**2 + (tf.reduce_sum(portfolioRate) - float(0.06/360))**2
optimizer = tf.train.GradientDescentOptimizer(0.1)
optimize = optimizer.minimize(loss)


freq = 'M'
# firstTradeDateInMonth = tradeDate.groupby(pd.TimeGrouper(freq='M')).min()
firstTradeDateInMonth = tradeDate.groupby(pd.TimeGrouper(freq=freq)).min()

portfolioRets = []
mdates = []
optimWeigts = []
print("-----"+str(len(firstTradeDateInMonth)))
init = tf.global_variables_initializer()
needReCal = True
calConvCount = 3
count = 0
#for k in range(3, 7):
for k in range(3, int(len(firstTradeDateInMonth))):
    curDate = firstTradeDateInMonth[k]

    pastData = indexRets[indexRets.index < curDate]
    print(curDate)

    if needReCal==True:
        # if freq=="M":
            # recentData = pastData[-30:]
        # elif freq=="BQ":
            # recentData = pastData[-30*calConvCount:]
        recentData = pastData[-30*calConvCount:]
        mean = recentData.mean()
        retCov = recentData.cov()

        import pdb;pdb.set_trace()
        sess.run(init)
        sess.run(optimize, feed_dict={tfCov:retCov.values,tfRateO:mean.values})
        rc0 = sess.run(loss, feed_dict={tfCov:retCov.values,tfRateO:mean.values})[0]

        for step in range(200000):
            sess.run(optimize, feed_dict={tfCov:retCov.values,tfRateO:mean.values})
            rc1 = sess.run(loss, feed_dict={tfCov:retCov.values,tfRateO:mean.values})[0]
            if abs(rc0 - rc1) < 1e-10:
                break
            rc0 = rc1
            if step % 1000 == 0:
                print(step, sess.run(tfWeight))
        
        optimWeight = sess.run(tfWeight)[0]
        import pdb;pdb.set_trace()
        sum = np.dot(optimWeight,mean)
        print(sum)


    mdate = firstTradeDateInMonth.index[k]
    mret = indexMonthlyRets[indexMonthlyRets.index == mdate]
    portfolioRet = np.dot(mret, optimWeight)
    
    portfolioRets.append(portfolioRet)
    mdates.append(mdate.date())
    optimWeigts.append(optimWeight)
    
    print('***', step, optimWeight, mdate.strftime('%Y-%m-%d'), portfolioRet)

     #if freq=="M":
         #needReCal = True
     #elif freq=="BQ":
     
    count = count+1
    if(count>=calConvCount):
        needReCal = True
        count = 0
    else:
        needReCal = False
    
    
       
    
portfolioRetDF = pd.DataFrame(portfolioRets[:])
portfolioRetDF.set_index(pd.to_datetime(mdates), inplace = True)
portfolioRetDF.columns = ['portfolio']
res = portfolioRetDF.merge(indexMonthlyRets[['hs300']], left_index = True, right_index = True)
res.plot()
(res+1).cumsum().plot()
plt.show()
name2 = input("Please intput your name:")