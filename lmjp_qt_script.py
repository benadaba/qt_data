# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 21:13:54 2016

@author: Bernard
"""
'''
1 - Build a quick script using python + numpy that: 
- creates a random vector with uniform statistical distribution between 1 and 3 
- selects all values from the vector that are above 2.5 and change these values to 2.5
 (so that the max of the array is 2.5) 
- plot the distribution with matplotlib

'''
import numpy as np

#creates a random vector with uniform statistical distribution between 1 and 3 
rv = np.random.uniform(1, 3, 50)
print(rv)

#check if generated vectors are each greater than 1

print(np.all(rv > 1))

#check if generated vectors are each less than 3
print(np.all(rv < 3))

#selects all values from the vector that are above 2.5 and change these values to 2.5
rv[rv > 2.5] = 2.5
        
print(rv)

#check the max value in the array
np.amax(rv)

#- plot the distribution with matplotlib
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
plt.hist(rv)
plt.title("Histogram for Generated Vector")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()

#################################################
'''
2 - Also, with numpy and python 
- create a gaussian normal distribution centered at 2 with standard dev of 1 
- create a uniform distribution between 3 and 4 
- calculate Percentile 75 for both distributions. 
Please explain the meaning of such result
'''
################################################
#- create a gaussian normal distribution centered at 2 with standard dev of 1 
mu = 2
stdv = 1
gd = np.random.normal(mu, stdv, 50)
print(gd)

'''
plt.plot(gd,mlab.normpdf(gd, mu, stdv))

plt.show()

plt.hist(gd, normed=True)
plt.title("Normal distribution centered at 2 with standard dev of 1")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()

'''
count, bins, ignored = plt.hist(gd, 10, normed=True)
plt.title("Normal distribution centered at 2 with standard dev of 1")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.plot(bins, 1/(stdv * np.sqrt(2 * np.pi)) *
               np.exp( - (bins - mu)**2 / (2 * stdv**2) ),
        linewidth=2, color='r')
plt.show()

#- create a uniform distribution between 3 and 4 
un34 = np.random.uniform(3, 4, 50)
print(un34)
#check if generated vectors are each greater than 3

print(np.all(un34 > 3))

#check if generated vectors are each less than 4
print(np.all(un34 < 4))

plt.hist(un34)
plt.title("Histogram for Generated Vector")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()

#- calculate Percentile 75 for both distributions. 
# 75th Percentile for gaussian normal distribution
g75 = np.percentile(gd, 75)
print("75th Percentile for gaussian normal distribution is:", g75 )
#meaning 
#This means that 75 percent of the distribution fall below the value of g75
print("This means that 75 percent of the gaussian normal distribution falls below ", g75)

# 75th Percentile for uniform distribution between 3 and 4 
un75 = np.percentile(un34, 75)
print("75th Percentile for uniform distribution between 3 and 4 is:", un75 )
#meaning 
#This means that 75 percent of the distribution fall below the value of g75
print("This means that 75 percent of the uniform distribution between 3 and 4 falls below ", un75)


################################################
'''
3- Calculate MAPE error and MPE error between the two arrays in the attached excel sheet
 - Are these errors representative of the difference/error between the two arrays?
Please suggest other ways to calculate such errors.
Also, create a visual representation (plot or web plot) that shows actuals, 
forecast, errors for each sample and most important statistics (mean, stdev,...)
'''
###################

from numpy import genfromtxt

#read csv file
my_data = genfromtxt('limejump.csv', delimiter=',')

#get actuals
act = my_data[4,1:]

#remove all nans
actuals = act[np.logical_not(np.isnan(act))]
#actuals =  actuals.astype(int)
#get forecast
fore = my_data[5,1:]

#remove all nans
forecast = fore[np.logical_not(np.isnan(fore))]
#forecast = forecast.astype(int)
#Mean absolute percentage error
MAPE = np.mean((np.abs( forecast != actuals ))) * 100

#Mean percentage error
MPE = np.mean(( forecast != actuals )) * 100

#errors
errors = np.abs(( actuals - forecast  ))

# mean , standard deviation, mode and median of actuals and forecasts
#mean
a_mean = actuals.mean()
f_mean = forecast.mean()

#standard deviation
a_std = actuals.std()
f_std = forecast.std()

#median
a_median = np.median(actuals)
f_std = np.median(forecast)

from scipy import stats
#mode
a_mode = stats.mode(actuals)
f_mode = stats.mode(forecast)

#median
#Are these errors representative of the difference/error between the two arrays?
# YES these errors are representative of the difference/error between the two arrays
print("YES these errors are representative of the difference/error between the two arrays")

#Please suggest other ways to calculate such errors.
#You can calculate writing your own function
# you can calculate using Excel

plt.figure(figsize=(20,12))
plt.plot(actuals, label='Actuals', color='red', linewidth=2)
plt.plot(forecast, 
     '--', color='blue', linewidth=3, label='Forecast')
#plt.plot(errors, label='Errors', color='green', linewidth=3)
plt.errorbar( actuals, forecast, errors, linestyle='None', marker='^', label='Errors')
plt.title("Visualizations of Actuals, Forecasts and Errors")
#plt.xticks(arange(len(dates))[1::4],dates[1::4], rotation=45)
plt.legend(loc='upper right')
plt.ylabel('Values to be predicted')
plt.show()
