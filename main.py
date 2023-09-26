import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# importing datasets

fraudTrain = pd.read_csv('C:/Users/navya/Downloads/fraudTrain.csv')
fraudTest = pd.read_csv('C:/Users/navya/Downloads/fraudTest.csv')
fraudfinal = pd.concat([fraudTrain, fraudTest])

# drop duplicate rows

fraudfinal.drop_duplicates(inplace = True)

# drop null values

fraudfinal.dropna(inplace = True)

# split "trans_date_trans_time" column into 2 and delete "trans_date_trans_time" column and changing datatype to datetime

fraudfinal[['trans_date', 'trans_time']] = fraudfinal.trans_date_trans_time.str.split(expand=True)
fraudfinal.drop(axis="columns", columns="trans_date_trans_time", inplace=True)
fraudfinal['trans_date'] = pd.to_datetime(fraudfinal['trans_date'])

# print info and 5 rows of final data set

print(fraudfinal.info())
print(fraudfinal.head())

job_amt = fraudfinal.loc[fraudfinal["state"] == "MO"].groupby(["job"])["amt"].mean()
print(job_amt)
job_amt.plot(kind = "bar", x = "job", y = "amt")
plt.show()

category_amt = fraudfinal.loc[fraudfinal["state"] == "MO"].groupby(["category"])["amt"].mean()
print(category_amt)
category_amt.plot(kind = "bar", x = "category", y = "amt")
plt.show()

ccnum_fraud = fraudfinal.loc[fraudfinal["state"] == "MO"].groupby(["cc_num"])["is_fraud"].sum()
print(ccnum_fraud)
ccnum_fraud.plot(kind = "bar", x = "cc_num", y = "is_fraud")
plt.show()

state_fraud = fraudfinal.groupby(["state"])["is_fraud"].sum()
print(state_fraud)
state_fraud.plot(kind = "bar", x = "state", y = "is_fraud")
plt.show()

city_fraud = fraudfinal.loc[fraudfinal["state"] == "NY"].groupby(["city"])["is_fraud"].sum()
print(city_fraud)
city_fraud.plot(kind = "bar", x = "city", y = "is_fraud")
plt.show()