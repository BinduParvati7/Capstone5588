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

# average amount spent by Job in missouri

job_amt = fraudfinal.loc[fraudfinal["state"] == "MO"].groupby(["job"])["amt"].mean()
print(job_amt)
job_amt.plot(kind = "bar", x = "job", y = "amt")
plt.title("average amount spent by Job in missouri")
plt.show()

# average amount spent by Category in missouri

category_amt = fraudfinal.loc[fraudfinal["state"] == "MO"].groupby(["category"])["amt"].mean()
print(category_amt)
category_amt.plot(kind = "bar", x = "category", y = "amt", color="red")
plt.title("average amount spent by Category in missouri")
plt.show()

# number of frauds recorded by Card Number in missouri

ccnum_fraud = fraudfinal.loc[fraudfinal["state"] == "MO"].groupby(["cc_num"])["is_fraud"].sum()
print(ccnum_fraud)
ccnum_fraud.plot(kind = "bar", x = "cc_num", y = "is_fraud", color="pink")
plt.title("number of frauds recorded by Card Number in missouri")
plt.show()

# number of frauds recorded by State

state_fraud = fraudfinal.groupby(["state"])["is_fraud"].sum()
print(state_fraud)
state_fraud.plot(kind = "bar", x = "state", y = "is_fraud", color="yellow")
plt.title("number of frauds recorded by State")
plt.show()

# number of frauds recorded by cities in new york

city_fraud = fraudfinal.loc[fraudfinal["state"] == "NY"].groupby(["city"])["is_fraud"].sum()
print(city_fraud)
city_fraud.plot(kind = "bar", x = "city", y = "is_fraud", color="violet")
plt.title("number of frauds recorded by cities in new york")
plt.show()

# number of frauds recorded by gender

gender_fraud = fraudfinal.groupby(["gender"])["is_fraud"].sum()
print(gender_fraud)
gender_fraud.plot(kind = "pie", x = "gender", y = "is_fraud")
plt.title("number of frauds recorded by gender")
plt.show()

