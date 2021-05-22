"""Importing modules"""
import numpy as np
import pandas as pd

"""Importing dataset"""
data = pd.read_csv("car.csv")

"""Processing to dataset clean""" 
data = data[data["year"].str.isnumeric()] # filter the year column in Alphabetic String and numerical string
data["year"] = data["year"].astype(int) # chage the data type of filtering year value

data = data[data["Price"] != "Ask For Price"] # remove alphabetic-string into the numeric-string column
data["Price"] = data["Price"].str.replace(",", "").astype(int) # change the data type as integer in Price column

data["kms_driven"] = data["kms_driven"].str.split().str.get(0).str.replace(",", "") # First split the Kms_driven column
                                                                # then get 0 index position value, and it has separated 
                                                                # with "," so replace it with "" space
data = data[data["kms_driven"].str.isnumeric()] # filter the kms_driven column in Alphabetic String and numerical string
data["kms_driven"] = data["kms_driven"].astype(int) # chage the data type of filtering value

data = data[~data["fuel_type"].isna()] # remove the NaN row based on fuel_type column

data["name"] = data["name"].str.split().str.slice(0,3).str.join(" ") # split the name column and get 3 words the then make it join again

data = data.reset_index(drop=True) # reset the index 

data = data[data["Price"]<6000000] # remove outlires row

"""Export new dataset to CSV before preprocessing"""
data.to_csv("clean_data_car.csv")

x = data[["name", "company", "year", "kms_driven", "fuel_type"]] # split the dataset into input 
y = data["Price"] # split the dataset into output

"""Prediction Part"""

### ENCODING THE CATAGORICAL DATA COLUMN ###
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
oht = OneHotEncoder()
oht.fit(x[["name", "company", "fuel_type"]])

ct = ColumnTransformer(transformers = [("Syed", OneHotEncoder(categories= oht.categories_, handle_unknown="ignore"), ["name", "company", "fuel_type"])], remainder="passthrough") # make the catagorical data 
                                                                                                    # as a encode through the OneHotEncoder function, and again
                                                                                                    # join this catagorical data with ColumnTransformer fanction
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size= 0.8, random_state= 0) # split all the dataset into training and testing dataset

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

from sklearn.pipeline import make_pipeline
pipe = make_pipeline(ct, regressor) # give the col-trans and regressor into the pipeline so that then can work together 

pipe.fit(x_train, y_train) # train the hole data set
y_pred = pipe.predict(x_test) # predict the dataset
pred = pipe.predict(pd.DataFrame(columns=["name", "company", "year", "kms_driven", "fuel_type"], data= np.array(["Skoda Fabia Classic", "Skoda", 2010, 6000, "petrol"]).reshape(1, 5)))

from sklearn.metrics import r2_score
score = r2_score(y_test, y_pred) * 100

import pickle
pickle.dump(pipe, open("car_predictor.pkl", "wb"))

