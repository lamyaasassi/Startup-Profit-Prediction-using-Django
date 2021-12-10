from django.shortcuts import render
from django.http import HttpResponse

import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.function_base import select
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error as mse, mean_absolute_error as mae
import math


# Create your views here.
def hi(request):
    return render(request, 'APPLICATION/index.html')
def predict(request):
    dataset = pd.read_csv('../DATASET/50_Startups.csv')
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values
    ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
    X = ct.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
    lin_reg = LinearRegression(normalize=True)
    lin_reg.fit(X_train,y_train)
    if "State" == "NewYork":
        var1 = 1
        var2 = 0
        var3 = 0
    elif "State" == "California":
        var1 = 0
        var2 = 1
        var3 = 0
    else:
        var1 = 0
        var2 = 0
        var3 = 1
    var4 = float(request.POST['RnDSpend'])
    var5 = float(request.POST['Administration'])
    var6 = float(request.POST['MarketingSpend'])

    prediction = lin_reg.predict(np.array([var1, var2, var3, var4, var5, var6]).reshape(1,-1))
    prediction = round(prediction[0])

    profit = "The predicted profit is " + str(prediction)
    RnDSpend = "\"R&D Spend\" value given : " + str(var4)
    Administration = "\"Administration\" value given : " + str(var5)
    MarketingSpend = "\"Marketing Spend\" value given : " + str(var6)
    if(var1==1 and var2==0 and var3==0):
        State = "\"State\" given : New York"
    elif(var1==0 and var2==1 and var3==0):
        State = "\"State\" given : California"
    elif(var1==0 and var2==0 and var3==1):
        State = "\"State\" given : Florida"

    return render(request, 'APPLICATION/index.html', {"result":profit, "RnDSpend":RnDSpend, "Administration":Administration, "MarketingSpend":MarketingSpend, "State":State})

