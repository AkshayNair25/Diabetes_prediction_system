from django.shortcuts import render
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score


def home(request):
    return render(request, "home.html")


def predict(request):
    return render(request, "predict.html")


def result(request):
    diabetes_dataset = pd.read_csv(r'D:\college\sem6\PROJECT\DIABETES PREDICTION SYSTEM\diabetes.csv')

    X = diabetes_dataset.drop(columns='Outcome', axis=1)
    Y = diabetes_dataset['Outcome']

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

    classifier = svm.SVC(kernel='linear')

    classifier.fit(X_train, Y_train)

    val1 = float(request.GET['n1'])
    val2 = float(request.GET['n2'])
    val3 = float(request.GET['n3'])
    val4 = float(request.GET['n4'])
    val5 = float(request.GET['n5'])
    val6 = float(request.GET['n6'])
    val7 = float(request.GET['n7'])
    val8 = float(request.GET['n8'])

    pred = classifier.predict([[val1, val2, val3, val4, val5, val6, val7, val8]])

    result1 = ""

    if pred == [1]:
        result1 = "Patient is diabetic."
        if (val8 < 20):
            result1 = "There might be chances of type 1 diabetes please proceed with A1C test."

    else:
        result1 = "Patient is not Diabetic."
    return render(request, "predict.html", {"result2": result1})


def analysis(request):
    return render(request, "analysis.html")


def Hospitals(request):
    return render(request, "Hospitals.html")

