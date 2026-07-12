import numpy as np
import pandas as pd
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
#load the iris dataset
iris=datasets.load_iris()

x=iris.data
print(x)
y=iris.target

#X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.2,random_state=42)