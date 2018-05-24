import sys
sys.path.insert(0, 'E:/indepedent project/project ver1/')

from machine_learning import learn
from machine_learning import model
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

path = "E:/datasets/iris/Iris.csv"
df = pd.read_csv(path)

def encode_labels(column, dataframe):
    unique = dataframe[column].unique()
    id_dict = {}
    counter = 0
    for key in unique:
        id_dict[key] = counter
        counter += 1
    return id_dict

id_dict = encode_labels('Species', df)
df['Species'] = df['Species'].map(id_dict)

features = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
target = ['Species']

X = df[features]
y = df[target]

classifier = learn.Classifier(verbose=1)
accuracy = classifier.random_forest_classifier(X, y, features, target)

graph = model.Graph()
graph.histogram(df['SpeciesLengthCm'], 'Length', 'Probability Density')
