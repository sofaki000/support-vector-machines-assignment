from matplotlib import pyplot as plt

from classifiers import MaxMarginClassifier, LinearSvmClassifier
from data_utilities.test_data import generateBatchBipolar
from plot_utilities import plotSvm

N = 100
xTrain0, yTrain0 = generateBatchBipolar(N,  mu=0.5, sigma=0.2)

model00 = MaxMarginClassifier()
model00.fit(xTrain0, yTrain0)
print(model00.w, model00.intercept)

fig, ax = plt.subplots(1, figsize=(12, 7))
plotSvm(xTrain0, yTrain0, model00.supportVectors, model00.w, model00.intercept, label='Training', ax=ax)

xTrain1, yTrain1 = generateBatchBipolar(N, mu=0.3, sigma=0.3)
plotSvm(xTrain1, yTrain1, label='Training')
model10 = LinearSvmClassifier(C=1)
model10.fit(xTrain1, yTrain1)
print(model10.w, model10.intercept)
fig, ax = plt.subplots(1, figsize=(11, 7))
plotSvm(xTrain1, yTrain1, model10.supportVectors, model10.w, model10.intercept, label='Training', ax=ax)