import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, LSTM
model = Sequential()

data = pd.read_csv("deliverytime.txt")
print("Food Delivery Time Prediction")
a = input("Age of Delivery Person: ")
b = float(input("Restaurant Ratings of Previous Deliveries: "))
c = int(input("Total Distance: "))

features = np.array([[ a, b, c ]])
print("Predicted Delivery Time in Minutes = ", model.predict(features))