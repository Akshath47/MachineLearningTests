import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

def mean_square_error(y_true, y_cap):
    # Calculating the loss or cost
    cost = np.sum((y_true-y_cap)**2) / len(y_true)
    return cost

def loadData():
    house_data = pd.read_csv('house-prices.csv')
    house_data = house_data.drop(['Home', 'Offers', 'Neighborhood'], axis=1)

    x_train, x_test = train_test_split(house_data, test_size=0.2, random_state=25)

    x_train['Brick'] = x_train['Brick'].replace({"Yes":1, "No":0})
    x_test['Brick'] = x_test['Brick'].replace({"Yes":1, "No":0})

    y_train, y_test = x_train['Price'].to_numpy(), x_test['Price'].to_numpy()
    x_train, x_test = x_train.drop(['Price'], axis=1), x_test.drop(['Price'], axis=1)
    
    x_train, x_test = x_train.to_numpy(), x_test.to_numpy()

    return x_train, y_train, x_test, y_test

def estimateCost(x, estimated_weights, estimated_bias):
     cost = np.dot(estimated_weights, x) + estimated_bias
     return cost

def linearRegression(x, y):
    reg = LinearRegression().fit(x, y)
    return reg

def trainModel():
    x_train, y_train, x_test, y_test = loadData()
    model = linearRegression(x_train,y_train)

    weights = model.coef_
    bias = model.intercept_

    print(f"\nEstimated Weights: {(weights).tolist()}\nEstimated Bias: {bias}")

    #list of predicted values of y
    y_cap = np.dot(x_test, weights) + bias

    return weights, bias, y_cap, y_test

def main():
    estimated_weights, estimated_bias, y_cap, y_test = trainModel()
    print(f"Error: {mean_square_error(y_cap, y_test)}")

    #Making an estimation for a house
    print("Estimate sell value of your house: \n")
    SqFt = int(input("Enter the area of the house in square feet: "))
    bedrooms = int(input("Enter the number of bedrooms: "))
    bathrooms = int(input("Enter the number of bathrooms: "))
    brick = int(input("Enter 1 if the house uses bricks or 0 if it doesn't: "))
    print(estimateCost([SqFt, bedrooms, bathrooms, brick], estimated_weights, estimated_bias))


    
if __name__=="__main__":
	main()