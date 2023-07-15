import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


def mean_square_error(y_true, y_cap):
    # Calculating the loss or cost
    cost = np.sum((y_true-y_cap)**2) / len(y_true)
    return cost


def linearRegression(x, y):
    reg = LinearRegression().fit(x, y)
    return reg


def trainModel():

    house_data = pd.read_csv('house-prices.csv')
    required_data = house_data.copy()
    for item in required_data:
        if (item != 'Price') and (item != 'SqFt') and (item != 'Bedrooms'):
            required_data.pop(item)

    x = np.array(required_data['SqFt']).reshape((-1,1))
    y = np.array(required_data['Price'])

    #calculating coefficients using linear regression
    model = linearRegression(x, y)
    weight = model.coef_[0]
    bias = model.intercept_

    #list of predicted values of y
    y_cap = (weight * x)+ bias

    #graphing scatter plot and line of best fit for estimations
    plt.figure(figsize = (8,6))
    plt.scatter(x,y, marker='o', color='red')
    plt.plot([min(x), max(x)], [min(y_cap), max(y_cap)], color='blue', markerfacecolor='red', markersize=10, linestyle='dashed')
    plt.title("Area vs Cost for Houses")
    plt.xlabel("Square Feet")
    plt.ylabel("Cost in USD")
    plt.show()

    return weight, bias


def estimateCost(SqFt, estimated_weight, estimated_bias):
     cost = (estimated_weight*SqFt) + estimated_bias
     return cost


def main():
    weight, bias = trainModel()

    #Making an estimation for a house
    print("Estimate sell value of your house: \n")
    sqFt = int(input("Enter the area of the house in square feet "))
    print(estimateCost(sqFt, weight, bias))


if __name__=="__main__":
	main()

