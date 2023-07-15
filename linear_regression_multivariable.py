import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


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


def mean_square_error(y_true, y_cap):
    #calculating cost
    return np.sum((y_cap-y_true)**2) / len(y_true)


def gradientFunction(x_train, y_train, iterations = 100, learning_rate = 1e-10, epsilon = 1e-10):
    '''
    x_train = input dataframe
    y = output array
    '''
    w = np.random.rand(4)
    b = 1
    iterations = iterations
    learning_rate = learning_rate
    epsilon = epsilon
    n = float(len(y_train))

    previous_cost = None

    for i in range(iterations):
        #calculating predictied y values
        y_cap = np.dot(x_train, w) + b

        current_cost = mean_square_error(y_train,y_cap)
         
        if previous_cost and abs(current_cost-previous_cost)<=epsilon:
            break

        #calculating derivatives
        w_list_temp = (np.c_[y_cap]-np.c_[y_train]) * x_train       
        w_derivative = sum(w_list_temp.sum(axis=1))
        b_derivative = (1/n) * sum(y_cap-y_train)

        #gradient descent function till convergence
        w = w - learning_rate*(w_derivative)
        b = b - learning_rate*(b_derivative)

        #printing each iteration
        print(f"Iteration {i+1}: Cost = {current_cost}, Weight = {w}, Bias = {b}")  

    return w, b


def estimateCost(x, estimated_weights, estimated_bias):
     cost = np.dot(estimated_weights, x) + estimated_bias
     return cost


def trainModel():
    x_train, y_train, x_test, y_test = loadData()
    estimated_weights, estimated_bias = gradientFunction(x_train,y_train)

    print(f"\nEstimated Weight: {(estimated_weights).tolist()}\nEstimated Bias: {estimated_bias}")

    #list of predicted values of y
    y_cap = np.dot(x_test, estimated_weights) + estimated_bias

    return estimated_weights, estimated_bias, y_cap, y_test


def main():
    estimated_weights, estimated_bias, y_cap, y_test = trainModel()
    print(f"Error: {mean_square_error(y_cap, y_test)}")

    # #Making an estimation for a house
    # print("Estimate sell value of your house: \n")
    # SqFt = int(input("Enter the area of the house in square feet: "))
    # bedrooms = int(input("Enter the number of bedrooms: "))
    # bathrooms = int(input("Enter the number of bathrooms: "))
    # brick = int(input("Enter 1 if the house uses bricks or 0 if it doesn't: "))
    # print(estimateCost([SqFt, bedrooms, bathrooms, brick], estimated_weights, estimated_bias))

    
if __name__=="__main__":
	main()