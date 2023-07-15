import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def mean_square_error(y_true, y_cap):
    # Calculating the loss or cost
    cost = np.sum((y_true-y_cap)**2) / len(y_true)
    return cost


#1 feature(variable) linear gradient descent function
def gradientDescent(x,y,iterations = 80, learning_rate = 1e-8, stopping_threshold = 1e-8):
    '''
    x=input array
    y=output array
    '''
    w = 1 #weight
    b = 1 #bias
    iterations = iterations
    learning_rate = learning_rate
    n = float(len(x))

    weights = []
    costs = []
    iterationNo = []
    previous_cost = None

    for i in range (iterations):
        y_cap = (w*x) + b
        current_cost = mean_square_error(y,y_cap)

        # If the change in cost is less than or equal to stopping_threshold we stop the gradient descent
        if previous_cost and abs(previous_cost-current_cost)<=stopping_threshold:
            break

        previous_cost = current_cost

        costs.append(current_cost)
        weights.append(w)
        iterationNo.append(i)
                
        #calculating derivatives
        w_derivative = (2/n) * sum(x * (y_cap-y))
        b_derivative = (2/n) * sum((y_cap-y))

        #updating weights and bias
        w = w - (learning_rate * w_derivative)
        b = b - (learning_rate * b_derivative)

        #printing each iteration
        print(f"Iteration {i+1}: Cost = {current_cost}, Weight = {w}, Bias = {b}")  

    #graphing cost vs weights
    plt.figure(figsize=(8,6))
    plt.plot(weights, costs)
    plt.scatter(weights, costs, marker='o', color='red')
    plt.title("Cost vs weights graph")
    plt.xlabel("Weight")
    plt.ylabel("Cost")
    plt.show()

    #graphing the learning curve
    #to see how many iterations required for gradient descent to converge
    plt.figure(figsize=(8,6))
    plt.plot(iterationNo, costs)
    plt.title("Learning curve")
    plt.xlabel("#Iterations")
    plt.ylabel("Cost")
    plt.show()

    return w,b 

def estimateCost(SqFt, estimated_weight, estimated_bias):
     cost = (estimated_weight*SqFt) + estimated_bias
     return cost


def trainModel():

    house_data = pd.read_csv('house-prices.csv')
    required_data = house_data.copy()
    for item in required_data:
        if (item != 'Price') and (item != 'SqFt') and (item != 'Bedrooms'):
            required_data.pop(item)

    x = np.array(required_data['SqFt'])
    y = np.array(required_data['Price'])

    #calculating estimated weight, bias using gradient descent function
    estimated_weight, estimated_bias = gradientDescent(x,y)
    print(f"\nEstimated Weight: {estimated_weight}\nEstimated Bias: {estimated_bias}")

    #list of predicted values of y
    y_cap = (estimated_weight * x)+ estimated_bias

    #graphing scatter plot and line of best fit for estimations
    plt.figure(figsize = (8,6))
    plt.scatter(x,y, marker='o', color='red')
    plt.plot([min(x), max(x)], [min(y_cap), max(y_cap)], color='blue', markerfacecolor='red', markersize=10, linestyle='dashed')
    plt.title("Area vs Cost for Houses")
    plt.xlabel("Square Feet")
    plt.ylabel("Cost in USD")
    plt.show()

    return estimated_weight, estimated_bias


def main():
    estimated_weight, estimated_bias = trainModel()

    #Making an estimation for a house
    print("Estimate sell value of your house: \n")
    sqFt = int(input("Enter the area of the house in square feet "))
    print(estimateCost(sqFt, estimated_weight, estimated_bias))


if __name__=="__main__":
	main()
        