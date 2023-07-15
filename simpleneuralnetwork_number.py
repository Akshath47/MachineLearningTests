import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

COLOR = 'black'
plt.rcParams['text.color'] = COLOR
plt.rcParams['axes.labelcolor'] = COLOR

def loadData():
    train_set = pd.read_csv('mnist_train.csv')
    test_set = pd.read_csv('mnist_test.csv')

    train_labels = train_set['label'].to_numpy()
    test_labels = test_set['label'].to_numpy()

    train_set = train_set.drop(['label'], axis=1).to_numpy()
    test_set = test_set.drop(['label'], axis=1).to_numpy()

    train_set = np.reshape(train_set, [60000,28,28])
    test_set = np.reshape(test_set, [10000,28,28])

    class_names = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    train_set = train_set / 255
    test_set = test_set / 255

    return train_set, train_labels, test_set, test_labels

def getModel():
    model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),  # input layer (1)
    keras.layers.Dense(32, activation='relu'),  # hidden layer (2)
    keras.layers.Dense(64, activation='relu'),  # hidden layer (3)
    keras.layers.Dense(10, activation='softmax') # output layer (4)
    ])
    return model

def trainModel(model, train_set, train_labels, test_set, test_labels):
    model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

    model.fit(train_set, train_labels, epochs=10)

    test_loss, test_acc = model.evaluate(test_set,  test_labels, verbose=1)

    print('Test accuracy:', test_acc)   
  

def predict(model, image, correct_label):
    class_names = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    prediction = model.predict(np.array([image]))
    predicted_class = class_names[np.argmax(prediction)]

    show_image(image, class_names[correct_label], predicted_class)

def show_image(img, label, guess):
    plt.figure()
    plt.imshow(img, cmap=plt.cm.binary)
    plt.title("Excpected: " + str(label))
    plt.xlabel("Guess: " + str(guess))
    plt.colorbar()
    plt.grid(False)
    plt.show()

def get_number():
    while True:
        num = input("Pick a test index number: ")
        if num.isdigit():
          num = int(num)
          if 0 <= num <= 10000:
              return int(num)
        else:
          print("Try again...")

train_set, train_labels, test_set, test_labels = loadData()
model = getModel()
trainModel(model, train_set, train_labels, test_set, test_labels)

num = get_number()
image = test_set[num]
label = test_labels[num]
predict(model, image, label)




