import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC  


def loadData():
    df = pd.read_csv('bank-full.csv', sep=";")

    df = df.drop(labels=['duration', 'campaign', 'pdays', 'previous', 'poutcome'], axis=1)

    #encoding categorical strings
    le = LabelEncoder()
    df['job'] = le.fit_transform(df['job'])
    df['marital'] = le.fit_transform(df['marital'])
    df['education'] = le.fit_transform(df['education'])
    df['default'] = le.fit_transform(df['default'])
    df['housing'] = le.fit_transform(df['housing'])
    df['loan'] = le.fit_transform(df['loan'])
    df['contact'] = le.fit_transform(df['contact'])
    df['day'] = le.fit_transform(df['day'])
    df['month'] = le.fit_transform(df['month'])

    scaler = StandardScaler()
    df[['balance']] = scaler.fit_transform(df[['balance']])

    test_data, train_data = train_test_split(df, test_size=0.2, random_state=20)

    y_test = test_data['y']
    test_data.drop(['y'], axis=1, inplace=True)
    y_train = train_data['y']
    train_data.drop(['y'], axis=1, inplace=True)

    return train_data, y_train, test_data, y_test


def logisticRegression(train_data, y_train, test_data, y_test):
    logreg = LogisticRegression(max_iter=4000)
    logreg.fit(train_data, y_train)

    y_pred_logreg = logreg.predict(test_data)

    #Accuracy metrics
    accuracy_logreg = logreg.score(test_data, y_test)

    scores_logreg = cross_val_score(logreg, train_data, y_train, cv=10)
    print('Cross-Validation Accuracy Scores', scores_logreg)
    scores_logreg = pd.Series(scores_logreg)

    return accuracy_logreg, scores_logreg, y_pred_logreg


def kNearestNeighbour(train_data, y_train, test_data, y_test, k):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(train_data, y_train)

    y_pred_knn = knn.predict(test_data)

    #Accuracy metrics
    accuracy_knn = knn.score(test_data, y_test)

    scores_knn = cross_val_score(knn, train_data, y_train, cv=10)
    print('Cross-Validation Accuracy Scores', scores_knn)
    scores_knn = pd.Series(scores_knn)

    return accuracy_knn, scores_knn, y_pred_knn


def supportVectorMachine(train_data, y_train, test_data, y_test):
    svm = SVC(kernel = 'linear')
    svm.fit(train_data, y_train)

    y_pred_svm = svm.predict(test_data)

    #Accuracy metrics
    accuracy_svm = svm.score(test_data, y_test)

    scores_svm = cross_val_score(svm, train_data, y_train, cv=10)
    print('Cross-Validation Accuracy Scores', scores_svm)
    scores_svm = pd.Series(scores_svm)

    return accuracy_svm, scores_svm, y_pred_svm


def findKValue(train_data, y_train, test_data, y_test):
    #Finding best value of K
    neighbors = np.arange(1, 30)
    train_accuracy = np.empty(len(neighbors))
    test_accuracy = np.empty(len(neighbors))
    
    for i, k in enumerate(neighbors):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(train_data, y_train)
        
        train_accuracy[i] = knn.score(train_data, y_train)
        test_accuracy[i] = knn.score(test_data, y_test)

    # Generate plot
    plt.plot(neighbors, test_accuracy, label = 'Testing dataset Accuracy')
    plt.plot(neighbors, train_accuracy, label = 'Training dataset Accuracy')
    
    plt.legend()
    plt.xlabel('n_neighbors')
    plt.ylabel('Accuracy')
    plt.show()


def confusionMatrix(y_test, y_pred):
    #compute the confusion matrix.
    cm = confusion_matrix(y_test,y_pred)
    
    #Plot the confusion matrix.
    sns.heatmap(cm,
                annot=True,
                fmt='g',
    )
    plt.ylabel('Prediction',fontsize=13)
    plt.xlabel('Actual',fontsize=13)
    plt.title('Confusion Matrix',fontsize=17)
    plt.show()


def main():
    train_data, y_train, test_data, y_test = loadData()

    #k = findKValue(train_data, y_train, test_data, y_test)

    accuracy_logreg, scores_logreg, y_pred_logreg = logisticRegression(train_data, y_train, test_data, y_test)
    accuracy_knn, scores_knn, y_pred_knn = kNearestNeighbour(train_data, y_train, test_data, y_test, 9)
    accuracy_svm, scores_svm, y_pred_svm = supportVectorMachine(train_data, y_train, test_data, y_test)
    #SVM takes a very long time to run

    #Plotting confusion matrices
    confusionMatrix(y_test, y_pred_logreg) #Logistic Regression
    confusionMatrix(y_test, y_pred_knn) #K Nearest Neighbour
    confusionMatrix(y_test, y_pred_svm) #Support Vector Machine

    #Displaying Accuracy
    print("\n")
    print(f"Logistic Regression: Accuracy = {accuracy_logreg*100}, Statistics: {scores_logreg.min(), scores_logreg.mean(), scores_logreg.max()}")
    print(f"K-Nearest Neighbour: Accuracy = {accuracy_knn*100}, Statistics: {scores_knn.min(), scores_knn.mean(), scores_knn.max()}")
    print(f"SVM: Accuracy = {accuracy_svm*100}, Statistics: {scores_svm.min(), scores_svm.mean(), scores_svm.max()}")

if __name__=="__main__":
	main()
