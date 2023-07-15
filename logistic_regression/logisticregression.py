import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


def loadData():
    heart_data = pd.read_csv('heart.csv')
    
    # scaler = StandardScaler()
    # x = scaler.fit_transform(x)

    training, test = train_test_split(heart_data, test_size=0.20, random_state = 20)

    x_train = heart_data.drop(['target'], axis=1)
    y_train = pd.array(heart_data['target'])
    x_test = test
    y_test = test.pop("target")

    return x_train, y_train, x_test, y_test


def main():
    x_train, y_train, x_test, y_test = loadData()
    clf = LogisticRegression(random_state=0, max_iter=2000).fit(x_train, y_train)

    y_pred = clf.predict(x_test)
    accuracy= accuracy_score(y_test,y_pred)

    print(f"Accuracy: {accuracy}")
    print(f"Score: {clf.score(x_test,y_test)}")
    print(f"F1 Score: {f1_score(y_train, clf.predict(x_train), average=None)}")

if __name__=="__main__":
	main()
