from joblib import load
from ..evaluation.predict import predict_score
import pandas as pd
from sklearn.model_selection import train_test_split
import os
from ..preprocessing.scale import scale

if __name__ == '__main__':
    for root, dirs, files in os.walk("./Models"):
        for model in files:
            if model.endswith('.joblib'):
                md = load(os.path.join("./Models", model))
                print("Loaded ", md)

                df = pd.read_excel('C:/Users/annin/PycharmProjects/Master-Degree-Thesis/Code/Data/Dataset -labels.xlsx')
                df.drop(['Unnamed: 0', 'Patient', 'Exercise'], axis=1, inplace=True)

                data = df.iloc[:, :-1]
                labels = df.iloc[:, -1]

                train, test, labeltrain, labeltest = train_test_split(data, labels, test_size=0.30, random_state=12345, stratify=labels)
                train, test = scale(train, test)

                file = open("performances.txt", "a")
                file.write("Test Accuracy: " + str(predict_score(md, test, labeltest)) + '\n')
