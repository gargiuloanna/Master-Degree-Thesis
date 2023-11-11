import pandas as pd
from Code.preprocessing.outlier_detection import remove_outliers
from Code.preprocessing.correlation import correlation

def preprocess(file = 'C:/Users/annin/PycharmProjects/Master-Degree-Thesis/Code/data/Dataset-fallrisk - TUGNOSIT.xlsx'):

    df = pd.read_excel(file)
    df.drop(['Height (cm)'], axis=1, inplace=True)

    #Check for outliers and remove them
    df = remove_outliers(df)

    # Feature Correlation
    df = correlation(df)
    data = df.iloc[:, :-1]
    labels = df.iloc[:, -1]
    # Split data

    return data, labels




