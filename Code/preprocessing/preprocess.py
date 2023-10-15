import pandas as pd
from Code.preprocessing.outlier_detection import remove_outliers
from Code.preprocessing.variance_thresholding import variance
from Code.preprocessing.scale import scale
from Code.preprocessing.correlation import correlation
from sklearn.model_selection import train_test_split

def preprocess(file = 'C:/Users/annin/PycharmProjects/Master-Degree-Thesis/Code/Data/Dataset-only normalized lengths -lr.xlsx'):

    df = pd.read_excel(file)
    df.drop(['Unnamed: 0'], axis=1, inplace=True)

    #Check for outliers and remove them
    df = remove_outliers(df)

    #Remove features with variance <0.01
    df = variance(df, threshold = 0.0001)

    #Feature Correlation
    correlation(df)

    data = df.iloc[:, :-1]
    labels = df.iloc[:, -1]

    #Split Data
    train, test, labeltrain, labeltest = train_test_split(data, labels, test_size=0.30, random_state=12345, stratify=labels)

    #Scale Data
    train, test = scale(train, test)

    return train, test, labeltrain, labeltest


