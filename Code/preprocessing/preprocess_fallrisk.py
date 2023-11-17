import pandas as pd
from Code.preprocessing.biological_data import get_age
from Code.preprocessing.outlier_detection import remove_outliers
from Code.preprocessing.correlation import correlation
from Code.preprocessing.variance_thresholding import variance
from Code.preprocessing.scale import scale
from sklearn.model_selection import train_test_split

f = 'C:/Users/annin/PycharmProjects/Master-Degree-Thesis/Code/data/Dataset-fallrisk_complete - TUG.xlsx'
def preprocess(file = f):

    df = pd.read_excel(file)
    df.drop(['Unnamed: 0', 'Height (cm)', 'train test'], axis=1, inplace=True)
    #Check for outliers and remove them
    df = remove_outliers(df)
    # Feature Correlation
    df = correlation(df)

    data = df.iloc[:, :-1]
    labels = df.iloc[:, -1]

    train, test, labeltrain, labeltest = train_test_split(data, labels, test_size=0.30, random_state=12345, stratify=labels)

    #Print test samples
    sav = []
    for patient in test['Patient']:
        sav.append(get_age(patient))
    to_print = pd.DataFrame()
    to_print['Patient'] = test['Patient']
    to_print['Task'] = test['Exercise']
    to_print['Age'] = sav
    to_print['Label'] = labeltest

    # Scale data
    train, test = scale(train, test)
    # Remove features with variance <0.4
    train, test = variance(train, test, threshold=0.4)

    return train, test, labeltrain, labeltest, to_print


def preprocess_folds(file = f):
    df = pd.read_excel(file)
    df.drop(['Unnamed: 0', 'Height (cm)', 'train test'], axis=1, inplace=True)

    # Check for outliers and remove them
    df = remove_outliers(df)
    # Feature Correlation
    df = correlation(df)

    data = df.iloc[:, :-1]
    labels = df.iloc[:, -1]

    return data, labels


