import pandas as pd
from sklearn.preprocessing import RobustScaler


def scale(data_train, data_test):

    scaler = RobustScaler()
    data_train.drop(['Patient', 'Exercise'], axis=1, inplace=True)
    data_test.drop(['Patient', 'Exercise'], axis=1, inplace=True)

    columns = data_train.columns
    scaler = scaler.fit(data_train)
    data_train = scaler.transform(data_train)
    data_test = scaler.transform(data_test)
    data_train = pd.DataFrame(data_train, columns = columns)
    data_test = pd.DataFrame(data_test, columns = columns)
    print("Scaled Dataset")
    return data_train,data_test

