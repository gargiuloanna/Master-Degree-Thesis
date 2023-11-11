import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler


def scale(data_train, data_test):

    scaler = RobustScaler()
    #scaler = StandardScaler()
    col = pd.DataFrame()
    col['Patient Train'] = data_train['Patient']
    col['Task Train'] = data_train['Exercise']
    col['Patient Test'] = data_test['Patient']
    col['Task Test'] = data_test['Exercise']

    data_train.drop(['Patient', 'Exercise'], axis=1, inplace=True)
    data_test.drop(['Patient', 'Exercise'], axis=1, inplace=True)

    columns = data_train.columns
    scaler = scaler.fit(data_train)
    data_train = scaler.transform(data_train)
    data_test = scaler.transform(data_test)
    data_train = pd.DataFrame(data_train, columns = columns)
    data_train.insert(loc=0, column='Patient', value=col['Patient Train'], allow_duplicates=True)
    data_train.insert(loc=1, column='Exercise', value=col['Task Train'], allow_duplicates=True)

    data_test = pd.DataFrame(data_test, columns = columns)

    data_test.insert(loc=0, column='Patient', value=col['Patient Test'], allow_duplicates=True)
    data_test.insert(loc=1, column='Exercise', value=col['Task Test'], allow_duplicates=True)
    print("Scaled Dataset")
    return data_train,data_test

