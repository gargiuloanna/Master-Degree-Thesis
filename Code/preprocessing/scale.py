import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import VarianceThreshold
import numpy as np

def scale(data_train, data_test):
    print("[INFO] Scaling dataset...")
    scaler = RobustScaler()
    columns = data_train.columns
    scaler = scaler.fit(data_train)
    data_train = scaler.transform(data_train)
    data_test = scaler.transform(data_test)
    data_train = pd.DataFrame(data_train, columns = columns)
    data_test = pd.DataFrame(data_test, columns = columns)
    print("[INFO] Finished scaling dataset")
    return data_train,data_test



def variance(data, threshold = 0.0):
    selector = VarianceThreshold(threshold = threshold)
    data_transformed = selector.fit_transform(data)
    data_transformed = pd.DataFrame(data_transformed, columns = selector.get_feature_names_out())
    print("Before--> ", len(data.columns), "\nAfter--> ", len(selector.get_feature_names_out()))
    return data_transformed

