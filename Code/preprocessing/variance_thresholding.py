from sklearn.feature_selection import VarianceThreshold
import pandas as pd
import numpy as np

def variance(train, test, threshold = 0.0):
    selector = VarianceThreshold(threshold = threshold)


    train.drop(['Patient', 'Exercise'], axis=1, inplace=True)
    test.drop(['Patient', 'Exercise'], axis=1, inplace=True)
    selector = selector.fit(train)
    #for i in range(len(selector.variances_)):
    #    print(f"{selector.feature_names_in_[i]} {selector.variances_[i]:.2f}")
    train_transformed = selector.transform(train)
    train_transformed = pd.DataFrame(train_transformed, columns = selector.get_feature_names_out())

    test_transformed = selector.transform(test)
    test_transformed = pd.DataFrame(test_transformed, columns=selector.get_feature_names_out())
    #train_transformed.insert(loc=0, column='Patient', value=col['Patient'], allow_duplicates=True)
    #train_transformed.insert(loc=1, column='Exercise', value=col['Task'], allow_duplicates=True)

    print("Variance Thresholding removed ", len(train.columns)-len(selector.get_feature_names_out()), " features")
    print("Features Removed: ", set(train.columns).difference(set(selector.get_feature_names_out())))
    return train_transformed,test_transformed