from sklearn.feature_selection import VarianceThreshold
import pandas as pd

def variance(data, threshold = 0.0):
    selector = VarianceThreshold(threshold = threshold)

    col = pd.DataFrame()
    col['Patient'] = data['Patient']
    col['Task'] = data['Exercise']
    data.drop(['Patient', 'Exercise'], axis=1, inplace=True)

    data_transformed = selector.fit_transform(data)
    data_transformed = pd.DataFrame(data_transformed, columns = selector.get_feature_names_out())

    data_transformed.insert(loc=0, column='Patient', value=col['Patient'], allow_duplicates=True)
    data_transformed.insert(loc=1, column='Exercise', value=col['Task'], allow_duplicates=True)

    print("Variance Thresholding removed ", len(data.columns)-len(selector.get_feature_names_out()), " features")
    return data_transformed