from sklearn.feature_selection import VarianceThreshold
import pandas as pd

def variance(data, threshold = 0.0):
    selector = VarianceThreshold(threshold = threshold)
    data_transformed = selector.fit_transform(data)
    data_transformed = pd.DataFrame(data_transformed, columns = selector.get_feature_names_out())
    print("Before--> ", len(data.columns), "\nAfter--> ", len(selector.get_feature_names_out()))
    return data_transformed