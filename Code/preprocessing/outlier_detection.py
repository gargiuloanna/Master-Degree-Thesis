from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import pandas as pd

def remove_outliers(df):
    forest = IsolationForest(n_estimators=100, max_samples='auto', contamination='auto', max_features=1.0,
                             bootstrap=False, n_jobs=-1, random_state=12345, verbose=0, warm_start=False)
    outlier = LocalOutlierFactor(n_neighbors=30, algorithm='auto', leaf_size=30, metric='minkowski', p=2,
                                 metric_params=None, contamination='auto', novelty=False, n_jobs=-1)


    col = pd.DataFrame()
    col['Patient'] = df['Patient']
    col['Task'] = df['Exercise']
    df.drop(['Patient', 'Exercise'], axis = 1, inplace = True)
    data = df.iloc[:, :-1]

    outfor = forest.fit_predict(data)
    outloc = outlier.fit_predict(data)

    df.insert(loc=0, column='Patient', value=col['Patient'], allow_duplicates=True)
    df.insert(loc=1, column='Exercise', value=col['Task'], allow_duplicates=True)
    for  i in range(len(data)):
        if outfor[i] == -1 and outloc[i] == -1 and not df.loc[i][0].startswith("pd"):
            print("Removed sample: ", df.loc[i][0], df.loc[i][1] )
            df.drop(i, axis = 0, inplace = True)

    return df

