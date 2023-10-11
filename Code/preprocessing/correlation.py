import pandas as pd
from Code.plotting.plots import plot_correlation
import numpy as np



def correlation(dp):

    col = pd.DataFrame()
    col['Patient'] = dp['Patient']
    col['Task'] = dp['Exercise']

    dp.drop(['Patient', 'Exercise'], axis=1, inplace=True)

    data = dp.iloc[:, :-1]

    c = data.corr()
    upper = c.where(np.tril(np.ones(c.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.8)]
    dp.drop(['Gait Speed'], axis=1, inplace=True)

    dp.insert(loc=0, column='Patient', value=col['Patient'], allow_duplicates=True)
    dp.insert(loc=1, column='Exercise', value=col['Task'], allow_duplicates=True)
    print("Removed correlated features")
    plot_correlation(upper, name='corr_label_lr')
    return dp
