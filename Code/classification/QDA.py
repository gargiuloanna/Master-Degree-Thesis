
from sklearn.model_selection import train_test_split, GridSearchCV
import pandas as pd

from Code.plotting.plots import confusion_matrix, plot_barh, random_forest_fimp, permutation_imp, feature_importance, plot_correlation
from Code.preprocessing.scale import scale
from  sklearn.metrics import balanced_accuracy_score
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
if __name__ == '__main__':
    name = "QDAGridSearch"
    df = pd.read_excel('C:/Users/annin/PycharmProjects/Master-Degree-Thesis/Code/Data/Dataset-only normalized lengths -lr.xlsx')
    df.drop(['Unnamed: 0', 'Patient', 'Exercise'], axis=1, inplace=True)

    data = df.iloc[:, :-1]
    labels = df.iloc[:, -1]

    clm = data.columns
    pca = PCA(n_components=10,copy=True, whiten=True, svd_solver='auto', tol=0.0, iterated_power='auto',
                              n_oversamples=10, power_iteration_normalizer='auto', random_state=12345)

    data = pca.fit_transform(data)
    print(pca.explained_variance_)
    '''
    for i in range(len(pca.components_)):
        most_important = pca.components_[i].argsort()[::-1]
        most_important_names = np.array(clm)[most_important][:20]
        plot_barh(most_important_names, np.sort(pca.components_[i])[::-1], name, i)
    '''
    train, test, labeltrain, labeltest = train_test_split(data, labels, test_size=0.30, random_state=12345, stratify=labels)
    #train, test = scale(train, test)
    qda = QuadraticDiscriminantAnalysis()
    '''
    param_grid = {
        'reg_param': [0.0, 0.3, 0.5, 0.7, 1.0],
        'tol': [0,1.0e-8, 1.0e-4, 1.0e-2, 1.0],
    }
    svm_gridcv = GridSearchCV(qda, param_grid=param_grid, cv=10, scoring='balanced_accuracy', error_score=0, n_jobs=-1, verbose=3, refit=True)
    svm_gridcv.fit(train, labeltrain)
    print(f"[SVM] Best SVM model with params: {svm_gridcv.best_params_} and score: {svm_gridcv.best_score_:.3f}")
    qda = svm_gridcv.best_estimator_'''


    qda.fit(train, labeltrain)

    if not os.path.exists("C:/Users/annin/PycharmProjects/Master-Degree-Thesis/Code/Data/Plots/" + name):
        os.mkdir("C:/Users/annin/PycharmProjects/Master-Degree-Thesis/Code/Data/Plots/" + name)
    print(qda.score(test, labeltest))
    confusion_matrix(qda, test, labeltest, name=name)
    plt.close()
