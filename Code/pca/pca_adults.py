import pandas as pd
from sklearn.decomposition import PCA
from Code.preprocessing.outlier_detection import remove_outliers
from Code.preprocessing.variance_thresholding import variance
from Code.preprocessing.correlation import correlation
import numpy as np
from Code.plotting.plots import plot_barh
if __name__ == '__main__':
    name = "pca_adults"
    df = pd.read_excel('C:/Users/annin/PycharmProjects/Master-Degree-Thesis/Code/data/Dataset-onlynormalizedlr.xlsx')
    df.drop(['Unnamed: 0', 'Height (cm)'], axis=1, inplace=True)

    # Check for outliers and remove them
    df = remove_outliers(df)
    # Feature Correlation
    df = correlation(df)

    data = df.iloc[:, :-1]
    labels = df.iloc[:, -1]

    adults = data.iloc[102:202]
    print(adults.to_markdown())
    adults.drop(['Patient', 'Exercise'], axis=1, inplace=True)


    clm = adults.columns
    pca = PCA(n_components=10,copy=True, whiten=True, svd_solver='auto', tol=0.0, iterated_power='auto',
                              n_oversamples=10, power_iteration_normalizer='auto', random_state=12345)

    adults = pca.fit_transform(adults)
    variance = pca.explained_variance_ratio_ * 100

    most_important = [np.abs(pca.components_[i]).argmax() for i in range(10)]
    most_important_names = [clm[most_important[i]] for i in range(10)]

    # LIST COMPREHENSION HERE AGAIN
    dic = {'PC{}'.format(i): most_important_names[i] for i in range(10)}

    # build the dataframe
    de = pd.DataFrame(dic.items(), columns=['PCA Component', 'Most Important Feature'])
    print(de)
    su = 0
    for v in variance:
        print(f"% Variance Ratio per PC ", v)
        su += v
        print("sum", su)

    print(pca.get_feature_names_out(clm))
    print("shape", pca.components_.shape)
    print(abs(pca.components_))

    for i in range(len(pca.components_)):
        component = abs(pca.components_[i])
        most_important = component.argsort()[::-1]
        most_important_names = np.array(clm)[most_important][:20]
        plot_barh(most_important_names, np.sort(component)[::-1], 'skyblue', name + str(i))