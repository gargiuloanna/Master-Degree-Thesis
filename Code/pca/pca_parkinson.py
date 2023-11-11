import pandas as pd
from sklearn.decomposition import PCA
from Code.preprocessing.outlier_detection import remove_outliers
from Code.preprocessing.variance_thresholding import variance
from Code.preprocessing.correlation import correlation
import numpy as np
from Code.plotting.plots import plot_barh
import matplotlib.pyplot as plt

def myplot(score, coeff, labels=None):
    xs = score[:, 0]
    ys = score[:, 1]
    n = coeff.shape[0]
    scalex = 1.0 / (xs.max() - xs.min())
    scaley = 1.0 / (ys.max() - ys.min())
    plt.scatter(xs * scalex, ys * scaley)
    for i in range(n):
        plt.arrow(0, 0, coeff[i, 0], coeff[i, 1], color='r', alpha=0.5)
        if labels is None:
            plt.text(coeff[i, 0] * 1.15, coeff[i, 1] * 1.15, "Var" + str(i + 1), color='g', ha='center', va='center')
        else:
            plt.text(coeff[i, 0] * 1.15, coeff[i, 1] * 1.15, labels[i], color='g', ha='center', va='center')


if __name__ == '__main__':
    name = "pca_parkinson"
    df = pd.read_excel('C:/Users/annin/PycharmProjects/Master-Degree-Thesis/Code/data/Dataset-onlynormalizedlr.xlsx')
    df.drop(['Unnamed: 0', 'Height (cm)'], axis=1, inplace=True)

    # Check for outliers and remove them
    df = remove_outliers(df)
    # Feature Correlation
    df = correlation(df)

    data = df.iloc[:, :-1]
    labels = df.iloc[:, -1]

    parkinson =data.iloc[71:102]
    print(parkinson.to_markdown())
    parkinson.drop(['Patient', 'Exercise'], axis=1, inplace=True)


    n_components = 10
    clm = parkinson.columns
    pca = PCA(n_components=n_components,copy=True, whiten=True, svd_solver='auto', tol=0.0, iterated_power='auto',
                              n_oversamples=10, power_iteration_normalizer='auto', random_state=12345)

    parkinson = pca.fit_transform(parkinson)

    variance = pca.explained_variance_ratio_ * 100

    most_important = [np.abs(pca.components_[i]).argmax() for i in range(n_components)]
    most_important_names = [clm[most_important[i]] for i in range(n_components)]

    # LIST COMPREHENSION HERE AGAIN
    dic = {'PC{}'.format(i): most_important_names[i] for i in range(n_components)}

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
        plt.close()

    plt.figure()
    colors = ["navy", "turquoise", "darkorange"]
    lw = 2


    plt.scatter(parkinson[:, 0],parkinson[:, 1], color="turquoise", alpha=0.8, lw=lw, label=1)
    plt.legend(loc="best", shadow=False, scatterpoints=1)
    plt.title("PCA of IRIS dataset")
    plt.show()
