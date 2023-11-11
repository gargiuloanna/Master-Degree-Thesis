import pandas as pd
import math
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from Code.preprocessing.outlier_detection import remove_outliers
from Code.preprocessing.variance_thresholding import variance
from Code.preprocessing.scale import scale
from Code.preprocessing.correlation import correlation
from sklearn.model_selection import train_test_split
def gaussian():
    mu = 0
    variance = 1
    sigma = math.sqrt(variance)
    x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
    plt.plot(x, scipy.stats.norm.pdf(x, mu, sigma))

    plt.show()

if __name__ == '__main__':
    file = '/Code/data/Dataset-onlynormalizedlr.xlsx'

    df = pd.read_excel(file)
    df.drop(['Unnamed: 0', 'Height (cm)'], axis=1, inplace=True)
    # Check for outliers and remove them
    df = remove_outliers(df)

    # Feature Correlation
    df = correlation(df)

    data = df.iloc[:, :-1]
    labels = df.iloc[:, -1]
    # Split data
    train, test, labeltrain, labeltest = train_test_split(data, labels, test_size=0.30, random_state=12345, stratify=labels)

    # Scale data
    train, test = scale(train, test)
    col = pd.DataFrame()
    col['Patient Train'] = train['Patient']
    col['Task Train'] = train['Exercise']
    col['Patient Test'] = test['Patient']
    col['Task Test'] = test['Exercise']
    # Remove features with variance <0.01
    train, test = variance(train, test, threshold=0.4)

    train.insert(loc=0, column='Patient', value=col['Patient Train'], allow_duplicates=True)
    train.insert(loc=1, column='Exercise', value=col['Task Train'], allow_duplicates=True)
    test.insert(loc=0, column='Patient', value=col['Patient Test'], allow_duplicates=True)
    test.insert(loc=1, column='Exercise', value=col['Task Test'], allow_duplicates=True)
    print(labeltrain.loc[labeltrain == 0].index)
    print(labeltrain)
    print(train)
    elderly = train.iloc[labeltrain.loc[labeltrain == 0].index]
    parkinson = train.iloc[labeltrain.loc[labeltrain == 1].index]
    adults = train.iloc[labeltrain.loc[labeltrain == 2].index]

    np.set_printoptions(precision=2)

    for column in train.columns:
        if not column.endswith("Patient") and not column.endswith("Exercise"):
            print(column)
            print("Elderly\t\t\t\t\t\t\t\tParkinson\t\t\t\t\t\t\t\tAdults")

            mean_elderly = str(np.around(np.mean(elderly[column].values), 3))
            std_elderly =str(np.around(np.std(elderly[column].values), 3))

            mean_parkinson = str(np.around(np.mean(parkinson[column].values), 3))
            std_parkinson = str(np.around(np.std(parkinson[column].values), 3))

            mean_adults = str(np.around(np.mean(adults[column].values), 3))
            std_adults = str(np.around(np.std(adults[column].values), 3))

            print("MEAN ", mean_elderly,"\t",mean_parkinson,"\t",mean_adults)
            print("STD ", std_elderly, "\t", std_parkinson, "\t", std_adults)
            fig, (ax1,ax2,ax3) = plt.subplots(3,1, figsize = (32,32))
            fig.suptitle(column, fontsize = 40)
            ax1.hist(elderly[column].values,linewidth=0.5, edgecolor="white", color = 'g',
                     label = "Elderly - μ " + mean_elderly + " - σ " + std_elderly)
            ax2.hist(parkinson[column].values, linewidth=0.5, edgecolor="white", color = 'y',
                     label = "Parkinson - μ " + mean_parkinson + " - σ " + std_parkinson)
            ax3.hist(adults[column].values, linewidth=0.5, edgecolor="white", color = 'b',
                     label = "Adults - μ " + mean_adults + " - σ " + std_adults)
            ax1.legend(fontsize = 40)
            ax2.legend(fontsize = 40)
            ax3.legend(fontsize = 40)
            fig.tight_layout()
            plt.savefig("C:/Users/annin/PycharmProjects/Master-Degree-Thesis/Code/results/plots/STATS/" + column + ".png")
