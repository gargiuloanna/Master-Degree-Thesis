import sklearn.metrics as skplt_m
from sklearn.inspection import permutation_importance
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import seaborn as sns
from pylab import rcParams

color = 'skyblue'

def confusion_matrix(classifier, data, labels, cmap, name = "Model_Confusion"):

    lab_pred = classifier.predict(data)

    plt.figure(figsize = (34,34))
    cm = skplt_m.confusion_matrix(labels, lab_pred)
    disp = skplt_m.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels = ['Elderly', 'Parkinson', 'Adults'])
    disp.plot(cmap = cmap)
    if not os.path.exists("C:/Users/annin/PycharmProjects/Master-Degree-Thesis/Code/results/plots/" + name):
        os.mkdir("C:/Users/annin/PycharmProjects/Master-Degree-Thesis/Code/results/plots/" + name)

    plt.savefig("C:/Users/annin/PycharmProjects/Master-Degree-Thesis/Code/results/plots/" + name +"_CONFUSION.png")


def plot_barh(x, y, color, name):
    zipped_feats = zip(x, y)
    zipped_feats = sorted(zipped_feats, key=lambda x: x[1])
    features, importances = zip(*zipped_feats)

    #plt.figure(figsize=(50, 24))
    plt.figure(figsize=(40, 32))
    plt.title('Feature Importances' +  name)
    plt.barh(range(len(features)), importances, height=0.6, color=color, align='center')
    plt.yticks(range(len(importances)), features)
    plt.yticks(fontsize=40)
    plt.xticks(fontsize=40)
    plt.xlabel('Relative Importance',fontsize=50)
    plt.tight_layout()
    if not os.path.exists("C:/Users/annin/PycharmProjects/Master-Degree-Thesis/Code/results/plots/" + name):
        os.mkdir("C:/Users/annin/PycharmProjects/Master-Degree-Thesis/Code/results/plots/" + name)

    plt.savefig("C:/Users/annin/PycharmProjects/Master-Degree-Thesis/Code/results/plots/" + name + "-BARPLOT.png", bbox_inches='tight')



def feature_importance(classifier, columns, color, threshold = 15, name = "Model_BARPLOT"):

    importances = classifier.feature_importances_
    indices = np.argsort(importances)[::-1][:threshold]
    feature_names = np.array(columns)[indices]
    plot_barh(feature_names, np.sort(importances)[::-1], color, name)

def random_forest_fimp(classifier, columns, color, name = "Model_FEATUREIMPORTANCE"):

    importances = classifier.feature_importances_
    indices = np.argsort(importances)[::-1]
    std = np.argsort(np.std([tree.feature_importances_ for tree in classifier.estimators_], axis=0))[::-1]
    forest_importances = pd.Series(indices, index=np.array(columns)[indices])
    fig, ax = plt.subplots()
    forest_importances.plot.bar(yerr=std, ax=ax, color = color)
    ax.set_title("Feature importances using MDI")
    ax.set_ylabel("Mean decrease in impurity")
    fig.tight_layout()

    if not os.path.exists("C:/Users/annin/PycharmProjects/Master-Degree-Thesis/Code/results/plots/" + name):
        os.mkdir("C:/Users/annin/PycharmProjects/Master-Degree-Thesis/Code/results/plots/" + name)

    plt.savefig("C:/Users/annin/PycharmProjects/Master-Degree-Thesis/Code/results/plots/" + name + "-FEATUREIMPORTANCE.png",bbox_inches='tight')


def permutation_imp(classifier, data, label, color, name = "Model_Permutation"):

    result = permutation_importance(classifier, data, label, n_repeats=20, random_state=0)
    forest_importances_indices = np.argsort(result.importances_mean)[::-1][:15]
    feature_names = np.array(data.columns)[forest_importances_indices]
    forest_importances = pd.Series(result.importances_mean[forest_importances_indices], index=feature_names)
    '''
    fig, ax = plt.subplots(figsize = (32,32))
    forest_importances.plot.bar(ax=ax, fontsize=40, color = color)
    ax.set_title("Feature importances using permutation on full model")
    ax.set_ylabel("Mean accuracy decrease")
    fig.tight_layout()
    '''
    plot_barh(feature_names, forest_importances, color, name)


    if not os.path.exists("C:/Users/annin/PycharmProjects/Master-Degree-Thesis/Code/results/plots/" + name):
        os.mkdir("C:/Users/annin/PycharmProjects/Master-Degree-Thesis/Code/results/plots/" + name)

    plt.savefig("C:/Users/annin/PycharmProjects/Master-Degree-Thesis/Code/results/plots/" + name + "-PERMUTATION.png",bbox_inches='tight')



def plot_correlation(data, name = 'Model_Correlation'):
    rcParams['figure.figsize'] = 100, 150
    fig = plt.figure()
    sns.heatmap(data, annot=True, fmt=".2f", cmap="crest")

    if not os.path.exists("C:/Users/annin/PycharmProjects/Master-Degree-Thesis/Code/results/plots/" + name):
        os.mkdir("C:/Users/annin/PycharmProjects/Master-Degree-Thesis/Code/results/plots/" + name)

    fig.savefig('C:/Users/annin/PycharmProjects/Master-Degree-Thesis/Code/results/plots/' + name + '.png',bbox_inches='tight')

def svm_importances(svm, data, name = "SVM_Importances"):
    for i in range(3):
        importances = np.sort(abs(svm.coef_[i]))[::-1]
        indices = np.argsort(svm.coef_[i])[::-1][:20]
        feature_names = np.array(data.columns)[indices]
        plot_barh(feature_names, importances, 'skyblue', name + '_' + str(svm.classes_[i]))

def knn(clf, Data, labels, score):
    for i in range(0, len(Data.columns) -1):
        plt.scatter(Data.iloc[:,i], Data.iloc[:,i+1], c=labels, cmap=ListedColormap(["#FF0000", "#00FF00", "#0000FF"]), edgecolor="k", s=20)
        plt.xlabel(Data.columns[i])
        plt.ylabel(Data.columns[i+1])
        plt.title("{} (k = {})".format("KNN", 1))
        plt.text(
                0.9,
                0.1,
                "{:.2f}".format(score),
                size=15,
                ha="center",
                va="center",
                transform=plt.gca().transAxes,
        )

        plt.show()

