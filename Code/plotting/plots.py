import sklearn.metrics as skplt_m
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import seaborn as sns
from pylab import rcParams

def confusion_matrix_fallrisk(classifier, data, labels, cmap, name = "Model_Confusion"):

    lab_pred = classifier.predict(data)

    plt.figure(figsize = (34,34))
    cm = skplt_m.confusion_matrix(labels, lab_pred)
    disp = skplt_m.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels = ['Not At Risk','At Risk'])
    disp.plot(cmap = cmap)
    if not os.path.exists("C:/Users/annin/PycharmProjects/Master-Degree-Thesis/Code/results/plots/" + name):
        os.mkdir("C:/Users/annin/PycharmProjects/Master-Degree-Thesis/Code/results/plots/" + name)

    plt.savefig("C:/Users/annin/PycharmProjects/Master-Degree-Thesis/Code/results/plots/" + name +"_CONFUSION.png")

def confusion_matrix_gait(classifier, data, labels, cmap, name = "Model_Confusion"):

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


def permutation_imp(classifier, data, label, color, name = "Model_Permutation"):

    result = permutation_importance(classifier, data, label, n_repeats=20, random_state=0)
    forest_importances_indices = np.argsort(result.importances_mean)[::-1][:15]
    feature_names = np.array(data.columns)[forest_importances_indices]
    forest_importances = pd.Series(result.importances_mean[forest_importances_indices], index=feature_names)
    plot_barh(feature_names, forest_importances, color, name)

def plot_correlation(data, name = 'Model_Correlation'):
    rcParams['figure.figsize'] = 100, 150
    fig = plt.figure()
    sns.heatmap(data, annot=True, fmt=".2f", cmap="crest")

    if not os.path.exists("C:/Users/annin/PycharmProjects/Master-Degree-Thesis/Code/results/plots/" + name):
        os.mkdir("C:/Users/annin/PycharmProjects/Master-Degree-Thesis/Code/results/plots/" + name)

    fig.savefig('C:/Users/annin/PycharmProjects/Master-Degree-Thesis/Code/results/plots/' + name + '.png',bbox_inches='tight')

def svm_importances(svm, train, labeltrain, name = "SVM_Importances"):

    perm_importance = permutation_importance(svm, train, labeltrain)

    # Making the sum of feature importance being equal to 1.0,
    # so feature importance can be understood as percentage
    perm_importance_normalized = perm_importance.importances_mean / perm_importance.importances_mean.sum()

    # Feature's name (considering your X a DataFrame)
    feature_names = train.columns

    # Sort to plot in order of importance
    sorted_idx = np.argsort(perm_importance_normalized)[::-1][:15]
    features = np.array(feature_names)[sorted_idx]
    plot_barh(features, np.sort(perm_importance_normalized)[::-1], 'chocolate', name)


