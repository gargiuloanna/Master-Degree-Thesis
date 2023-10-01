import scikitplot.metrics as skplt_m
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

def confusion_matrix(classifier, data, labels, name = "RandomForestGridSearch"):
    lab_pred = classifier.predict(data)
    skplt_m.plot_confusion_matrix(labels, lab_pred)
    plt.savefig("C:/Users/annin/PycharmProjects/Tesi/Data/Plots/" + name + "/" + name +"_CONFUSION.png")
    plt.show()

def plot_barh(x, y, name):
    zipped_feats = zip(x, y)
    zipped_feats = sorted(zipped_feats, key=lambda x: x[1])
    features, importances = zip(*zipped_feats)

    print(features)
    print(importances)

    plt.figure(figsize=(32, 12))
    plt.title('Feature Importances' +  name)
    plt.barh(range(len(features)), importances, height=0.6, color='#D8BFD8', align='center')
    plt.yticks(range(len(importances)), features)
    plt.xlabel('Relative Importance')
    plt.savefig("C:/Users/annin/PycharmProjects/Tesi/Data/Plots/"+ name + "/" + name + "-BARPLOT.png")
    #plt.savefig("C:/Users/annin/PycharmProjects/Tesi/Data/Plots/" + name + "-BARPLOT.png")
    plt.show()

def feature_importance(classifier, columns, threshold = 30, name = "RandomForestGridSearch-BARPLOT"):
    importances = classifier.feature_importances_
    indices = np.argsort(importances)[::-1][:threshold]
    feature_names = np.array(columns)[indices]
    plot_barh(feature_names, np.sort(importances)[::-1], name)

def random_forest_fimp(classifier, columns, name = "RandomForestGridSearch-FEATUREIMPORTANCE"):
    importances = classifier.feature_importances_
    indices = np.argsort(importances)[::-1]
    std = np.argsort(np.std([tree.feature_importances_ for tree in classifier.estimators_], axis=0))[::-1]
    forest_importances = pd.Series(indices, index=np.array(columns)[indices])
    fig, ax = plt.subplots()
    forest_importances.plot.bar(yerr=std, ax=ax)
    ax.set_title("Feature importances using MDI")
    ax.set_ylabel("Mean decrease in impurity")
    fig.tight_layout()
    plt.savefig("C:/Users/annin/PycharmProjects/Tesi/Data/Plots/" + name + "/" + name + "-FEATUREIMPORTANCE.png")
    plt.show()

def permutation_imp(classifier, data, label, name = "RandomForestPermutation"):
    result = permutation_importance(classifier, data, label, n_repeats=10, random_state=0)
    #forest_importances = pd.Series((result.importances_mean), index=data.columns)
    forest_importances_indices = np.argsort(result.importances_mean)[::-1]
    feature_names = np.array(data.columns)[forest_importances_indices]
    forest_importances = pd.Series(result.importances_mean[forest_importances_indices], index=feature_names)
    fig, ax = plt.subplots()
    forest_importances.plot.bar(ax=ax)
    ax.set_title("Feature importances using permutation on full model")
    ax.set_ylabel("Mean accuracy decrease")
    fig.tight_layout()
    plt.savefig("C:/Users/annin/PycharmProjects/Tesi/Data/Plots/" + name + "/" + name + "-PERMUTATION, sorted.png")
    plt.show()



