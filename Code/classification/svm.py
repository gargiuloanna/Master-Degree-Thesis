from sklearn.svm import LinearSVC, SVC
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from preprocessing.scale import scale
from evaluation.predict import predict_score
import os
from plotting.plots import confusion_matrix
from sklearn.inspection import DecisionBoundaryDisplay
import matplotlib.pyplot as plt
import numpy as np

def plot_training_data_with_decision_boundary(kernel):
    # Train the SVC
    df = pd.read_excel('C:/Users/annin/PycharmProjects/Tesi/Data/Dataset -labels.xlsx')
    df.drop(['Unnamed: 0', 'Patient', 'Exercise'], axis=1, inplace=True)

    data = df.iloc[:, 65:67]
    labels = df.iloc[:, -1]
    train, test, labeltrain, labeltest = train_test_split(data, labels, test_size=0.25, random_state=12345, stratify=labels)
    train, test = scale(train, test)
    X = train
    y = labeltrain
    clf = SVC(degree = 5, kernel=kernel, gamma=2).fit(X, y)
    print("Test Accuracy: ", predict_score(clf, test, labeltest))
    # Settings for plotting
    _, ax = plt.subplots(figsize=(4, 3))
    x_min, x_max, y_min, y_max = -3, 3, -3, 3
    ax.set(xlim=(x_min, x_max), ylim=(y_min, y_max))

    # Plot decision boundary and margins
    common_params = {"estimator": clf, "X": X, "ax": ax}
    DecisionBoundaryDisplay.from_estimator(
        **common_params,
        response_method="predict",
        plot_method="pcolormesh",
        alpha=0.3,
    )
    DecisionBoundaryDisplay.from_estimator(
        **common_params,
        response_method="auto",
        plot_method="contour",
        levels=[-1, 0, 1],
        colors=["k", "k", "k"],
        linestyles=["--", "-", "--"],
    )

    # Plot bigger circles around samples that serve as support vectors
    ax.scatter(
        clf.support_vectors_[:, 0],
        clf.support_vectors_[:, 1],
        s=250,
        facecolors="none",
        edgecolors="k",
    )
    # Plot samples by color and add legend
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, s=150, edgecolors="k")
    ax.legend(*scatter.legend_elements(), loc="upper right", title="Classes")
    ax.set_title(f" Decision boundaries of {kernel} kernel in SVC")

    plt.show()

if __name__ == "__main__":
    name = "SVM-test20%-refit"
    df = pd.read_excel('C:/Users/annin/PycharmProjects/Tesi/Data/Dataset -labels.xlsx')
    df.drop(['Unnamed: 0', 'Patient', 'Exercise'], axis=1, inplace=True)

    data = df.iloc[:, :-1]
    labels = df.iloc[:, -1]
    train, test, labeltrain, labeltest = train_test_split(data, labels, test_size=0.2, random_state=12345, stratify=labels)
    print(test)
    train, test = scale(train, test)
    print("[SVM] Searching best params with GridSearchCV")
    svm = SVC(random_state=0, class_weight='balanced')
    param_grid = {
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'C': [0.1,0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0],
        'degree': [1,2, 3, 5, 7],
        'coef0': [0.0,1,2,3,4, 5, 6,7,8,9,10],
        'decision_function_shape':['ovo'],
    }
    svm_gridcv = GridSearchCV(svm, param_grid=param_grid, cv=5, scoring='balanced_accuracy', error_score=0, n_jobs=-1, verbose=3, refit=True)
    svm_gridcv.fit(train, labeltrain)
    print(f"[SVM] Best SVM model with params: {svm_gridcv.best_params_} and score: {svm_gridcv.best_score_:.3f}")
    print("Test Accuracy: ", predict_score(svm_gridcv.best_estimator_, test, labeltest))
    file = open("performances.txt", "a")
    file.write("Model " + name + '\n')
    file.write("Train Accuracy: " + str(svm_gridcv.best_score_) + '\n')
    file.write("Test Accuracy: " + str(predict_score(svm_gridcv.best_estimator_, test, labeltest)) + '\n')
    file.close()
    '''
    for i in range(3):
        importances = np.sort(abs(svm.coef_[i]))[::-1]
        indices = np.argsort(svm.coef_[i])[::-1][:20]
        feature_names = np.array(data.columns)[indices]
        plot_barh(feature_names, importances, name + '_' + str(svm.classes_[i]))
    '''
    y_test = np.asarray(labeltest)
    misclassified = np.where(y_test != svm_gridcv.best_estimator_.predict(test))
    for ind in misclassified[0]:
        print(ind)

    #os.mkdir("C:/Users/annin/PycharmProjects/Tesi/Data/Plots/" + name)
    confusion_matrix(svm_gridcv.best_estimator_, test, labeltest, name=name)