import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from Code.evaluation.predict import predict_score
from Code.plotting.plots import confusion_matrix, svm_importances
from Code.preprocessing.preprocess import preprocess


if __name__ == "__main__":
    name = "svm/SVM_GS"

    #train, test, labeltrain, labeltest = preprocess()
    train, test, labeltrain, labeltest = \
        preprocess(file='C:\\Users\\annin\PycharmProjects\Master-Degree-Thesis\Code\Data\Dataset-fallrisk - TUGNOSIT.xlsx')

    print("[SVM] Searching best params with GridSearchCV")

    svm = SVC(random_state=0, class_weight='balanced')

    param_grid = {
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'C': [0.1,0.2, 0.3, 0.4, 0.5, 0.6, 0.65, 0.7,0.75, 0.8, 0.9, 1.0, 1.5, 2.0],
        'degree': [1,2, 3, 5, 7, 8, 9, 10],
        'coef0': [10,11,12,13,14,15,16,17,18,19,10,21,22,23,24,25,26,27,28,29,30],
        'decision_function_shape':['ovo'],
    }

    svm_gridcv = GridSearchCV(svm, param_grid=param_grid, cv=10, scoring='balanced_accuracy', error_score=0, n_jobs=-1, verbose=3, refit=True)
    svm_gridcv.fit(train, labeltrain)

    print(f"[SVM] Best SVM model with params: {svm_gridcv.best_params_} and score: {svm_gridcv.best_score_:.3f}")
    print("Test Accuracy: ", predict_score(svm_gridcv.best_estimator_, test, labeltest))

    file = open("C:/Users/annin/PycharmProjects/Master-Degree-Thesis/Code/results/logs/svm/svmGS.txt", "a")

    file.write(f"[SVM] Best SVM with params: {svm_gridcv.best_params_} and score {svm_gridcv.best_score_}")
    file.write("\nModel " + name + '\n')
    file.write("Test Accuracy: " + str(predict_score(svm_gridcv.best_estimator_, test, labeltest)) + '\n')
    file.close()


