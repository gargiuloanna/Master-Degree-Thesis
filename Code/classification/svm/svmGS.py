import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from Code.evaluation.predict import predict_score
from Code.plotting.plots import confusion_matrix, svm_importances
from Code.preprocessing.preprocess import preprocess


if __name__ == "__main__":
    name = "svm/SVM_GS"

    train, test, labeltrain, labeltest = preprocess()


    print("[SVM] Searching best params with GridSearchCV")

    svm = SVC(random_state=0, class_weight='balanced')

    param_grid = {
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'C': [0.1,0.2, 0.3, 0.4, 0.5, 0.6, 0.65, 0.7,0.75, 0.8, 0.9, 1.0, 1.5, 2.0],
        'degree': [1,2, 3, 5, 7, 8, 9, 10],
        'coef0': [0.0,1,2,3,4,4.5, 4.6, 4.7, 4.8, 4.9, 5,5.5, 5.6, 5.7, 5.8, 6,7,8,9,10],
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



    #[SVM] Best SVM model with params: {'C': 0.7, 'coef0': 4.9, 'decision_function_shape': 'ovo', 'degree': 3, 'kernel': 'poly'} and score: 0.841
    #Test Accuracy:  0.7984297661717017