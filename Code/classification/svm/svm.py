from sklearn.svm import SVC
from Code.evaluation.predict import predict_score
from Code.plotting.plots import confusion_matrix, plot_barh
from Code.preprocessing.preprocess import preprocess
from  sklearn.metrics import classification_report
import numpy as np
from joblib import dump
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt



if __name__ == '__main__':

    train, test, labeltrain, labeltest = preprocess()
    name = "svm/SVM_BEST"


    svm = SVC(class_weight='balanced', kernel='poly',
              C=0.7, degree=3, coef0=11, decision_function_shape='ovo')

    svm.fit(train, labeltrain)

    # Performance
    print("Train Accuracy: ", predict_score(svm, train, labeltrain))
    print("Test Accuracy: ", predict_score(svm, test, labeltest))


    print(classification_report(labeltest, svm.predict(test),  target_names=['Elderly', 'Parkinson', 'Adults'],  digits=3, output_dict=False, zero_division='warn'))

    # Write to File
    file = open("C:/Users/annin/PycharmProjects/Master-Degree-Thesis/Code/results/logs/svm/svm.txt", "a")
    file.write("\nModel " + name + '\n')
    file.write("Train Accuracy: " + str(predict_score(svm, train, labeltrain)) + '\n')
    file.write("Test Accuracy: " + str(predict_score(svm, test, labeltest)) + '\n')
    file.close()

    # Plots

    rdf = dump(svm, "C:/Users/annin/PycharmProjects/Master-Degree-Thesis/Code/results/models/" + name + ".joblib")
    print("[SVM] Model Saved")

    # Plots
    print("Dual Coefficients: ", svm.dual_coef_.shape)
    #print("Support Vectors: ", svm.support_vectors_)
    confusion_matrix(svm, test, labeltest, name = name)
    perm_importance = permutation_importance(svm, train, labeltrain)

    # Making the sum of feature importance being equal to 1.0,
    # so feature importance can be understood as percentage
    perm_importance_normalized = perm_importance.importances_mean / perm_importance.importances_mean.sum()

    # Feature's name (considering your X a DataFrame)
    feature_names = train.columns
    features = np.array(feature_names)

    # Sort to plot in order of importance
    sorted_idx = perm_importance_normalized.argsort()[::-1][:15]

    # Plotting
    plt.figure(figsize=(40, 32))
    plt.title('Feature Importance', fontsize=20)
    plt.barh(features[sorted_idx], perm_importance_normalized[sorted_idx], color='g', align='center')
    plt.xlabel('Relative Importance', fontsize=15)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    for index, value in enumerate(perm_importance_normalized[sorted_idx]):
        plt.text(value, index,
                 str(round(value, 2)), fontsize=15)
    plt.savefig("C:/Users/annin/PycharmProjects/Master-Degree-Thesis/Code/results/plots/" + name + "-FEATUREIMPORTANCE.png")
    # Misclassified Samples
    '''
    y_test = np.asarray(labeltest)
    misclassified = np.where(y_test != svm.predict(test))
    for ind in misclassified[0]:
        print(test.loc[ind])
    '''