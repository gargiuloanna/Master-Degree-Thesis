from sklearn.svm import SVC
from Code.evaluation.predict import predict_score
from Code.plotting.plots import confusion_matrix, svm_importances
from Code.preprocessing.preprocess import preprocess
from  sklearn.metrics import classification_report
import numpy as np
from joblib import dump




if __name__ == '__main__':

    train, test, labeltrain, labeltest = preprocess()

    seed = 5436398
    name = "svm/SVMWholePreprocess "+str(seed)

    svm = SVC(random_state=0, class_weight='balanced', kernel='poly',
              C=0.1, degree=1, coef0=1, decision_function_shape='ovo')

    svm.fit(train, labeltrain)

    # Performance
    print("Train Accuracy: ", predict_score(svm, train, labeltrain))
    print("Test Accuracy: ", predict_score(svm, test, labeltest))

    #Save Model
    rdf = dump(svm, "C:/Users/annin/PycharmProjects/Master-Degree-Thesis/Code/results/models/svm/" + name + ".joblib")
    print("[RANDOM FOREST] Model Saved")

    print(classification_report(labeltest, svm.predict(test),  target_names=['Elderly', 'Parkinson', 'Adults'],  digits=3, output_dict=False, zero_division='warn'))

    # Write to File
    file = open("C:/Users/annin/PycharmProjects/Master-Degree-Thesis/Code/results/logs/svm/svm.txt", "a")
    file.write("\nModel " + name + '\n')
    file.write("Train Accuracy: " + str(predict_score(svm, train, labeltrain)) + '\n')
    file.write("Test Accuracy: " + str(predict_score(svm, test, labeltest)) + '\n')
    file.close()

    # Plots
    svm_importances(svm, train)
    confusion_matrix(svm, test, labeltest, name=name)

    #Misclassified Samples
    y_test = np.asarray(labeltest)
    misclassified = np.where(y_test != svm.predict(test))
    for ind in misclassified[0]:
        print(test.loc[ind])