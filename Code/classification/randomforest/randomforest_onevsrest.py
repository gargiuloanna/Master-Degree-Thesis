from sklearn.multiclass import OneVsRestClassifier
from Code.preprocessing.preprocess import preprocess
from sklearn.ensemble import RandomForestClassifier
from  sklearn.metrics import balanced_accuracy_score
from Code.plotting.plots import feature_importance

if __name__ == '__main__':

    train, test, labeltrain, labeltest = preprocess()

    clf = RandomForestClassifier(criterion='entropy', max_depth=8, max_features='sqrt', min_samples_split=10,
                                 min_samples_leaf=1, n_estimators=100, random_state=8779374, oob_score=balanced_accuracy_score,
                                 class_weight='balanced')

    ovr = OneVsRestClassifier(clf, verbose = 51, n_jobs=-1)

    print("Fitting\n", ovr.fit(train,labeltrain))
    print("Test Accuracy ", ovr.score(test, labeltest))
    print("Classes ", ovr.classes_)
    #Plots
    feature_importance(ovr.estimators_[0], train.columns,name = "randomforest/ONEVSREST_0")
    feature_importance(ovr.estimators_[1], train.columns,name = "randomforest/ONEVSREST_1")
    feature_importance(ovr.estimators_[2], train.columns,name = "randomforest/ONEVSREST_2")
