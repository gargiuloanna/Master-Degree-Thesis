from sklearn.linear_model import ElasticNetCV
from Code.preprocessing.preprocess import preprocess
import pandas as pd
import numpy as np
import os
if __name__ == '__main__':
    name = "ElasticNet10CV"
    reg = ElasticNetCV(l1_ratio = [0.1, 0.2, 0.3, 0.4, 0.5], eps=0.001, n_alphas=100,
                    fit_intercept=True, precompute='auto', max_iter=5000, tol=0.5, cv=10, copy_X=True,
                        verbose=3, n_jobs=-1, positive=False, random_state=12345, selection='cyclic')

    train, test, labeltrain, labeltest = preprocess()

    reg.fit(train, labeltrain)
    reg.score(test, labeltest)

    file = open("C:/Users/annin/PycharmProjects/Master-Degree-Thesis/Code/preprocessing/elasticnet.txt", "a")
    file.write("Model " + name + '\n')
    file.write("Params: " + str(reg.coef_) + '\n')
    file.write("Test R^2 Predicted:\n")
    file.write(str(reg.score(test, labeltest)) + '\n')

    file.close()
