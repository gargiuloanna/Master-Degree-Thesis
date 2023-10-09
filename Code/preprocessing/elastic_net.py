from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import train_test_split
from Code.preprocessing.scale import scale, variance
import pandas as pd
import numpy as np
import os
if __name__ == '__main__':
    reg = ElasticNetCV(l1_ratio = [0.1, 0.2, 0.3, 0.4, 0.5], eps=0.001, n_alphas=100,
                    fit_intercept=True, precompute='auto', max_iter=5000, tol=0.5, cv=10, copy_X=True,
                        verbose=3, n_jobs=-1, positive=False, random_state=12345, selection='cyclic')

    for i in ['Dataset -features_changed.xlsx','Dataset -features_changed - lr.xlsx','Dataset -features_changed - mean.xlsx' ,
              'Dataset-normalized lengths.xlsx', 'Dataset-normalized lengths - lr.xlsx','Dataset-normalized lengths - mean.xlsx',
              'Dataset-only normalized lengths.xlsx','Dataset-only normalized lengths -lr.xlsx','Dataset-only normalized lengths - mean.xlsx']:
        name = i.split(".")[0]
        print(name)
        df = pd.read_excel('C:/Users/annin/PycharmProjects/Master-Degree-Thesis/Code/Data/' + i)


        columns = pd.DataFrame()
        columns['Patient'] = df['Patient']
        columns['Exercise'] = df['Exercise']
        df.drop(['Unnamed: 0', 'Patient', 'Exercise'], axis=1, inplace=True)

        data = df.iloc[:, :-1]
        labels = df.iloc[:, -1]

        for j in np.arange(0.0, 1.0, 0.01):
            varianced = variance(data, threshold = j)
            varianced['Patient'] = columns['Patient']
            varianced['Exercise'] = columns['Exercise']
            varianced['Labels'] = labels
            if not os.path.exists("C:/Users/annin/PycharmProjects/Master-Degree-Thesis/Code/Data/data_varianced/" + name):
                os.mkdir("C:/Users/annin/PycharmProjects/Master-Degree-Thesis/Code/Data/data_varianced/" + name)
            varianced.to_excel("C:/Users/annin/PycharmProjects/Master-Degree-Thesis/Code/Data/data_varianced/" + name + '/' + name + '_' + str(j) + '.xlsx')
        '''
        train, test, labeltrain, labeltest =train_test_split(data, labels, test_size=0.30, random_state=12345, stratify=labels)
        train, test = scale(train, test)

        reg.fit(train, labeltrain)
        reg.score(test, labeltest)

        file = open("C:/Users/annin/PycharmProjects/Master-Degree-Thesis/Code/preprocessing/elasticnet.txt", "a")
        file.write("Model " + name + '\n')
        file.write("Params: " + str(reg.coef_) + '\n')
        file.write("Test R^2 Predicted:\n")
        file.write(str(reg.score(test, labeltest)) + '\n')
        file.write(str(labeltest) + '\n')

        file.close()
        '''
