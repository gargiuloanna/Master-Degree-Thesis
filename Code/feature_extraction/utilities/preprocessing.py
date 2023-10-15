import pandas as pd
import numpy as np
from sklearn.preprocessing import MaxAbsScaler
from sklearn.impute import KNNImputer


'''
Checks for any missing value in the dataframe, and creaters a new column
The column = True if there are missing values in the current row, else False
'''
def check_nan(data):
  data['nan'] = data.isnull().values.any(axis = 1)
  return data

'''
Cleans the Data by removing the initial and final samples if there are values missing.
'''
def clean_data(dn_complete):

  dn_complete = check_nan(dn_complete)

  #Remove initial NaN
  previous = True
  for elem in dn_complete.index:
    if dn_complete['nan'][elem] == True and previous:
      dn_complete.drop(elem, axis = 0, inplace = True)
      previous = True
    else:
      previous = False

  # Resets the Index Column so that the values are continous
  dn_complete = dn_complete.reset_index(drop = True)

  #Remove final NaN
  last = True
  len = dn_complete.shape[0]
  for elem in range(len-1, 0, -1):
    if dn_complete['nan'][elem] == True and last:
      dn_complete.drop(elem, axis = 0, inplace = True)
      last = True
    else:
      last = False

  return dn_complete

'''
Fills in missing Data using a KNN imputer.
'''
def impute(dn_complete):
  imputer = KNNImputer(missing_values = np.nan, weights = 'distance', keep_empty_features=True)

  column  = dn_complete.columns[:51]
  label_columns = dn_complete.iloc[:, 51:54]

  dn_complete = imputer.fit_transform(dn_complete.iloc[:, 0:51])

  dn_complete = pd.DataFrame(dn_complete, columns = column)
  dn_complete = pd.concat([dn_complete, label_columns],axis = 1)
  return dn_complete

'''
Creates new columns containing the average values of subsets of pressure sensors.
'''
def compute_average(dn_complete):
  dn_complete.insert(51, 'avg left pressure', dn_complete.iloc[:, 1:17].mean(axis=1), True)
  dn_complete.insert(52, 'avg right pressure', dn_complete.iloc[:, 26:42].mean(axis=1), True)
  #left heel and toe pressure averages
  dn_complete.insert(53, 'avg left heel pressure', dn_complete.iloc[:, 1:7].mean(axis=1), True)
  dn_complete.insert(54, 'avg left toe pressure', dn_complete.iloc[:, 9:17].mean(axis=1), True)

  #right heel and toe pressure averages
  dn_complete.insert(55, 'avg right heel pressure', dn_complete.iloc[:, 26:32].mean(axis=1), True)
  dn_complete.insert(56, 'avg right toe pressure', dn_complete.iloc[:, 34:42].mean(axis=1), True)

  return dn_complete

'''
Scales the Data in the range [-1, 1] with a MaxAbsScaler
'''
def scale(dn_complete):

  columns  = dn_complete.columns[1:57]
  label_columns = dn_complete.iloc[:, 57:60]

  #Save and drop time column to avoid scaling the time
  cl = pd.DataFrame(dn_complete['# time'], columns = ['# time'])
  dn_complete.drop('# time', axis = 1, inplace = True)

  #MaxAbsScaler to scale in the range [-1,1]
  transformer = MaxAbsScaler(copy = False)
  dn_complete = transformer.fit_transform(dn_complete.iloc[:, 0:56])
  dn_complete = pd.DataFrame(dn_complete, columns = columns)

  #read the time column
  dn_complete['# time'] = cl['# time']
  dn_complete = pd.concat([dn_complete, label_columns], axis = 1)
  #round to result to 2 decimal numbers
  dn_complete = dn_complete.round(2)

  return dn_complete