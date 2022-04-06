import pandas as pd

import numpy as np
import re
import io
import copy
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

import pickle

#import data
final_df = pd.read_csv('/content/drive/MyDrive/IS453 FA Project/merged_processed.csv')

#remove features of data
final_df.drop([ 'ID' ,'CODE_GENDER1'], axis=1 , inplace= True)

# final_df
x = final_df[final_df.columns.difference(['Status'])]
y = final_df['Status']

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20, stratify=y, random_state=424)

# Oversampling
oversample = SMOTE(sampling_strategy= 'minority', random_state = 0)
X_train, y_train = oversample.fit_resample(X_train, y_train)

#Training RF classifier
clf =  RandomForestClassifier()
clf.fit(X_train, y_train)


#Saving the model
pickle.dump(clf, open('model.pkl','wb'))

#Loading model to compare the results
# model = pickle.load(open('model.pkl','rb'))
# print(model.predict([[10,2,9,89,2,329,288.28,40,1]]))