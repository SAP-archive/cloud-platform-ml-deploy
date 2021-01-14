'''
Created on Aug 30, 2020

@author: I054127
'''

# if __name__ == '__main__':
#     pass

import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pickle
from sklearn.externals import joblib

# save filepath to variable for easier access
medical_dataset_file_path = 'C:\\Python\\data\\heart_failure_dataset.csv'
# read the data and store data in DataFrame titled medical_data
medical_data = pd.read_csv(medical_dataset_file_path) 
medical_data_features = ['age', 'anaemia', 'creatinine_phosphokinase', 'diabetes',
       'ejection_fraction', 'high_blood_pressure', 'platelets',
       'serum_creatinine', 'serum_sodium', 'sex', 'smoking']

y = medical_data.DEATH_EVENT
X = medical_data[medical_data_features]

medical_model = DecisionTreeRegressor(random_state=1)
medical_model.fit(X, y)

print("Making predictions for the following 5 houses:")
print(X.head())
print("The predictions are")
print(medical_model.predict(X.head()))

#Simple logistic Regression Algo

# medical_model_lr = LogisticRegression()
# medical_model_lr.fit(X,y)
# 
# print("Making predictions for the following 5 houses:")
# print(X.head())
# print("The predictions are")
# print(medical_model_lr.predict(X.head()))

#Using Test/Train/Split Technique


X_train, X_test, y_train, y_test  = train_test_split(medical_data.drop('DEATH_EVENT', axis=1), medical_data['DEATH_EVENT'],test_size=0.30,random_state = 101)
model_log_reg = LogisticRegression()
model_log_reg.fit(X_train,y_train)

predictions = model_log_reg.predict(X_test)
print(classification_report(y_test,predictions))

#***********************************Using Pickle *********************************

# # Save your Model
# pkl_filename = "pickle_model.pkl"
# with open(pkl_filename, 'wb') as file:
#     pickle.dump(model_log_reg, file)
#     
# # Load from file
# with open(pkl_filename, 'rb') as file:
#     pickle_model = pickle.load(file)    
# 
# # Calculate the accuracy score and predict target values
# score = pickle_model.score(X_test, y_test)
# print("Test score: {0:.2f} %".format(100 * score))
# Ypredict = pickle_model.predict(X_test)    


#*********************************Using JobLib Module *******************************

# Save to file in the current working directory
joblib_file = "joblib_model.pkl"
joblib.dump(model_log_reg, joblib_file)

# Load from file
joblib_model = joblib.load(joblib_file)

# Calculate the accuracy and predictions
score = joblib_model.score(X_test, y_test)
print("Test score: {0:.2f} %".format(100 * score))


# print a summary of the data in medical data 
# print(medical_data.describe())
# print(medical_data.columns)