# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 07:58:53 2023

@author: Siddhant
"""

import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier,VotingClassifier,IsolationForest
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score
import os
import joblib
import csv
from pathlib import Path

path = str(os.getcwd())
model_name = 'VotingClassifier.pkl'


def pre_process(data):
    for i in data.columns:
        if "Num_Col" in i:
            data[i].fillna(data[i].median(),inplace=True)
        elif "Cat_Col" in i:
            data[i].fillna(data[i].mode()[0],inplace=True)
    return data

def outlier_detection(data):
    clf = IsolationForest(random_state=0).fit(data)
    pred = clf.predict(data)
    data["Outlier"] = pred
    return data[data.Outlier==-1].index

def train(path,model_name):
    data = pd.read_csv(path+"//"+"train.csv")
    add_data = pd.read_csv(path+"//"+"add_train.csv")
    
    print("Data Read Completed")
    
    ALL_Data = pd.concat([data,add_data])
    
    ALL_Data = pre_process(ALL_Data)
    
    print("Data Pre Processing Completed")
    
    outlier_index = outlier_detection(ALL_Data.drop('Label',axis=1))
    
    print(f"Outliers of the data are {len(outlier_index)} out of {ALL_Data.shape[0]}")
    
    
    ALL_Data.drop(outlier_index,axis=0,inplace=True)
    
    print("Training Started")
    
    eclf = VotingClassifier(estimators=[('DT', DecisionTreeClassifier(criterion='gini', max_depth=30, min_samples_leaf=4, min_samples_split=10, splitter='best')), 
                                    ('rf', RandomForestClassifier(criterion='gini', max_depth=20, max_features='sqrt', min_samples_leaf=2, min_samples_split=2, n_estimators=100)),
                                    ('knn', KNeighborsClassifier(metric='manhattan', n_neighbors= 15, weights='distance'))],voting='soft')
    eclf.fit(ALL_Data.drop('Label',axis=1),ALL_Data.Label)
    
    print("Training Completed")
    
    accuracy = accuracy_score(ALL_Data.Label, eclf.predict(ALL_Data.drop('Label',axis=1)))
    f1_macro = f1_score(ALL_Data.Label, eclf.predict(ALL_Data.drop('Label',axis=1)), average='macro')
    
    joblib.dump(eclf,path+"\\"+model_name)
    
    print("model saved successfully at path "+path+"\\"+model_name)
    return accuracy,f1_macro

def predict(data):
    predictions = []
    if Path(path+"\\"+model_name).is_file():
        data = pre_process(data)
        model = joblib.load(path+"\\"+model_name)
        #score_d = joblib.load(path+"\\score.pkl")
        predictions = model.predict(data)
        predictions = predictions.tolist()
        s = ""
        for i in predictions:
            s+=str(int(i))+";"
        
        data_list = s.split(';')
        data_transposed = [[item] for item in data_list] 


    else:
        train(path,model_name)
        data = pre_process(data)
        model = joblib.load(model_path)
        #score_d = joblib.load(path+"\\score.pkl")
        predictions = model.predict(data)
        predictions = predictions.tolist()
        s = ""
        for i in predictions:
            s+=str(int(i))+";"
        data_list = s.split(';')
  
        data_transposed = [[item] for item in data_list] 

        with open(path+"\\s4771984.csv", 'w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerows(data_transposed)
        print("predictions saved at path "+path+"\\s4771984.csv")
    return predictions

# Define the outputFormatter function
def outputFormatter(pred, acc, f1, filename):
    # Round acc and f1 to 3rd decimal place
    acc = "{:.3f}".format(acc)
    f1 = "{:.3f}".format(f1)
    if isinstance(pred, pd.DataFrame):
        pred = pred.values.tolist()
    if isinstance(pred, np.ndarray):
        pred = pred.tolist()
    assert isinstance(pred, list), "Unsupported type for pred. It should be either a list, numpy array, or pandas dataframe"
    assert len(pred) == 300, "pred should be a list of 300 elements"
    pred_int = [int(x) for x in pred]
    csv_string = ',\n'.join(map(str, pred_int))
    csv_string += ',\n' + acc + ',' + f1
    filename = filename if filename.endswith('.csv') else filename + '.csv'
    with open(filename, 'w') as f:
        f.write(csv_string)
    return csv_string

def main(): 
    acc,f1 = train(path,model_name)
    testData = pd.read_csv(path+"//"+"test.csv")
    predictions = predict(testData)
    outputFormatter(predictions, acc, f1, "s4771984.csv")
main()