import pandas as pd 
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm, tree
import xgboost
import mlflow
from sklearn.model_selection import train_test_split
if __name__ == "__main__":
    
    data = pd.read_csv('train.csv')
    x = data.drop(['price_range'],axis=1)
    y = data['price_range']
    X = (x- np.min(x)) / (np.max(x) - np.min(x))
    x_train,x_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=42)
    print(len(x_train))
    print(len(x_test))
    print(len(y_train))
    print(len(y_test))
    model1 = xgboost.XGBClassifier()
    model1.fit(x_train, y_train)
    y_pred= model1.predict(x_test)
    # print(y_pred)
    acc = accuracy_score(y_test, y_pred)
    print("Accuracy is %s"%(acc))
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix is %s"%(cm))

    model2 = RandomForestClassifier()
    model2.fit(x_train, y_train)
    y_pred= model2.predict(x_test)
    # print(y_pred)
    acc = accuracy_score(y_test, y_pred)
    print("Accuracy is %s"%(acc))
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix is %s"%(cm))
    
   
    mlflow.set_tracking_uri("http://ilcepoc2353:5000")
    mlflow.set_experiment("Test")
    
    with mlflow.start_run():
        mlflow.sklearn.log_model(model1,"model1")
