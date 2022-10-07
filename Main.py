import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm, tree
# import xgboost
import mlflow
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from mlflow.models.signature import infer_signature



def load_data(path):
    data = pd.read_csv(path)
    return data


def data_cleaning(data):
    print("NA values in dataset\n")
    print(data.isna().sum())
    data = data.dropna()
    return data


def split_into_train_test(data):
    x = data.drop('price_range', axis=1)
    y = data['price_range']
    X = (x - np.min(x)) / (np.max(x) - np.min(x))
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    return x_train, x_test, y_train, y_test

def model_LogisticRegression():
    model = LogisticRegression()
    return model

def model_KNeighborsClassifier():
    model = KNeighborsClassifier(n_neighbors=5, leaf_size=10, n_jobs=1, algorithm='auto')
    return model

def model_GradientBoostingClassifier():
    model=GradientBoostingClassifier()
    return model

def model_RandomForestClassifier():
    model=RandomForestClassifier(n_estimators=20, random_state=42)
    return model


def training_model(x_train, y_train, model_c):
    model_c.fit(x_train, y_train)
    return model_c


def predict_test_data(model, x_test):
    y_pred = model.predict(x_test)
    return y_pred


def predict_prob_test_data(model, x_test):
    y_pred = model.predict_proba(x_test)
    return y_pred


def get_metrics(y_true, y_pred, y_pred_prob):
    acc = accuracy_score(y_true, y_pred)
    pre = precision_score(y_true, y_pred, average='weighted')
    rec = recall_score(y_true, y_pred, average='weighted')

    return {'accuracy': round(acc, 4), 'precision': round(pre, 4), 'recall': round(rec, 4)}


def create_confusion_matrix(clf, x_test, y_test):
    from sklearn.metrics import plot_confusion_matrix
    plot_confusion_matrix(clf, x_test, y_test)
    plt.savefig('confusion_matrix.png')

def create_signature():
    from mlflow.models.signature import ModelSignature
    from mlflow.types.schema import Schema, ColSpec
    input_schema = Schema([
    ColSpec("double", "battery_power"),
    ColSpec("double", "blue"),
    ColSpec("double", "clock_speed"),
    ColSpec("double", "dual_sim"),
    ColSpec("double", "fc"),
    ColSpec("double", "four_g"),
    ColSpec("double", "int_memory"),
    ColSpec("double", "m_dep"),
    ColSpec("double", "mobile_wt"),
    ColSpec("double", "n_cores"),
    ColSpec("double", "pc"),
    ColSpec("double", "px_height"),
    ColSpec("double", "px_width"),
    ColSpec("double", "ram"),
    ColSpec("double", "sc_h"),
    ColSpec("double", "sc_w"),
    ColSpec("double", "talk_time"),
    ColSpec("double", "three_g"),
    ColSpec("double", "touch_screen"),
    ColSpec("double", "wifi"),
    ])

    output_schema = Schema([ColSpec("long","price_range")])

    return ModelSignature(inputs = input_schema, outputs = output_schema)


##def param(model_name):
    #return {'n_estimators': 5, 'max_depth': 3, 'learning_rate': 0.01}

def create_experiment(experiment_name, run_name, run_metrics, model, model_name, signature, confusion_matrix_path=None, run_p=None):
   mlflow.set_tracking_uri("http://ilcepoc2353:1235")
   mlflow.set_experiment(experiment_name)
   with mlflow.start_run(run_name='Simran'):
        for param in run_p:
            mlflow.log_param(param, run_p[param])
        for metric in run_metrics:
            mlflow.log_metric(metric, run_metrics[metric])


        if not confusion_matrix_path == None:
            mlflow.log_artifact(confusion_matrix_path, 'confusion_matrix')
        mlflow.set_tag("Model_Name", model_name)
        mlflow.log_artifact("train.csv")
        mlflow.sklearn.log_model(model, "model",signature= signature)
        print('Run - %s is logged to experiment - %s'%(run_name, experiment_name))


if __name__ == "__main__":
    data = load_data('train.csv')
    X_train, X_test, Y_train, Y_test = split_into_train_test(data)

    experiment_name = "MLHOLICS"
    # run_name = "Random Forest"

    model_array = np.array(["KNeighborsClassifier", "GradientBoostingClassifier", "RandomForestClassifier"])

    for model_name in model_array:
        if model_name == "LogisticRegression":
            model = model_LogisticRegression()
        elif model_name == "KNeighborsClassifier":
            model = model_KNeighborsClassifier()
        elif model_name == "GradientBoostingClassifier":
            model = model_GradientBoostingClassifier()
        elif model_name == "RandomForestClassifier":
            model = model_RandomForestClassifier()
        
        params = {'RandomForestClassifier':{'n_estimators':10, 'random_state':42 , 'max_depth':1},
              'KNeighborsClassifier':{'leaf_size':30, 'n_jobs':1, 'n_neighbors':5},
              'GradientBoostingClassifier':{'n_estimators':5, 'max_depth':3,'learning_rate':0.01},
              'LogisticRegression':{}
              }

        model = training_model(X_train, Y_train, model)
        Y_pred = predict_test_data(model, X_test)
        Y_pred_proba = predict_prob_test_data(model, X_test)
        eval_metrics = get_metrics(Y_test, Y_pred, Y_pred_proba)
        create_confusion_matrix(model, X_test, Y_test)
        run_metrics = get_metrics(Y_test, Y_pred, Y_pred_proba)
        print(run_metrics)
        signature = create_signature()
        # infer_signature(X_train, model.predict(X_test))
        run_name="Simran"
        create_experiment(experiment_name, run_name, run_metrics, model, model_name, signature,'confusion_matrix.png', params[model_name])
   
