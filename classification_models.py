from tabular_data import load_airbnb_category
from sklearn.model_selection import train_test_split, GridSearchCV 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from modelling import save_model
import os
import json
import numpy as np


features, labels = load_airbnb_category('Category')

X_train, X_temp, y_train, y_temp = train_test_split(features, labels, test_size=0.2, random_state=42)

X_validation, X_test, y_validation, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)



def baseline_classification_model (model_class, X_validation, y_validation, X_train, y_train):
    model_class.fit(X_train, y_train)

    y_pred = model_class.predict(X_validation)

    accuracy = accuracy_score(y_validation, y_pred)
    f1 = f1_score(y_validation, y_pred, average="weighted")
    precision = precision_score(y_validation, y_pred, average='weighted')
    recall = recall_score(y_validation, y_pred, average='weighted')
    print("Validation Accuracy:", accuracy)
    print("Validation f1:", f1)
    print("Validation precision:", precision)
    print("Validation recall:", recall)


def tune_classification_model_hyperparameters(model_class, param_grid, X_train, y_train, X_validation, y_validation):
    grid_search = GridSearchCV(model_class, param_grid, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    best_model_params = grid_search.best_params_
    best_model = grid_search.best_estimator_

    y_pred = best_model.predict(X_validation)
    accuracy = accuracy_score(y_validation, y_pred)
    
    performance_metrics = {'validation_accuracy': accuracy}

    return best_model, best_model_params, performance_metrics

def evaluate_all_models():
    #Using Logistic Regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_validation_scaled = scaler.transform(X_validation)
    model_class = LogisticRegression(random_state=42)

    param_grid = {
        'penalty': ['l1', 'l2'], 
        'C': [0.001, 0.01, 0.1, 1, 10, 100], 
        'solver': ['liblinear', 'saga'],
        'max_iter': [200, 300, 400],
        'tol': [1e-4, 1e-3, 1e-2]  
    }

    best_model, best_model_params, performance_metrics = tune_classification_model_hyperparameters(model_class, param_grid, X_train_scaled, y_train, X_validation_scaled, y_validation)

    save_model(best_model, best_model_params, performance_metrics, folder="models/classification/logistic_regression")


    #Using Random Forest Classifier
    model_class = RandomForestClassifier(random_state=42)

    param_grid = {
        'n_estimators': [50, 100, 200],  
        'max_depth': [None, 5, 10],      
        'min_samples_split': [2, 5, 10],  
        'min_samples_leaf': [1, 2, 4]     
    }

    best_model, best_model_params, performance_metrics = tune_classification_model_hyperparameters(model_class, param_grid, X_train, y_train, X_validation, y_validation)

    save_model(best_model, best_model_params, performance_metrics, folder="models/classification/random_forest_classifier")

    #Using Decision Tree Classifier

    model_class = DecisionTreeClassifier(random_state=42)

    param_grid = {
        'splitter': ['best', 'random'],
        'max_depth': [3, 4, 5],
        'min_samples_leaf': [1, 2, 4]
    }

    best_model, best_model_params, performance_metrics = tune_classification_model_hyperparameters(model_class, param_grid, X_train, y_train, X_validation, y_validation)

    save_model(best_model, best_model_params, performance_metrics, folder="models/classification/decision_tree_classifier")

    #Using Gradient Boosting Classifier
    model_class = GradientBoostingClassifier(random_state=42)

    param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.1, 0.05, 0.01],  
    'max_depth': [None, 5, 10],      
    'min_samples_split': [2, 5, 10],  
    'min_samples_leaf': [1, 2, 4]
    }

    best_model, best_model_params, performance_metrics = tune_classification_model_hyperparameters(model_class, param_grid, X_train, y_train, X_validation, y_validation)

    save_model(best_model, best_model_params, performance_metrics, folder="models/classification/gradient_boosting_classifier")

root_dir = '/path/to/your/root/directory'
target_filename = 'your_target_filename.txt'

def find_best_model():
    root_dir = r'C:\Users\harde\Documents\AiCore\Airbnb\models\classification'

    best_accuracy = -1
    best_document = None

    for folder_name in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder_name)
        if os.path.isdir(folder_path):
            metrics_file_path = os.path.join(folder_path, 'metrics.json')
            with open(metrics_file_path, 'r') as metrics_file:
                metrics_data = json.load(metrics_file)
                validation_accuracy = metrics_data.get('validation_accuracy', -1)
                if validation_accuracy > best_accuracy:
                    best_accuracy = validation_accuracy
                    best_document = metrics_file_path

    return best_document, best_accuracy


if __name__ == "__main__":
    evaluate_all_models()
    find_best_model()
