from tabular_data import load_airbnb_numeric
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, GridSearchCV 
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import itertools
import os
import json
from joblib import dump


features, labels = load_airbnb_numeric('Price_Night')

X_train, X_temp, y_train, y_temp = train_test_split(features, labels, test_size=0.2, random_state=42)

X_validation, X_test, y_validation, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)


def baseline_regression_model (model_class, X_test, y_test, X_train, y_train):
    model_class.fit(X_train,y_train)
    y_pred = model_class.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    print("Root Mean Squared Error:", rmse)
    r2 = r2_score(y_test, y_pred)
    print("R^2 Score:", r2)


hyperparameters = {
    'alpha': [0.0001, 0.001, 0.01],
    'penalty': ['l1', 'l2'],
    'learning_rate': ['constant', 'optimal', 'invscaling'],
    'eta0': [0.01, 0.1, 0.5]
}

def custom_tune_regression_model_hyperparameters(model_class, X_test, y_test, X_train, y_train, X_validation, y_validation, hyperparameters:dict):
    best_rmse = float('inf')

    for hyperparameter_combination in itertools.product(*hyperparameters.values()):
        model = model_class.set_params(**dict(zip(hyperparameters.keys(), hyperparameter_combination)))

        model.fit(X_train, y_train)
        y_pred_validation = model.predict(X_validation)
        validation_RMSE = np.sqrt(mean_squared_error(y_validation, y_pred_validation))

        if validation_RMSE < best_rmse:
            best_model = model
            best_hyperparameters = dict(zip(hyperparameters.keys(), hyperparameter_combination))
            best_rmse = validation_RMSE

    y_pred_test = best_model.predict(X_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
           
    performance_metrics = {'Test RSME: ':test_rmse, 'Validation RSME: ':validation_RMSE}

    return best_model, best_hyperparameters, performance_metrics


def tune_regression_model_hyperparameters(model_class, param_grid, X_train, y_train, X_validation, y_validation):
    grid_search = GridSearchCV(model_class, param_grid, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    best_model_params = grid_search.best_params_
    best_model = grid_search.best_estimator_

    y_pred_test = best_model.predict(X_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

    y_pred_validation = best_model.predict(X_validation)
    validation_rmse = np.sqrt(mean_squared_error(y_validation, y_pred_validation))
    
    performance_metrics = {'Test RSME: ':test_rmse, 'Validation RSME: ':validation_rmse}

    return best_model, best_model_params, performance_metrics


def save_model(model, hyperparameters, metrics, folder):
    os.makedirs(folder, exist_ok=True)
    model_path = os.path.join(folder, "model.joblib")
    dump(model, model_path)
    
    hyperparameters_path = os.path.join(folder, "hyperparameters.json")
    with open(hyperparameters_path, "w") as f:
        json.dump(hyperparameters, f)
    
    metrics_path = os.path.join(folder, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f)

def evaluate_all_models():
    #Using SGD Regressor
    model_class = SGDRegressor(random_state=42)

    param_grid = {
        'loss': ['epsilon_insensitive', 'huber'],
        'penalty': ['l1', 'l2'],
        'alpha': [0.0001, 0.001, 0.01],
        'learning_rate': ['constant', 'optimal'],
        'eta0': [0.01, 0.05]
    }

    best_model, best_model_params, performance_metrics = tune_regression_model_hyperparameters(model_class, param_grid, X_train, y_train, X_validation, y_validation)

    save_model(best_model, best_model_params, performance_metrics, folder="models/regression/linear_regression")

    #Using Random Forest Regressor
    model_class = RandomForestRegressor(random_state=42)

    param_grid = {
        'n_estimators': [50, 100, 200],  
        'max_depth': [None, 5, 10],      
        'min_samples_split': [2, 5, 10],  
        'min_samples_leaf': [1, 2, 4]     
    }
    best_model, best_model_params, performance_metrics = tune_regression_model_hyperparameters(model_class, param_grid, X_train, y_train, X_validation, y_validation)

    save_model(best_model, best_model_params, performance_metrics, folder="models/regression/random_forest_regression")

    #Using Gradient Boosting Regressor
    model_class = GradientBoostingRegressor(random_state=42)

    param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 4, 5]
    }
    best_model, best_model_params, performance_metrics = tune_regression_model_hyperparameters(model_class, param_grid, X_train, y_train, X_validation, y_validation)

    save_model(best_model, best_model_params, performance_metrics, folder="models/regression/gradient_boosting_regression")

    #Using Decision Tree Regressor
    model_class = DecisionTreeRegressor(random_state=42)

    param_grid = {
        'splitter': ['best', 'random'],
        'max_depth': [3, 4, 5],
        'min_samples_leaf': [1, 2, 4]
    }
    best_model, best_model_params, performance_metrics = tune_regression_model_hyperparameters(model_class, param_grid, X_train, y_train, X_validation, y_validation)

    save_model(best_model, best_model_params, performance_metrics, folder="models/regression/decision_tree_regression")

def find_best_model():
    #Checking through the metrics of each model, its clear the Gradient Boosting Regression Model is the best one as it has the lowest validation rsme
    model_class = GradientBoostingRegressor(random_state=42)

    param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 4, 5]
    }
    best_model, best_model_params, performance_metrics = tune_regression_model_hyperparameters(model_class, param_grid, X_train, y_train, X_validation, y_validation)
    return best_model, best_model_params, performance_metrics

if __name__ == "__main__":
    evaluate_all_models()
    find_best_model()