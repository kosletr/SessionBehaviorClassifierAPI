# %%
"""

This script is responsible for exporting the most suitable anomaly detection model 
given a particular dataset. Add your settings to the `Parameters` section of the code.

"""

# %%

import os
import numpy as np
import pandas as pd
from sklearn import cluster, ensemble, neighbors, svm, covariance
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, silhouette_score, calinski_harabasz_score
from matplotlib import pyplot as plt
import itertools
import joblib

# %%


"""
Parameters - Section
"""


dir_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'datasets')

class_name = "behavior"
class_dict = {"Abnormal": -1, "Normal": 1}
cols_to_remove = ['_id', 'sessionID', 'active',
                  'clientIP', 'endTimestamp', 'reqBodiesData']

metric = calinski_harabasz_score
export_model_filename = "model.joblib"

# %%


"""
Function Definitions - Section
"""


# Preprocessing Function
def preprocess_and_split(df, cols_to_remove, class_name, class_dict, test_size=0.2, random_state=42, split=True):

    # Remove unecessary columns
    df = df.drop(columns=cols_to_remove)

    # Replace string values by category numbers
    # df.replace({class_name: class_dict}, inplace=True)

    # Split the dataframe to data and labels
    X, y = df.drop(columns=[class_name]).to_numpy(), df[class_name].to_numpy()
    

    if split == True:

        # Split the data and labels to training and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y)

        # Replace Attack Labels by the Abnormal Value
        y_train[y_train != 'Normal'] = class_dict['Abnormal']
        y_test[y_test != 'Normal'] = class_dict['Abnormal']

        # Replace Normal labels by the Normal Value
        y_train[y_train == 'Normal'] = class_dict['Normal']
        y_test[y_test == 'Normal'] = class_dict['Normal']

        return X_train, X_test, np.array(y_train, dtype="int64"), np.array(y_test, dtype="int64")

    else:

        # Replace Attack Labels by the Abnormal Value
        y[y != 'Normal'] = class_dict['Abnormal'] 
        # Replace Normal labels by the Normal Value
        y[y == 'Normal'] = class_dict['Normal']
        
        return X, np.array(y, dtype="int64")


# Classifiers with params to be tested
def choose_models():

    isolFor = {'name': 'Isolation Forest',
               'class': ensemble.IsolationForest(),
               'parameters': {
                   'n_estimators': [5, 10, 20, 50, 100, 150, 200]
               }
               }

    locOutFac = {'name': 'Local Outlier Factor',
                 'class': neighbors.LocalOutlierFactor(novelty=True),
                 'parameters': {
                     'n_neighbors': range(5, 50, 5)
                 }
                 }
    # ocSVM = {'name': 'One Class SVM',
    #          'class': svm.OneClassSVM(),
    #          'parameters': {
    #              'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    #              'nu': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]
    #          }
    #          }

    elEnv = {'name': 'Elliptic Envelope',
             'class': covariance.EllipticEnvelope(),
             'parameters': {
                 'contamination': np.linspace(0.05, 0.45, 9)
             }
             }

    return [isolFor, locOutFac, elEnv]

# Recursively generate grid params


def make_generator(params):
    # If params is empty, return empty dict
    if not params:
        yield dict()
    else:
        # Get the name of the first param
        key_to_iterate = list(params.keys())[0]

        # Dictionary with the rest of the keys
        next_round_parameters = {p: params[p]
                                 for p in params if p != key_to_iterate}

        # Get the value of the first param
        for val in params[key_to_iterate]:
            # Recursion starts here
            for pars in make_generator(next_round_parameters):
                temp_res = pars
                temp_res[key_to_iterate] = val
                yield temp_res

# Custom grid search function
def grid_search(estimator, X, param_grid, scoring):

    results = []

    # Loop through the different param sets
    for params in make_generator(param_grid):

        # Load and merge predefined parameters
        # with the current grid params
        final_params = estimator.get_params()
        final_params.update(params.copy())
        # Update model's parameters
        estimator.set_params(**final_params)
        # Fit the model
        model = estimator.fit(X)
        y_pred = model.predict(X)
        # Evaluate the model
        num_of_labels = len(np.unique(y_pred))
        if num_of_labels > 1 and num_of_labels < X.shape[0]:
            score = scoring(X, y_pred)
        else:
            score = -1

        results.append({
            "estimator": model,
            "params": params,
            "score": score
        })


    df = pd.DataFrame(results)
    del df['estimator']
    print(df.to_latex(index=False))

    # Find the element with the best score
    best_score = max([x['score'] for x in results])
    # Find the index of the element with the best score
    best_idx = next((idx for (idx, d) in enumerate(
        results) if d["score"] == best_score), None)

    return results[best_idx]


# Train all models and sort them by metric
def train_models(models, X, metric):
    best_model_list = []

    # Grid Search with Stratified K-Fold Cross-Validation
    for model in models:
        best_model = grid_search(model['class'], X, model['parameters'], metric)

        # For each model append a tuple to the final list
        best_model_list.append(
            # The tuple contains the following params
            (
                # Classifier's name
                model['name'],
                # Best Estimator of the k-fold cross validation
                best_model["estimator"],
                # Best Parameters of the estimator
                best_model["params"],
                # Metric of the best estimator
                best_model["score"]
            )
        )

    # Sort the best_model_list (desc order) based on best_score
    return sorted(best_model_list, key=lambda x: x[3], reverse=True)


# Plot Confusion Matrix Function
def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          normalize=False):

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.show()

# %%


"""
Preprocessing - Section
"""


# Import the dataset
df = pd.concat([pd.read_csv(os.path.join(dir_path,f), index_col=False) for f in os.listdir(dir_path) if f.endswith('.csv')], ignore_index=True)

# Preprocessing and split to train, test sets
X, y = preprocess_and_split(
    df, cols_to_remove, class_name, class_dict, test_size=0.2, random_state=42, split=False)

# %%


"""
Training - Section
"""

# Train the models
final_model = train_models(choose_models(), X, metric=metric)
best_model_name, best_model, best_model_params, best_score = final_model[0][
    0], final_model[0][1], final_model[0][2], final_model[0][3]

# %%


"""
Evaluation - Section
"""

# Calculate Predictions
y_pred = best_model.predict(X)

# Get class names and labels
class_names = list(class_dict.keys())
labels = list(class_dict.values())

# Confusion Matrix
cm = confusion_matrix(y, y_pred, labels=labels)

# Classification Report
cr = classification_report(
    y, y_pred, labels=labels, target_names=class_names)

# %%


"""
Results - Section
"""

# Print Results
print('Best Model: ', best_model_name)
print('Best Parameter(s): ', best_model_params)
print('Best Score: ', best_score)

# Train Set Results
print('\nClassification Report: \n')
print(cr)
print('Confusion Matrix: ')
print(cm)

# Plot Confusion Matrix
plot_confusion_matrix(cm, target_names=class_names)
plt.show()

# Export the final model
# joblib.dump(best_model, export_model_filename)

# %%
