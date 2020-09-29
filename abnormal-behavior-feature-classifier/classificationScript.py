# %%
"""
This script is responsible for exporting the most suitable binary classifier given a particular 
dataset. Add your settings to the `Parameters` section of the code.

Important: To run the following script scikit-learn 0.21.2 must be installed!
"""

# %%

import os
import numpy as np
import pandas as pd
import operator
from sklearn import neighbors, svm, linear_model, tree, ensemble, metrics
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, recall_score, make_scorer
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers import Dense, Activation, Dropout
from keras.models import Sequential
from matplotlib import pyplot as plt
import itertools
import joblib

# %%


"""
Parameters - Section
"""


dir_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'datasets')

class_name = "behavior"
class_dict = {"Abnormal": 0, "Normal": 1}
cols_to_remove = ['_id', 'sessionID', 'active',
                  'clientIP', 'endTimestamp', 'reqBodiesData']

# Use two metrics to solve ties between models
metrics = {
    'sc1': make_scorer(recall_score, pos_label=class_dict["Abnormal"]), 
    'sc2':'f1_macro'
    }

export_model_filename = "model.joblib"

# %%


"""
Function Definitions - Section
"""


# Preprocessing Function
def preprocess_and_split(df, cols_to_remove, class_name, class_dict, test_size=0.2, random_state=42, split=True):

    # Remove unecessary columns
    df = df.drop(columns=cols_to_remove)

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


# ANN Model Struture
def create_model(layers, activation, dropout_rate, optimizer):

    # Define model
    model = Sequential()
    # Input Layer
    model.add(Dense(layers[0], input_dim = X_train.shape[1], activation = activation[0]))
    # Hidden Layers
    for l, _ in enumerate(layers):
        model.add(Dense(layers[l], activation = activation[l+1]))
        if dropout_rate[l] < 1:
            model.add(Dropout(dropout_rate[l]))
    # Output Layer
    model.add(Dense(1, activation = 'sigmoid'))

    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy', metrics=['accuracy'])
    return model


# Create Architecture Object to add in CV
def create_nn_arch(idx, params):
    return {'name': ('Multi-Layer Perceptron - Neural Network ' + str(idx+1)),
            'class': KerasClassifier(build_fn=create_model, verbose=0),
            'parameters': params
            }


# Classifiers with params to be tested
def choose_models():

    svc_linear = {'name': 'Support Vector Classifier with Linear Kernel',
                  'class': svm.LinearSVC(),
                  'parameters': {
                      'C': [0.001, 0.01, 0.1, 1, 10, 100]
                  }
                  }

    svc_radial = {'name': 'Support Vector Classifier with Radial Kernel',
                  'class': svm.SVC(kernel='rbf'),
                  'parameters': {
                      'C': [0.001, 0.01, 0.1, 1, 10, 100],
                      'gamma': [0.001, 0.01, 0.1, 1, 10, 100]
                  }
                  }

    knn = {'name': 'K Nearest Neighbors Classifier',
           'class': neighbors.KNeighborsClassifier(),
           'parameters': {
               'n_neighbors': range(1, 12)
           }
           }

    dec_tree = {'name': "Decision Tree Classifier",
                'class': tree.DecisionTreeClassifier(),
                'parameters': {
                    'max_depth': range(3, 15)
                }
                }

    rand_for = {'name': "Random Forest Classifier",
                'class': ensemble.RandomForestClassifier(),
                'parameters': {
                    'n_estimators': [10, 20, 50, 100, 200]
                }
                }

    stoch_grad_desc_class = {'name': "Stochastic Gradient Descent Classifier",
                             'class': linear_model.SGDClassifier(),
                             'parameters': {
                                 'max_iter': [100, 1000],
                                 'alpha': [0.0001, 0.001, 0.01, 0.1]
                             }
                             }

    log_reg_lasso = {'name': "Logistic Regression with LASSO",
                     'class': linear_model.LogisticRegression(penalty='l1'),
                     'parameters': {
                         'C': [0.001, 0.01, 0.1, 1, 10, 100]
                     }
                     }

    nn_arch = [
        {
            'layers': [[12, 15]],
            'activation':[['relu', 'relu', 'relu']],
            'dropout_rate':[[0.8, 0.5]],
            'batch_size':[16],
            'epochs':[100],
            'optimizer':['adam']
        }, {
            'layers': [[20, 35]],
            'activation':[['relu', 'relu', 'relu']],
            'dropout_rate':[[0.5, 0.5]],
            'batch_size':[16],
            'epochs':[100],
            'optimizer':['adam']
        }, {
            'layers': [[15, 18, 8]],
            'activation':[['relu', 'relu', 'relu', 'relu']],
            'dropout_rate':[[0.8, 0.5, 0.2]],
            'batch_size':[16],
            'epochs':[100],
            'optimizer':['adam']
        },
    ]

    # Call create_nn_arch function to create separate neural_net instances
    mlp_nn = [create_nn_arch(idx, arch) for idx, arch in enumerate(nn_arch)]
    other_classifiers = [svc_linear, svc_radial, knn, dec_tree, rand_for,
                         stoch_grad_desc_class, log_reg_lasso]

    return mlp_nn + other_classifiers


# Returns latex code of the results
def returnLatexDf(best_fit_model, metrics, k):
    df_model = pd.DataFrame(best_fit_model.cv_results_)

    # Remove cols
    time_cols = ['mean_fit_time', 'std_fit_time',
                 'mean_score_time', 'std_score_time']
    param_cols = [x for x in df_model.columns if 'param_' in x]
    df_model.drop(time_cols + param_cols, axis=1, inplace=True)

    # Split df to separate dfs based on metric
    split_cols = ["split"+str(x)+"_test" for x in range(k)] + ["mean_test"]
    df_latex = []

    for m in list(metrics.keys()):
        df_col_names = ["params"] + [x + '_' + m for x in split_cols]
        df_part = df_model[df_model.columns & df_col_names]
        df_latex.append(df_part.to_latex(index=False))

    return df_latex


# Train all models and sort them by metric
def train_models(models, X, y, metrics, refit, k):
    best_model_list = []
    latex_list = []

    # Grid Search with Stratified K-Fold Cross-Validation
    for model in models:
        best_model = GridSearchCV(
            estimator=model['class'], param_grid=model['parameters'], cv=k, scoring=metrics, refit=refit, n_jobs=-1)

        # Fit the best model
        best_fit_model = best_model.fit(X, y)

        # For each model append a tuple to the final list
        best_model_list.append(
            # The tuple contains the following params
            (
                # Classifier's name
                model['name'],
                # Best Estimator of the k-fold cross validation
                best_fit_model.best_estimator_,
                # Best Parameters of the estimator
                best_fit_model.best_params_,
                # Metric_0 of the best estimator (4 decimals)
                round(best_fit_model.best_score_, 4),
                # Metric_1 of the best estimator (4 decimals)
                round(best_fit_model.cv_results_['mean_test_sc2'][best_model.best_index_], 4)
            )
        )

        # Return latex code of cv_results
        # Split each model's results into
        # separate arrays based on metric
        latex_list.append(returnLatexDf(best_fit_model, metrics, k))

    # Print results - latex code
    [print(*l) for l in latex_list]

    # Sort the best_model_list (desc order) based on the scores (metric0, metric1)
    return sorted(best_model_list, key=operator.itemgetter(3, 4), reverse=True)


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
X_train, X_test, y_train, y_test = preprocess_and_split(
    df, cols_to_remove, class_name, class_dict, test_size=0.2, random_state=42)

# %%


"""
Training - Section
"""

# Train the models
final_model = train_models(choose_models(), X_train,
                           y_train, metrics=metrics, refit='sc1', k=5)
best_model_name, best_model, best_model_params, best_score_0, best_score_1 = final_model[0][
    0], final_model[0][1], final_model[0][2], final_model[0][3], final_model[0][4]

# %%


"""
Evaluation - Section
"""

# Train Set Evaluation
y_train_pred = best_model.predict(X_train)
# Test Set Evaluation
y_test_pred = best_model.predict(X_test)

# Get class names and labels
class_names = list(class_dict.keys())
labels = list(class_dict.values())

# Confusion Matrix on the train set
cm_train = confusion_matrix(y_train, y_train_pred, labels=labels)
# Confusion Matrix on the test set
cm_test = confusion_matrix(y_test, y_test_pred, labels=labels)

# Classification Report on the train set
cr_train = classification_report(
    y_train, y_train_pred, labels=labels, target_names=class_names)
# Classification Report on the test set
cr_test = classification_report(
    y_test, y_test_pred, labels=labels, target_names=class_names)

# %%


"""
Results - Section
"""

# Print Results
print('Best Model: ', best_model_name)
print('Best Parameter(s): ', best_model_params)
print('Cross Validation - Best Mean Score 0: ', best_score_0)
print('Cross Validation - Best Mean Score 1: ', best_score_1)

# Train Set Results
print('\nClassification Report - Train Set: \n')
print(cr_train)
print('Confusion Matrix - Train Set: ')
print(cm_train)

# Plot Confusion Matrix - Train Set
plot_confusion_matrix(cm_train, title = "Confusion Matrix - Train Set", target_names=class_names)
plt.show()

# Test Set Results
print('\nClassification Report - Test Set: \n')
print(cr_test)
print('Confusion Matrix - Test Set: ')
print(cm_test)

# Plot Confusion Matrix - Test Set
plot_confusion_matrix(cm_test, title = "Confusion Matrix - Test Set", target_names=class_names)
plt.show()

# Export the final model
joblib.dump(best_model, export_model_filename)

# %%