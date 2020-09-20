# Abnormal Behavior Feature Classifier


# Motivation

The following scripts were created for the scope of my Bachelor Thesis in the Aristotle University of Thessaloniki during the academic year 2019-2020.

The purpose of this collection of scripts is to provide a classification model that given a set of a user's session features classifies the session as normal or abnormal. Moreover, this code is also responsible for extracting the dataset from the MongoDB database and also apply Grid-Search with K-Fold Cross Validation Techniques on multiple classifier-algorithms to identify the one with the highest efficiency.

In a nutshell the scripts provided do the following:

- `exportFeatures.py`: Extract data from a MongoDB collection to a `.csv` file provided the `MONGO_URI`, database and collection name.
- `classificationScript.py`: Run Grid-Search with 5-Fold Cross Validation on the extracted data to decide which classifier (and for which parameters) is the most efficient. After extracting the best estimator, evaluate it and export it.  
- `anomalyDetScript.py`: Run Grid-Search on the extracted data to decide which anomaly detection algorithm (and for which parameters) is the most efficient. After extracting the best estimator, evaluate it and export it.  

## Prerequisites

- Python 3 (tested for Python 3.6)

# Classifiers

The deafult classifiers being used within the code of the `gridSearch.py` script are listed below:

| Classifier          | Hyper-Parameters    |
| ------------------- | ------------------- |
Multi-Layer Perceptron - Neural Network 1 | layers: [12, 15], activation: [relu, relu, relu], dropout_rate: [0.8, 0.5], batch_size: 16, epochs: 100, optimizer: adam |
Multi-Layer Perceptron - Neural Network 2 | layers: [20, 35], activation: [relu, relu, relu], dropout_rate: [0.5, 0.5], batch_size: 16, epochs: 100, optimizer: adam |
Multi-Layer Perceptron - Neural Network 3 | layers: [15, 18, 8], activation: [relu, relu, relu, relu], dropout_rate: [0.8, 0.5, 0.2], batch_size: 16, epochs: 100, optimizer: adam |
Support Vector Classifier with Linear Kernel | C: 0.001, 0.01, 0.1, 1, 10, 100 |
Support Vector Classifier with Radial Kernel | C: 0.001, 0.01, 0.1, 1, 10, 100, gamma: 0.001, 0.01, 0.1, 1, 10, 100 |
K Nearest Neighbors Classifier |  n_neighbors: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 |
Decision Tree Classifier | max_depth: 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14 |
Random Forest Classifier -Linear SVM | n_estimators: 10, 20, 50, 100, 200 |
Stochastic Gradient Descent Classifier | max_iter: 100, 1000, alpha: 0.0001, 0.001, 0.01, 0.1 |
Logistic Regression with LASSO | C: 0.001, 0.01, 0.1, 1, 10, 100 |


Feel free to  add, remove or change parameters, such as:

- The grid search parameters in the `choose_model()` method.
- The `K` cross-validation parameter specifying the number of folds. (default: 5)
- The `random_state` parameter for shuffling the dataset. (default: 42)
- The train-test split ratio, `test_size`. (default: 0.2)
- Other parameters.

or even add new classifiers to the grid search list.

# Usage

Install the requirements:

```markdown
pip install -r requirements.txt
```

## Export Features from MongoDB

To export features from a MongoDB Database - Collection just open the `exportFeatures.py` file using a text editor of your choice and add your settings to the `Parameters` section:

| Parameter           | Description         |
| ------------------- | ------------------- |
| `dbName`            | The name of the database to get the data from. |
| `collectionName`    | The name of the collection in the specified dataset. |
| `exportFilename`    | The name of the file to save the dataset. |
| `mongoUri`          | The URI that identifies the MongoDB Database. |

Then run the script using python:

```markdown
python exportfeatures.py
```

## Grid Search with 5-Fold Cross-Validation

To run the Grid Search with 5-Fold Cross-Validation - Classification script just open the `classificationScript.py` file using a text editor of your choice and add your settings to the `Parameters` section:

| Parameter           | Description         |
| ------------------- | ------------------- |
| `dir_path`          | The path where the dataset files are located. |
| `class_name`        | The name of the class attribute. |
| `class_dict`        | A dictionary containing each class name and a corresponding integer value (ex. {"Abnormal": 0, "Normal": 1}). |
| `cols_to_remove`    | List with the names of any unecessary columns to remove. |
| `metrics`           | List with the names of the metrics used for scoring in CV. Two metrics must be provided to break any ties. |
| `export_model_filename` | The name of the model file to be exported. |

Important 1: To run this script scikit-learn 0.21.2 must be installed!
Important 2: The datasets should have a behavior attribute with a different string for each attack (ex. dos.csv -> behavior: dos, dict_att.csv -> behavior: dict, etc.).

# Anomaly Detection Algorithm

To run the Grid Search Anomaly Detection script just open the `anomalyDetScript.py` file using a text editor of your choice and add your settings to the `Parameters` section as above.
