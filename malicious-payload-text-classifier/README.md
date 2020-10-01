# Malicious Payload Text Classifier


# Motivation

The following scripts were created for the scope of my Bachelor Thesis in the Aristotle University of Thessaloniki during the academic year 2019-2020. 

The purpose of this collection of scripts is to provide a character-level text-classifier to classify text as an attack payload [SQL Injection, XSS, Path Traversal, Command Injection] or not.

In a nutshell the scripts provided do the following:

- Import the provided dataset (Morzeux's HttpParamsDataset + Website Strings dataset -- Check the `datasets` README file).
- Edit/Keep useful information from the dataset.
- Split data to training (60%), validation (20%) and testing (20%) sets and export them into three separate files.
- Apply preprocessing to each set. Assign each carachter of the data to an integer and encode each label to One-Hot-Encoding.
- Train a character-level convolutional neural network on the training set, while using the validation test to avoid overfitting.
- Test the model and evaluate its efficiency on the training set for given metrics.

## Prerequisites

- Python 3 (tested for Python 3.6)

# Usage

Install requirements

```markdown
pip install -r requirements.txt
```

In order to train the model just run:

```markdown
python textClassifierTrain.py
```

## Parameters
You can specify the following parameters:

| Name             | Default           | 
| ---------------- | ----------------  |
| `optimizer`      |       adam        | 
| `batch_size `    |       128         | 
| `epochs`         |        7          |
| `metrics`        | Precision, Recall |

# Credits ✨

This application uses Open Source components. You can find the source code of their open source projects along with license information below. We acknowledge and are grateful to these developers for their contributions to open source.

Project: HttpParamsDataset [https://github.com/Morzeux/HttpParamsDataset](https://github.com/Morzeux/HttpParamsDataset)  
Copyright (c) 2016 Morzeux  
License (MIT) [https://github.com/Morzeux/HttpParamsDataset/blob/master/LICENSE](https://github.com/Morzeux/HttpParamsDataset/blob/master/LICENSE)
  
Project: ChCNN [https://github.com/rashimo/ChCNN](https://github.com/rashimo/ChCNN)  
Copyright (c) 2020 Danijel Grah  
License (MIT) [https://github.com/rashimo/ChCNN/blob/master/LICENSE](https://github.com/rashimo/ChCNN/blob/master/LICENSE)