# Abnormal-Behavior-Detection-System

# Description

This collection of scripts was created for the scope of my Bachelor Thesis in the Aristotle University of Thessaloniki during the academic year 2019-2020. 

Specifically, it contains:

- `Session Behavior Classifier API` which is an API written in Python's Flask framework, responsible for assigning behavior labels in user's web sessions using machine learning techniques.
- `Malicious Payload Text Classifier` which is a text classification model written using python's library Keras (backend: tensorflow), responsible for classifying user's request-bodies as either an attack-payload or normal text.
- `Abnormal Behavior Feature Classifier` which is a binary classification model (or anomaly detection model) written using python's sklearn and Keras libraries, responsible for classifying other http features as either abnormal or normal.
- `Datasets` which contains the data collected from a deployed social-network website for the purpose of evaluating the ability of the implemented system.

## Prerequisites

- Node JS - as a Backend Service
- Express JS - REST API (express - node package)
- MongoDB - Database (mongoose - node package)
- Seham - node package
- Python 3 - Classification Models, API, etc. (Tested on Python 3.6)

More information about the requirements, installation process and execution details can be found in the corresponding folder. 

# Quick Start

The order in which each component should be used is the following:

- Install and configure `seham` on your website.
- Disable the use of the classification models in `Session Behavior Classifier API` in order to collect user's web traffic data.
- Deploy the `Session Behavior Classifier API`.
- Create two datasets (numeric + text features) using `exportFeatures.py` script located in the `datasets` folder and add labels to the `behavior` attribute.
- Train the `Abnormal Behavior Feature Classifier` using the numeric dataset.
- Train the `Malicious Payload Text Classifier` using the provided dataset, after merging it with the extracted string dataset.
- Copy the `model.joblib` (or `model.h5`) and the `textModel.h5` trained models to the `session-behavior-classifier-api\api` directory.
- Enable the use of the classification models in `Session Behavior Classifier API` to start using the system.

The provided `model.joblib` file was created based on `misa-j`'s `social-network` website for the purpose of experimentation and evaluation of the system's efficiency.

# Credits ✨

This application uses Open Source components. You can find the source code of their open source projects along with license information below. We acknowledge and are grateful to these developers for their contributions to open source.

Project: social-network [https://github.com/misa-j/social-network](https://github.com/misa-j/social-network)  
Copyright (c) 2020 Misa Jakovljevic  
License (MIT) [https://github.com/misa-j/social-network/blob/master/LICENSE](https://github.com/misa-j/social-network/blob/master/LICENSE)  

Project: HttpParamsDataset [https://github.com/Morzeux/HttpParamsDataset](https://github.com/Morzeux/HttpParamsDataset)  
Copyright (c) 2016 Morzeux  
License (MIT) [https://github.com/Morzeux/HttpParamsDataset/blob/master/LICENSE](https://github.com/Morzeux/HttpParamsDataset/blob/master/LICENSE)
  
Project: ChCNN [https://github.com/rashimo/ChCNN](https://github.com/rashimo/ChCNN)  
Copyright (c) 2020 Danijel Grah  
License (MIT) [https://github.com/rashimo/ChCNN/blob/master/LICENSE](https://github.com/rashimo/ChCNN/blob/master/LICENSE)