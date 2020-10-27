# Social Network - Datasets

This dataset was created for the scope of my Bachelor Thesis in the Aristotle University of Thessaloniki during the academic year 2019-2020. The main purpose of this dataset is to enable the creation of a machine learning model that is able to detect user's malicious web sessions provided certain features.

# Tools
For the malicious instances (Abnormal) we used:

- [THC-Hydra](https://github.com/vanhauser-thc/thc-hydra) - for the Brute-Force (Dictionary) attacks.
- [Hulk](https://github.com/grafov/hulk) - for Denial of Service attacks.
- [LOIC](https://github.com/NewEraCracker/LOIC) - for Denial of Service attacks.
- [OWASP ZAP](https://github.com/zaproxy/zaproxy) - for vulnerability testing in general.

while for the benign instances (Normal) we used:

- A [Social-Network](https://github.com/misa-j/social-network) MERN stack website  (MongoDB-ExpressJS-ReactJS-NodeJS).
- Abnormal-Behavior-Detection-System (seham - npm package and session-behavior-classifier API).

# Data

## Malicious Instances

| Category            | Number of Instances |
| ------------------- | ------------------- |
| Dictionary Attacks  |        96           |
| DoS Attacks         |        38           |
| Aggresive Vuln Scan |        71           |

## Total Instances

| Category            | Number of Instances |
| ------------------- | ------------------- |
|  Abnormal           |       205           |
|  Normal             |       800           |


## Export Features from MongoDB

To export features from a MongoDB Database - Collection just open the `exportFeatures.py` file using a text editor of your choice and add your settings to the `Parameters` section:

| Parameter           | Description         |
| ------------------- | ------------------- |
| `dbName`            | The name of the database to get the data from. |
| `collectionName`    | The name of the collection in the specified database. |
| `exportFilename`    | The name of the file to save the dataset. |
| `mongoUri`          | The URI that identifies the MongoDB Database. |

Then run the script using python:

```markdown
python exportfeatures.py
```

The script produces two .csv files. One for the numeric features and another one for the reqBodies string feature in the appropriate form for training.
Cut the string-dataset file and paste it in the `malicious-payload-text-classifier` directory. Merge it with the provided dataset before training.

# Credits ✨

This project uses Open Source components. You can find the source code of their open source projects along with license information below. We acknowledge and are grateful to these developers for their contributions to open source.

Project: social-network [https://github.com/misa-j/social-network](https://github.com/misa-j/social-network)  
Copyright (c) 2020 Misa Jakovljevic  
License (MIT) [https://github.com/misa-j/social-network/blob/master/LICENSE](https://github.com/misa-j/social-network/blob/master/LICENSE)  
