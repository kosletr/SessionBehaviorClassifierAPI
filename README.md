# Session Behavior Classifier API


# Motivation

The following API was created for the scope of my Bachelor Thesis in the Aristotle University of Thessaloniki during the academic year 2019-2020.

This Python Flask API leverages Machine Learning techniques to classify a website user's behavior as normal or abnormal based on their stored HTTP activity/sessions (in MongoDB). It may also be used to automatically blacklist users if they exceed a number of active abnormal sessions. After an IP gets blacklisted, all its abnormal sessions are changed from active to inactive. 

In a nutshell the Session Behavior Classifier API serves following purposes:

- Extract Features from User's Web Sessions.
- Use Pretrained Machine Learning Models to Classify HTTP Activity as Normal or Abnormal.
- Blacklist Malicious IP Addresses.

The provided API uses two classifiers to predict user's session behavior. The first model is a multi-class text-classifier used for classifying session's request-bodies as an attack payload [SQL Injection, XSS, Path Traversal, Command Injection] or not. The second one is a feature classifier model used for classifying sessions based on the rest of their features. If any of the models classifies its inputs as abnormal then session's overall behavior is classified as abnormal. Afterwards the API stores an instance containing the various features, properties and behavior in the database's collection  `sesFeatures`.

This is a fully customizable API. Feel free to add/remove code, change machine learning models etc. based on your needs.

## Prerequisites

- Seham (npm package)
- Python 3 (tested for Python 3.6)

# Usage

Send a POST request at `/api` endpoint containing a JSON with the following syntax:

```json
{
  "sesID": <sessionID>
}
```
The API looks for the user's requests assigned to this specific Session ID in the MongoDB Database specified by `MONGO_URI` in the collection `httplogs`.

If the request is valid and the SessionID exists, then the API responds with a status code `200` and a message containing the request.

## Installation - Flask API Integration

Create a `.env` file and add your `MONGO_URI` details as:

```markdown
MONGO_URI=<MONGO-URI>
```

Add any additional settings to the `.flaskenv` file.


## Normal Installation

Install python requirements by running the following command:

```python
pip install -r requirements
```

Run:

```python
flask run
```

## Deploy on Heroku

Create a new file named `run.py` and add the following:

```python
from api import create_app

app = create_app()
```

Then create another file named `Procfile` and add the following line:

```markdown
web: gunicorn run:app
```


## Docker Installation

Add your settings to docker-compose.yml file.

Run Docker Compose:

```
 docker-compose up
```

Note: To run the container with a local database you must change `.env` file's `MONGO_URI` as shown below:

```
MONGO_URI=mongodb://host.docker.internal:27017/DB_NAME
```

## Properties

The Session Behavior Classifier has the following properties:

| Name             | Default          | Description      |
| ---------------- | ---------------- | ---------------- |
| `blockIP`        |        False     | If False then no IPs will be blacklisted.|
| `maxAbnormal `   |        2         | Maximum number of active abnormal sessions before an IP gets blacklisted.|
| `custGroupToSess`| timestamp based  | Use a custom function to group HTTP into sessions.|
| `useModels`      |        True      | If False then only feature extraction (no behavior evaluation) takes place.|
| `modelFile`      |    'model.h5'    | Name of the feature classifier saved model.|
| `textModelFile`  |  'textModel.h5'  | Name of the text classifier saved model.|
| `dir_path`       |   <Script-Dir>   | Path of the saved models.|

## Features

Feel free to modify `handleFeatures.py` file to add your own features based on your needs and the subject of your website. The default features provided are:

| Name             |  Description     |
| ---------------- | ---------------- |
| `numOfLoginAttempts`| Number of `POST` Requests in `/login` endpoint of the session.|
| `numOfPostReqs `    | Number of `POST` Requests of the session.|
| `numOfReqs`         | Number of Requests of the session.|
| `reqBodiesData`     | A dictionary containing user's text-field input data - request bodies of the session.|
| `sesCodes4xx`       | Number of Response Status Codes from `400` to `499`.|
| `sesDuration`       | Time Difference between the session's timestamp of the first and the last request.|

### Add custom Features

To add your personal features just create a custom method as shown below:

```python
def countGETReqsPerSession(HttpLogs, sesID):
    return HttpLogs.count_documents({"sessionID": sesID, "method": "GET"})
```

where:
- `HttpLogs` is a `PyMongo` instance regarding the connection to MongoDB's database specified by the `MONGO_URI` and the collection `httplogs`.
- `sesID` is the ID of the given session.

and then include it to the `featuresExtract` method:

```python
def featuresExtract(mongo, sesID):
    HttpLogs = mongo.db.httplogs
    return {
            ...
            # Other Features and properties
            ...
            # Append custom feature
            "numOfGetReqs": countGETReqsPerSession(HttpLogs, sesID)
    }
```

# Train your own models

In progress.. Soon enough scripts for automated production of trained feature classifiers will be provided.