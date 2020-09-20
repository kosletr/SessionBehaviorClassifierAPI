from flask import Flask, request
from flask_cors import CORS
from flask_pymongo import PyMongo
from kanpai import Kanpai
import os
from keras.models import load_model
import joblib

from .handleFeatures import featuresExtract, setSessionsInactive
from .handleModel import predictBehavior, countAbnormalIPSessions
from .handleIPs import blacklistIP


###################  Parameters  #####################

# If False then no IPs will be blacklisted.
blockIP = False
# Maximum number of abnormal sessions an IP
# may have not to be blacklisted.
maxAbnormal = 2
# If False then only feature extraction
# (no behavior evaluation) takes place.
useModels = True
# If True, then import the model
modelFile = 'model.joblib'
textModelFile = 'textModel.h5'
dir_path = os.path.dirname(os.path.realpath(__file__))

######################################################


# Request Validation
# JSON Input Schema
schema = Kanpai.Object({
    "sesID": Kanpai
    .String(error="Session ID must be string")
    .trim()
    .required(error="Session ID is required")
    .max(64, error='Maximum allowed length is 64')
})


# Create python app
def create_app():
    app = Flask(__name__)
    # Load Environment Variables
    app.config.from_pyfile('settings.py')
    # Cross Origin Resource Sharing
    # Enable AJAX to access MongoDB
    CORS(app)
    # MongoDB instance
    mongo = PyMongo(app)
    # Load ML models
    if useModels:
        textModel = load_model(os.path.join(
            dir_path, textModelFile), compile=False)
        if modelFile.endswith('h5'):
            model = load_model(os.path.join(
                dir_path, modelFile), compile=False)
        else:
            model = joblib.load(open(os.path.join(dir_path, modelFile), 'rb'))
    else:
        textModel = None
        model = None

    # POST /api route
    @app.route('/api', methods=['POST'])
    def index():
        # Check for Bad Request
        session, valid = reqReceiveAndValidate()
        if valid is False:
            return session, 400

        # If Request valid -> get Session ID
        sesID = session['sesID']

        # Check that Session ID exists in DB
        if mongo.db.httplogs.find_one({"sessionID": sesID}) == None:
            return session, 404

        # Extract Session Features
        sessionDict = featuresExtract(mongo, sesID)
        # Classify Session Behavior
        predictBehavior(mongo, sessionDict, useModels, model, textModel)

        if blockIP:
            # Blacklist IP if necessary
            ip = sessionDict['clientIP']
            abnormalCount = countAbnormalIPSessions(mongo, ip)
            if abnormalCount > maxAbnormal:
                blacklistIP(mongo, ip)
                setSessionsInactive(mongo, ip)

        return session, 200

    return app


# Function to validate JSON input
def reqReceiveAndValidate():
    try:
        session = request.get_json(force=True)
    except TypeError:
        return session, False

    if schema.validate(session).get('success', False) is False:
        return session, False

    return session, True
