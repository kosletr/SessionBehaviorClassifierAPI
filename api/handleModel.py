import numpy as np
import pandas as pd
from .utils import transformData

classes = ["Abnormal", "Normal"]
featureLabels = ["numOfLoginAttempts", "numOfPostReqs", "numOfReqs",
                 "sesCodes4xx", "sesDuration"]


def predictBehavior(mongo, sessionDict, useModels, model, textModel):

    # Get the Session ID
    sesID = sessionDict['sessionID']

    # If model classification is enabled
    if useModels == True:

        predModel = 0

        # Text Model Prediction - Text Classification
        # Output is a list of the predictions
        reqBodiesData = list(sessionDict['reqBodiesData'].values())

        if reqBodiesData:
            print(reqBodiesData)
            transformed_data = transformData(
                input_data=reqBodiesData)
            textModelInput = transformed_data.get_all_data()
            predTextModel = np.argmax(
                textModel.predict(textModelInput), axis=-1)
            # normal: 0, sqli: 1, xss: 2, path-traversal: 3, cmdi: 4
            print(predTextModel)

        # If all request bodies contain
        # normal strings (all values are
        # zero) check the rest of the features
        if (not reqBodiesData) or (not any(predTextModel)):
            
            # Keep only the features
            features = sessionDict.copy()
            [features.pop(k, None) for k in list(
                features.keys()) if k not in featureLabels]

            # Get the input ready for the model
            modelInput = pd.DataFrame(features, index=['idx', ]).values
            # print(modelInput)

            # Feature Model Prediction - Output is a list
            # or a list of an array of the prediction based
            # on the type of model (ie. NN, SVM, etc.)
            pred = model.predict(modelInput)

            # Extract the prediction result
            while isinstance(pred, list) or 'ndarray' in str(type(pred)):
                pred = pred[0]
            
            # NN produces a value in range [0,1] as pred
            # svcSVM produces either 0 or 1 as pred
            predModel = int(round(pred))

        # Set behavior - Abnormal/Normal
        sessionDict["behavior"] = classes[predModel]

    # Set the behavior of the session
    mongo.db.sesfeatures.update_one({"sessionID": sesID}, {
        "$set": sessionDict}, upsert=True)


def countAbnormalIPSessions(mongo, ip):
    return mongo.db.sesfeatures.count_documents({"clientIP": ip, "behavior": "Abnormal", "active": True})
