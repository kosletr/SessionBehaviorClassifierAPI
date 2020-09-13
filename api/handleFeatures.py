from json_flatten import flatten
from .utils import deleteKeysFromDict


###################  Features  #######################

# Count the number of Response Status Codes from S to S + 100
def numResCodeSxx(HttpLogs, sesID, S):
    return HttpLogs.count_documents(
        {"sessionID": sesID, "statusCode": {"$gte": S, "$lt": S + 100}})


# startend is either 1 (start) or -1 (end)
def getStartEndOfSessionTimestamp(HttpLogs, sesID, startend):
    startendSession = HttpLogs.find_one({
        "sessionID": sesID,
    }, sort=[("timestamp", startend)])
    return startendSession["timestamp"]


def sesDuration(HttpLogs, sesID):
    startSession = getStartEndOfSessionTimestamp(HttpLogs, sesID, 1)
    endSession = getStartEndOfSessionTimestamp(HttpLogs, sesID, -1)
    duration = endSession - startSession
    return int(duration.total_seconds()*1000)


def resContentType(HttpLogs, sesID, type):
    return HttpLogs.count_documents({"sessionID": sesID, "resContentType.type": {"$regex": ".*" + type + ".*"}})


def getIPfromSession(HttpLogs, sesID):
    req = HttpLogs.find_one({"sessionID": sesID})
    return req['clientIP']


def reqsPerSession(HttpLogs, sesID):
    return HttpLogs.count_documents({"sessionID": sesID})


def reqsMethodPerSession(HttpLogs, sesID, method):
    return HttpLogs.count_documents({"sessionID": sesID, "method": method})


def countSpecificRequests(HttpLogs, sesID, reqhttp):
    return HttpLogs.count_documents({"sessionID": sesID, "reqhttp": reqhttp})


def getRequestBodies(HttpLogs, sesID):
    # list of dictionaries of request bodies
    reqBodies = HttpLogs.find({"sessionID": sesID}, {
                              "requestBody": 1, "_id": 0})

    # Remove unecessary items from each dictionary
    keysToRemove = ['user', 'initialFetch', 'followings']
    filteredReqBodies = [deleteKeysFromDict(
        x, keysToRemove) for x in reqBodies]

    # Flatten nested dictionaries
    flatReqBodies = [d for d in list(map(flatten, filteredReqBodies))]

    # Merge list of non-empty dictionaries to a single dictionary
    finalDict = {}
    for idx, x in enumerate(flatReqBodies):
        finalDict.update({(k+"."+str(idx)): v for k,
                          v in x.items() if v != "{}"})

    return finalDict

######################################################


# Save a dictionary with the following
# propertires to a mongo Collection
def featuresExtract(mongo, sesID):
    HttpLogs = mongo.db.httplogs
    return {
        "sessionID": sesID,
        "active": True,
        "behavior": "newSession",
        "endTimestamp": getStartEndOfSessionTimestamp(HttpLogs, sesID, -1),
        "clientIP": getIPfromSession(HttpLogs, sesID),
        "numOfLoginAttempts": countSpecificRequests(HttpLogs, sesID, "POST /login"),
        "numOfPostReqs": reqsMethodPerSession(HttpLogs, sesID, "POST"),
        "numOfReqs": reqsPerSession(HttpLogs, sesID),
        "reqBodiesData": getRequestBodies(HttpLogs, sesID),
        "sesCodes4xx": numResCodeSxx(HttpLogs, sesID, 400),
        "sesDuration": sesDuration(HttpLogs, sesID),
    }


# Inactive sessions are not taken
# into consideration for blacklist
def setSessionsInactive(mongo, ip):
    return mongo.db.sesfeatures.update_many(
        {"clientIP": ip, "active": True}, {"$set": {"active": False}})
