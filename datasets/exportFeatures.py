#%%

import pandas as pd
from pymongo import MongoClient
from datetime import datetime
import csv

#%%

# Read from Mongo and Store into DataFrame
def readMongo(db, collection, mongoUri, query={}):

    # Connect to MongoDB
    conn = MongoClient(mongoUri)
    db = conn[db]

    # Make a query to the specific DB and Collection
    cursor = db[collection].find(query)

    # Expand the cursor and construct the DataFrame
    return pd.DataFrame(list(cursor))

def reqBodiesStrings(reqBodies, label):
    # final list of payloads
    payloads = []

    # For every row in dataset
    for r in reqBodies:
        # Convert dictionary to list and append it
        payloads = payloads + list(r.values())

    # Choose dataset label
    att_label = [label for i,_ in enumerate(payloads)]

    # Return dataframe with the appropriate form for the "malicious payload classifier"
    return pd.DataFrame({'payload': payloads, 'length':[
        str(len(x)) for x in payloads] , 'attack_type': att_label, 'label': att_label})

#%%

# Parameters
dbName = 'dbName'
collectionName = 'sesfeatures'
exportFilename = 'data_'+datetime.utcnow().strftime("%d%m_%H%M")+'.csv'
mongoUri="mongodb://localhost:27017/dbName"

#%%

# Get data from MongoDB Collection (behavior: newSession -> data collection mode)
df = readMongo(dbName, collectionName, mongoUri, query={"behavior": "newSession"})

# Split dataset into two dataframes
df_numeric = df.loc[:, df.columns != 'reqBodiesData']
df_strings = reqBodiesStrings(df['reqBodiesData'], 'norm')

# Export the DataFrames as .csv files
df_numeric.to_csv(exportFilename, index=False)
df_strings.to_csv('string_'+exportFilename, index=False, quoting=csv.QUOTE_ALL)

#%%