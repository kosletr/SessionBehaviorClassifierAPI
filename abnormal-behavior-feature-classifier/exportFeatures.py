#%%

import pandas as pd
from pymongo import MongoClient
from datetime import datetime

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

#%%

# Parameters
dbName = 'socialNetwork'
collectionName = 'sesfeatures'
exportFilename = 'data_'+datetime.utcnow().strftime("%d%m_%H%M")+'.csv'
mongoUri="mongodb://localhost:27017/socialNetwork"


#%%

# Get data from MongoDB Collection
df = readMongo(dbName, collectionName, mongoUri)

# Export DataFrame as a .csv file
df.to_csv(exportFilename, index=False)

#%%