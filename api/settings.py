from dotenv import load_dotenv
from os import environ

load_dotenv('.env')
MONGO_URI = environ.get('MONGO_URI')
