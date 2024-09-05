import pymongo
from dotenv import load_dotenv
import os

load_dotenv()

url = "mongodb+srv://" + os.getenv("MONGODB_USERNAME") + ":" + os.getenv("MONGODB_PASSWORD") + "@" + os.getenv("MONGODB_HOST")

url = "mongodb://localhost:27017/treksathi"
client = pymongo.MongoClient(url) 

db = client['treksathi_db0']