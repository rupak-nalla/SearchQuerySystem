from pymongo import MongoClient
from main import main

# # For local MongoDB
# client: MongoClient = MongoClient("mongodb+srv://candidate:aQ7hHSLV9QqvQutP@hardfiltering.awwim.mongodb.net/")

# # OR for MongoDB Atlas (cloud)
# # client = MongoClient("mongodb+srv://<username>:<password>@cluster.mongodb.net/?retryWrites=true&w=majority")

# # Access a database
# db = client["interview_data"]

if __name__ == "__main__":
    main()