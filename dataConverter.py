# Filename: dataConverter.py
# Author: Haythem (Hayhay1231)
# Purpose: to convert download of DynamoDB overall JSON to multiple JSON Files, to work easier? maybe?

from dynamodb_json import json_util as DBjson
import json

mousePaths = []
count = 0
with open('training_data.json') as f:
    for jsonObj in f:
        mousePath = DBjson.loads(jsonObj)
        with open(f"convertedPaths/mousePath{count}.json", "w") as f:
            json.dump(mousePath, f)
        count = count + 1


