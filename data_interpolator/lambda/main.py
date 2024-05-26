from os import environ
import boto3
import json
import numpy as np
from uuid import uuid4

dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table(environ['TABLE_NAME'])

def interpolate_path(path):
    """Interpolates the path to have a time resolution of 1ms"""
    interpolated_path = []
    for i in range(len(path) - 1):
        x1, y1, t1 = path[i]
        x2, y2, t2 = path[i + 1]
        
        # Calculate the number of interpolation points needed
        duration = t2 - t1
        if duration > 0:
            for ms in range(duration + 1):
                ratio = ms / duration
                x = x1 + ratio * (x2 - x1)
                y = y1 + ratio * (y2 - y1)
                interpolated_path.append((x, y, t1 + ms))
    return interpolated_path

def handler(event,context):
    """Takes a list of record objects from the path_collector UI and stores the time-series of 
    position changes in the database for later model training. Record objects look like: 
    {"start_timestamp":n,"source":[x,y],"end_timestamp":m,"destination":[x,y],"path":[[x1,y1,t1],[x2,y2,t2],...]} 
    where tn is the unix timestamp in ms when the cursor had the position xn,yn. 
    The distance changes are interpolated such that the timeseries has a resolution of 1ms"""
    try: #attempt to convert body from string to dict
        event.body = json.loads(event.body)
    except:
        pass

    records = event.body #should be an array of record objects
    batch_id = uuid4() #the id for this batch of records

    for record in records:
        # Interpolate the path
        interpolated_path = interpolate_path(record['path'])
        
        # Store the record in DynamoDB
        item = {
            'batch_id': batch_id,
            'start_timestamp': record['start_timestamp'],
            'source': record['source'],
            'end_timestamp': record['end_timestamp'],
            'destination': record['destination'],
            'path': json.dumps(interpolated_path)  # Store path as JSON string
        }

        print("Item: ", item)
        
        table.put_item(Item=item)

    return { 'statusCode': 200, 'body': 'Records stored successfully' }