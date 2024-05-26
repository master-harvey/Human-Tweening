from os import environ
import boto3
import json
import random
import string
import math
from concurrent.futures import ThreadPoolExecutor, as_completed

dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table(environ['TABLE_NAME'])

def generate_ID(length=8):
    """Generates a random batch ID"""
    # Define the character set to use: lowercase letters and digits
    characters = string.ascii_lowercase + string.digits
    # Generate a random string of the specified length
    random_string = ''.join(random.choice(characters) for _ in range(length))
    return random_string

# def interpolate_path(path): # Interpolation may not even be necessary, or even a consistent timeseries
#     """Interpolates the path to have a time resolution of 1ms"""
#     interpolated_points = []
#     for i in range(len(path) - 1):
#         x1, y1, t1 = path[i]['x'], path[i]['y'], path[i]['t']
#         x2, y2, t2 = path[i + 1]['x'], path[i + 1]['y'], path[i + 1]['t']
        
#         # Calculate the number of interpolation points needed
#         duration = t2 - t1
#         if duration > 0:
#             for ms in range(duration + 1):
#                 ratio = ms / duration
#                 x = math.floor(x1 + ratio * (x2 - x1))
#                 y = math.floor(y1 + ratio * (y2 - y1))
#                 interpolated_points.append({"x": x, "y": y, "t": t1 + ms})
#     return path+interpolated_points

def process_record(batch_id, record):
    """Processes a single record and stores it in DynamoDB"""
    item = {
        'batch_id': batch_id,
        'start_timestamp': record['start_timestamp'],
        'source': record['source'],
        'end_timestamp': record['end_timestamp'],
        'destination': record['destination'],
        'raw_path': record['path'],
        # Just store the data, calculations can be done localy during training
        # 'translation': [record['destination'][0] - record['source'][0], record['destination'][1] - record['source'][1]],
        # 'duration': record['end_timestamp'] - record['start_timestamp'],
        # 'interpolated_path': interpolate_path(record['path'])
    }
    return item

def handler(event,context):
    """Takes a list of record objects from the path_collector UI and stores the time-series of 
    position changes in the database for later model training. Record objects look like: 
    {"start_timestamp":n,"source":[x,y],"end_timestamp":m,"destination":[x,y],"path":[[x1,y1,t1],[x2,y2,t2],...]} 
    where tn is the unix timestamp in ms when the cursor had the position xn,yn. 
    The distance changes are interpolated such that the timeseries has a resolution of 1ms"""
    # print("EVENT: ", event, " :EVENT")

    try: #attempt to convert body from string to dict
        event['body'] = json.loads(event['body'])
    except:
        pass

    records = event['body'] #should be an array of record objects
    batch_id = generate_ID() #the id for this batch of records
    
    items = []
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = [executor.submit(process_record, batch_id, record) for record in records]
        
        for future in as_completed(futures):
            item = future.result()
            items.append(item)

    # Perform batch write
    with table.batch_writer() as batch:
        for item in items:
            batch.put_item(Item=item)

    return { 'statusCode': 200, 'body': 'Records stored successfully' }