#This script is an effort to convert the dynamodb typed json into regular json for use in training

import json
import sys
import os
from boto3.dynamodb.types import TypeDeserializer

def ddb_deserialize(r):
    """
    Deserialize a DynamoDB JSON structure to a regular Python dictionary.

    Parameters:
    r (dict): The DynamoDB JSON structure.

    Returns:
    dict: The deserialized Python dictionary.
    """
    type_deserializer = TypeDeserializer()
    return type_deserializer.deserialize({"M": r})

def process_file(input_filename):
    """
    Process a JSON file containing DynamoDB typed JSON, deserialize it,
    and save the result to a new file with '_new' appended to the name.

    Parameters:
    input_filename (str): The name of the input JSON file.
    """
    try:
        # Read the input file
        with open(input_filename, 'r') as file:
            dynamodb_json = file

        # Check if the input is a list of items or a single item
        if isinstance(dynamodb_json, list):
            deserialized_data = [ddb_deserialize(item) for item in dynamodb_json]
        else:
            deserialized_data = ddb_deserialize(dynamodb_json)
        
        # Generate the output filename
        base, ext = os.path.splitext(input_filename)
        output_filename = f"{base}_new{ext}"
        
        # Write the deserialized data to the output file
        with open(output_filename, 'w') as file:
            json.dump(deserialized_data, file, indent=4)
        
        print(f"Deserialized data has been saved to {output_filename}")

    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <input_json_file>")
        sys.exit(1)
    
    input_filename = sys.argv[1]
    process_file(input_filename)