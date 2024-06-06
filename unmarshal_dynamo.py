#This script is an effort to convert the dynamodb typed json into regular json for use in training

import json
import sys
from boto3.dynamodb.types import TypeDeserializer
from decimal import Decimal

def process_file(input_filename):
    """
    Process a JSON file containing DynamoDB typed JSON, deserialize it,
    and save the result to a new file.

    Parameters:
    input_filename (str): The name of the input JSON file.
    """
    type_deserializer = TypeDeserializer()
    
    # Generate the output filename
    output_filename = "deserialized_data.json"
    
    # Read the input file
    with open(input_filename, 'r') as file:
        training_data = file.readlines()
        
        # Write the deserialized data to the output file
        with open(output_filename, 'w') as file:
            newlines = []
            for line in training_data:
                decoded_line = json.loads(line)['Item']
                new_line = {k:int(type_deserializer.deserialize(v)) if type(k) is not str else type_deserializer.deserialize(v) for k,v in decoded_line.items()}
                newlines.append(str(new_line).replace("Decimal('","").replace("')","").replace("'",'"'))
            file.write("["+",\n".join(newlines)+"]")
    
    print(f"Deserialized data has been saved to {output_filename}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <input_json_file>")
        sys.exit(1)
    
    input_filename = sys.argv[1]
    process_file(input_filename)