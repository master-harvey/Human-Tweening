import json

with open('training_data.json') as json_file:
    data = json.load(json_file)
    for key, value in mydic.iteritems():
        print(key, value)



    
