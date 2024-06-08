import json

def preprocess_mouse_movements(data):
    """Convert path data from absolute into relative coordinates (points in time to steps with delay)"""
    results = []
    
    for item in data:
        translation = item['translation']
        timeframe = item['end_timestamp'] - item['start_timestamp']
        path = item['raw_path']
        
        steps = []
        prev_point = None
        for point in path:
            if prev_point is not None:
                dx = point['x'] - prev_point['x']
                dy = point['y'] - prev_point['y']
                dt = point['t'] - prev_point['t']
                if dt < 3:
                    dt = 3 #near 0 values are unacceptable but I don't feel like looking into it further
                steps.append({"x": dx, "y": dy, "t": dt})
            prev_point = point
        
        # Construct the new object
        results.append({"translation": {"x": translation[0], "y": translation[1], "t": timeframe}, "steps": steps})
    
    return results

# Example usage:
with open('deserialized_data.json') as file:
    data = json.loads(file.read())
    with open('preprocessed_training_data.json','w') as output_file:
        json.dump(preprocess_mouse_movements(data), output_file, indent=2)