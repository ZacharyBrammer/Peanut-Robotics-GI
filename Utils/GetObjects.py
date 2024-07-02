import json
import re

def extract_unique_names(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    # Convert JSON data to string
    json_string = json.dumps(data)
    
    # Extract the relevant substring between "objects": [ and "extrinsics":
    start_index = json_string.find('"objects": [')
    end_index = json_string.find('"extrinsics":', start_index)
    
    if start_index == -1 or end_index == -1:
        raise ValueError('The required sections ("objects": [ and "extrinsics":) were not found in the JSON file.')
    
    relevant_json_string = json_string[start_index:end_index]
    
    # Regular expression to find names
    pattern = re.compile(r'"name":\s*"([^"]+)"')
    matches = pattern.findall(relevant_json_string)
    
    # Extract unique names
    unique_names = set(matches)
    
    return unique_names

# Ask the user for the file path

# file_path = input("Please enter the path to the JSON file: ")

# try:
#     unique_names = extract_unique_names(file_path)
#     print("Unique names found:")
#     for name in unique_names:
#         print(name)
# except FileNotFoundError:
#     print("The file was not found. Please check the path and try again.")
# except json.JSONDecodeError:
#     print("There was an error decoding the JSON. Please check the file content.")
# except ValueError as e:
#     print(e)
