from GetObjects import *
import os

def search_folder(folder_path):
    annotation_folders = []
    objects = []

    contents = os.listdir(folder_path)
    folders = [folder for folder in contents if os.path.isdir(os.path.join(folder_path, folder))]
    
    for folder in folders:
        for root, dirs, files in os.walk(folder):
            if 'annotation' in dirs:
                annotation_folders.append(os.path.join(root, 'annotation'))

    
    for folder in annotation_folders:
        files = os.listdir(folder)
        path = os.path.join(folder, files[0])
        items = extract_unique_names(path)
        for item in items:
            objects.append(item)
    
    returned_objects = list(set(objects))
    return returned_objects




folder_path = input("Please enter the path to file: ")

run = search_folder(folder_path)
print("Unique names found:")
for object in run:
    print(object)


