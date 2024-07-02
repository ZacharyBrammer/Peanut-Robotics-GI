from GetObjects import *
import os

def search_folder(folder_path):
    annotation_folders = []
    contents=[]
    objects = []

    for root, dirs, files in os.walk(folder_path):
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            contents.append(dir_path)


    for folder in contents:
        for root, dirs, files in os.walk(folder_path):
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


