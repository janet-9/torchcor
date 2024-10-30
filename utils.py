import os
import meshio

def file_exists(folder_path, prefix="0.01"):
    # Look for the file whose name starts with the prefix
    for file_name in os.listdir(folder_path):
        if file_name.startswith(prefix):
            file_path = os.path.join(folder_path, file_name)
            # Ensure it is a file (not a directory)
            if os.path.isfile(file_path):
                return True

    return False