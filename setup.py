import os

"""Creates the file structure used for this code by default, run this script from within your desired working directory"""

# Function to create directories
def create_folders(root_directory, folder_structure):
    for folder in folder_structure:
        path = os.path.join(root_directory, folder)
        try:
            os.makedirs(path, exist_ok=True)
            print(f'Created: {path}')
        except OSError as e:
            print(f'Error creating {path}: {e}')

# Define your root directory and folder structure
root_directory = './'

# Folder structure you want to create
folder_structure = [
    'Comb_Data/KuHar',
    'Comb_Data/MotionSense',
    'Comb_Data/RealWorld_thigh',
    'Comb_Data/RealWorld_upperarm',
    'Comb_Data/RealWorld_waist',
    'Comb_Data/UCI',
    'Comb_Data/WISDM',
    'output'
]

# Call the function to create the folders
create_folders(root_directory, folder_structure)