import os
import shutil

# Define file mappings (source -> target)
file_mappings = {
    'system_environment.py': 'scripts/system_environment.py',
    'quantum_processor.py': 'core/quantum_processor.py',
    'main_training_loop.py': 'scripts/main_training_loop.py',
    'integrate.py': 'scripts/integrate.py',
    'q_learning_optimizer.py': 'scripts/q_learning_optimizer.py',
    'run.py': 'scripts/run.py',
    'tests.py': 'scripts/tests.py',
}

# Define the source directory where the files are currently located
source_directory = '/mnt/data/'

# Ensure the directories exist
directories = [
    'scripts',
    'core',
    'tests'
]

for directory in directories:
    if not os.path.exists(directory):
        os.makedirs(directory)

# Move and rename files based on the mapping
for original_filename, new_filename in file_mappings.items():
    source_path = os.path.join(source_directory, original_filename)
    destination_path = os.path.join(os.getcwd(), new_filename)  # Destination path in current directory
    
    if os.path.exists(destination_path):
        print(f"File already exists at {destination_path}, skipping move.")
    elif os.path.exists(source_path):
        shutil.move(source_path, destination_path)
        print(f'Moved: {original_filename} to {destination_path}')
    else:
        print(f'File not found: {original_filename}')

