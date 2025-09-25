import os
import re

# Directory containing the TIF files
directory = '/scratch.local3/juanp/dataset/WSI_D2'  # Replace with your directory path

# Define the patterns and their replacements
patterns = {
    r'he': 'HE',
    r'h-e': 'HE',
    r'h\.e': 'HE',
    r'pas': 'PAS',
    r'pm': 'PM',
    r'p\.m': 'PM'
}

# Compile the patterns into regex objects
compiled_patterns = {re.compile(pattern, re.IGNORECASE): replacement for pattern, replacement in patterns.items()}

# Iterate through the files in the directory
for filename in os.listdir(directory):
    # Check if the file is a TIF file
    if filename.lower().endswith('.mat'):
        new_filename = filename
        # Replace the patterns in the filename
        for pattern, replacement in compiled_patterns.items():
            new_filename = pattern.sub(replacement, new_filename)
        
        # Rename the file if the name has changed
        if new_filename != filename:
            old_path = os.path.join(directory, filename)
            new_path = os.path.join(directory, new_filename)
            os.rename(old_path, new_path)
            print(f'Renamed: {filename} -> {new_filename}')
        
        # Check if the new filename contains 'PAS', 'PM', or 'HE'
        if not any(keyword in new_filename for keyword in ['PAS', 'PM', 'HE']):
            print(f"Filename does not contain 'PAS', 'PM', or 'HE': {new_filename}")