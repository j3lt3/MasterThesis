import os
import json
import re

def load_contexts_from_json(file_path):
    """
    Load all 'context' fields from the 'results' field in a JSON file.
    """
    contexts = set()
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
            if 'results' in data and isinstance(data['results'], list):
                for item in data['results']:
                    context = item.get('context')
                    if context:
                        normalized_context = normalize_string(context)
                        contexts.add(normalized_context)
                        #print(f"Loaded and normalized context: '{normalized_context}'")  # Debug print
    except (json.JSONDecodeError, KeyError, IOError) as e:
        print(f"Error reading {file_path}: {e}")
    print(f"Total contexts loaded from {file_path}: {len(contexts)}")  # Debug print
    return contexts

def normalize_string(input_string):
    """
    Normalize a string by converting it to lowercase, stripping whitespace,
    and removing special characters.
    """
    input_string = input_string.strip().lower()  # Convert to lowercase and strip whitespace
    input_string = re.sub(r'\s+', ' ', input_string)  # Replace multiple spaces with a single space
    return input_string

def get_all_files_in_folder(folder_path):
    """
    Recursively get all files in the given folder and its subdirectories.
    """
    all_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            full_path = os.path.join(root, file)
            all_files.append(full_path)
            #print(f"Found file: {full_path}")  # Debug print
    print(f"Total files found in {folder_path}: {len(all_files)}")  # Debug print
    return all_files


def search_contexts_in_files(folder_path, contexts):
    """
    Search for each context string in the normalized contents of all files in the folder.
    """
    all_files = get_all_files_in_folder(folder_path)
    matched_files = set()
    file_context_map = {}

    print(f"Processing {len(all_files)} files...")  # Debug print

    for file_path in all_files:
        try:
            with open(file_path, 'r', errors='ignore') as file:
                content = file.read()
                normalized_content = normalize_string(content)  # Normalize file content
                for context in contexts:
                    if context in normalized_content:  # Compare normalized strings
                        print(f"Match found! Context: '{context}' in file: {file_path}")  # Debug print
                        matched_files.add(file_path)
                        if file_path not in file_context_map:
                            file_context_map[file_path] = []
                        file_context_map[file_path].append(context)
        except IOError as e:
            print(f"Error reading {file_path}: {e}")
    
    print(f"Total matched files: {len(matched_files)}")  # Debug print
    return matched_files, file_context_map

    
def analyze_coverage(folder_path, json_file_paths):
    """
    Analyze how much the contexts from JSON files cover the folder contents.
    """
    # Collect all contexts from the given JSON files
    all_contexts = set()
    for json_file in json_file_paths:
        contexts = load_contexts_from_json(json_file)
        all_contexts.update(contexts)

    if not all_contexts:
        print("No contexts found in the JSON files.")
        return

    # Search for contexts in the files within the folder
    matched_files, file_context_map = search_contexts_in_files(folder_path, all_contexts)

    # Calculate coverage
    total_files = len(get_all_files_in_folder(folder_path))
    coverage_percentage = (len(matched_files) / total_files) * 100 if total_files else 0

    # Display results
    print(f"Total files in folder (including subfolders): {total_files}")
    print(f"Total contexts from JSONs: {len(all_contexts)}")
    print(f"Matched files: {len(matched_files)}")
    print(f"Coverage: {coverage_percentage:.2f}%")
    #print(f"Matched file-context mapping: {file_context_map}")
    
    return {
        "total_files": total_files,
        "total_contexts": len(all_contexts),
        "matched_files": len(matched_files),
        "coverage_percentage": coverage_percentage,
        "file_context_map": file_context_map,
    }

# Example Usage
if __name__ == "__main__":
    # Resolve the folder path relative to the script's location
    base_dir = os.path.dirname(os.path.abspath(__file__))
    folder_path = os.path.join(base_dir, "maildir", "allen-p")
    
    # List of paths to the 5 JSON documents
    json_file_paths = [
        os.path.join(base_dir, "results", "attack_results_20241230_162606.json"),
    ]
    
    # Run analysis
    analyze_coverage(folder_path, json_file_paths)
