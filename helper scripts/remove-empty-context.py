import json

def clean_json_file(file_name):
    # Read the JSON file
    with open(file_name, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # Check if the "results" category exists in the data
    if "results" in data:
        # Find entries in "results" where "context" is empty and remove them
        initial_length = len(data["results"])
        data["results"] = [entry for entry in data["results"] if entry.get("context") != ""]

        # Count the number of removed entries
        removed_count = initial_length - len(data["results"])

        # Output the count of removed entries
        print(f"Removed {removed_count} entries from 'results' where 'context' was empty.")

    # Overwrite the original file with the updated content
    with open(file_name, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4)

# Input the file name
file_name = "results/steal_context_20241115_143940.json"
clean_json_file(file_name)
