import json

# List of JSON file paths
file_paths = [
    "results/steal_context_20241114_143438.json",
    "results/steal_context_20241114_170643.json",
    "results/steal_context_20241120_162559.json",
    "results/steal_context_20241115_160101.json",
    "results/steal_context_20241115_143940.json"
]

# Initialize variables to track total score and count
total_similarity_score = 0
similarity_count = 0

# Iterate through the files
for file_path in file_paths:
    try:
        # Load each JSON file
        with open(file_path, 'r') as file:
            data = json.load(file)
        
        # Check for "results" field and iterate through it
        if "results" in data:
            for entry in data["results"]:
                if "similarity_score" in entry:
                    total_similarity_score += entry["similarity_score"]
                    similarity_count += 1
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")

# Calculate the average similarity score
if similarity_count > 0:
    average_similarity_score = total_similarity_score / similarity_count
    print(f"Processed {similarity_count} similarity_score fields.")
    print(f"Average similarity_score: {average_similarity_score:.4f}")
else:
    print("No similarity_score fields found.")
