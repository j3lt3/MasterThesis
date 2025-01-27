import json

def process_similarity_scores(file_name):
    # Read the JSON file
    with open(file_name, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # Check if "results" category exists in the data
    if "results" in data:
        removed_llm = 0  # Counter for removed entries
        removed_ans = 0  # Counter for removed entries

        # Iterate over each item in the "results" list
        for entry in data["results"]:
            # Check if "similarity_score" is less than 0.9
            if entry.get("similarity_score", 1) < 0.9:
                # Truncate "context" and "response" to 1000 characters if they exist
                context = entry.get("context", "")[:1000]
                response = entry.get("response", "")[:1000]

                # Output the context and response for review
                print(f"\nContext:\n {context}")
                print(f"\nResponse:\n {response}")

                # Ask the user for input
                user_input = input("Type 'a' to accept or 'rllm'/'rans' to reject this entry: ").strip().lower()

                # If user rejects, remove the entry
                if user_input == 'rllm':
                    data["results"].remove(entry)
                    removed_llm += 1
                    print(f"\nRemoved because of LLM rejection.\n")
                if user_input == 'rans':
                    data["results"].remove(entry)
                    removed_ans += 1
                    print(f"\nRemoved because the LLM answered a question instead of outputting context.\n")

        # Output how many entries were removed
        print(f"\nRemoved {removed_llm} entries where there was an LLM rejection.")
        print(f"\nRemoved {removed_ans} entries where the LLM answered a question instead of outputting context.")

    # Overwrite the original file with the updated content
    with open(file_name, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4)

# Input the file name
#file_name = "results/steal_context_20241114_143438.json"
#file_name = "results/steal_context_20241114_170643.json"
#file_name = "results/steal_context_20241120_162559.json"
#file_name = "results/steal_context_20241115_160101.json"
file_name = "results/steal_context_20241115_143940.json"
process_similarity_scores(file_name)
