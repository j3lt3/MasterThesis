import json
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize Presidio components
analyzer_engine = AnalyzerEngine()
anon_engine = AnonymizerEngine()

# List of filenames
files = [
    "results/steal_context_20241114_143438.json",
    "results/steal_context_20241114_170643.json",
    "results/steal_context_20241120_162559.json",
    "results/steal_context_20241115_160101.json",
    "results/steal_context_20241115_143940.json"
]

# Function to load the JSON data
def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

# Load all the JSON files into a list of data
data = [load_json(file) for file in files]

# Extracting responses and contexts from the JSON data
responses_and_contexts = []

for document in data:
    for entry in document.get('results', []):
        response = entry.get('response')
        context = entry.get('context')
        if response and context:
            responses_and_contexts.append((response, context))

# Function to clean response using Presidio
def remove_pii(response_text):
    analyzer_results = analyzer_engine.analyze(text=response_text, language="en")
    censored_text = anon_engine.anonymize(text=response_text, analyzer_results=analyzer_results).text
    return censored_text

# Apply PII removal to each response
cleaned_responses_and_contexts = []

for response, context in responses_and_contexts:
    cleaned_response = remove_pii(response)
    cleaned_responses_and_contexts.append((cleaned_response, context))

# Function to compute cosine similarity between two texts
def compute_cosine_similarity(text1, text2):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([text1, text2])
    return cosine_similarity(vectors[0:1], vectors[1:2])[0][0]

# Calculate cosine similarity for each cleaned response/context pair
similarities = []

for cleaned_response, context in cleaned_responses_and_contexts:
    similarity_score = compute_cosine_similarity(cleaned_response, context)
    similarities.append(similarity_score)

# Calculate the average cosine similarity
average_similarity = sum(similarities) / len(similarities) if similarities else 0

# Output the average similarity
print(f"Average Cosine Similarity: {average_similarity:.4f}")
