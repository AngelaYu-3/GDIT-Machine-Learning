from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, SearchRequest, VectorParams, Distance
from sentence_transformers import SentenceTransformer
import numpy as np

# Initialize Qdrant client (local or cloud)
client = QdrantClient("localhost", port=6333)  # Replace with cloud URL if using Qdrant Cloud

# Define the vector size based on your model
# all-MiniLM-L6-v2 produces vectors with 384 dimensions
vector_size = 384

# Create the collection
client.recreate_collection(
    collection_name="my_collection",
    vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
)

# Load the model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Sample documents
documents = [
    "Deep learning approaches for natural language processing",
    "Recent advances in artificial intelligence research",
    "Machine learning algorithms for image recognition",
    "Neural networks and their applications in computer vision",
    "Transformer models for sequence-to-sequence tasks"
]

# Convert documents to vectors
vectors = [model.encode(doc).tolist() for doc in documents]

# Create points with IDs and payloads
points = [
    {"id": i, "vector": vector, "payload": {"text": doc}}
    for i, (doc, vector) in enumerate(zip(documents, vectors))
]

# Upload points to the collection
client.upsert(
    collection_name="my_collection",
    points=points
)

# Sample user input
user_input = "Find research papers on deep learning and AI"

# Convert user input to vector
query_vector = model.encode(user_input).tolist()

# Search using query_points (not search)
search_results = client.query_points(
    collection_name="my_collection",
    query=query_vector,
    limit=5
)

# Debug information
# print(f"Type of search_results: {type(search_results)}")

# Access the actual results through the .results attribute
if hasattr(search_results, 'points'):
    results_to_process = search_results.points
    
    # Print the number of results
    print(f"Found {len(results_to_process)} results")
    
    # Process each result
    for i, result in enumerate(results_to_process):
        print(f"Result {i+1}:")
        print(f"  ID: {result.id}")
        print(f"  Score: {result.score}")
        if hasattr(result, 'payload') and result.payload:
            print(f"  Payload: {result.payload}")
        print()
else:
    # If .results doesn't exist, print available attributes
    print(f"Available attributes: {dir(search_results)}")