from qdrant_client import QdrantClient
from fastembed import TextEmbedding

# Initialize client and embedder
client = QdrantClient(url="http://localhost:6333")
embedder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
COLLECTION = "support_kb"

print("=" * 80)
print("CHECKING QDRANT DATABASE")
print("=" * 80)

# 1) Retrieve specific points by ID (first and third point)
print("\n" + "=" * 80)
print("RETRIEVING SPECIFIC POINTS BY ID")
print("=" * 80)

point_ids = [1, 3]
retrieved_points = client.retrieve(
    collection_name=COLLECTION,
    ids=point_ids,
    with_vectors=True
)

for point in retrieved_points:
    print(f"\nPoint ID: {point.id}")
    print(f"Vector: {point.vector}")
    print(f"Text: {point.payload.get('text')}")
    print(f"Source: {point.payload.get('source')}")
    print(f"Title: {point.payload.get('title')}")
    print("-" * 80)

# 2) Perform a semantic search query (like an app would)
print("\n" + "=" * 80)
print("SEMANTIC SEARCH QUERY")
print("=" * 80)

# Example search query - modify this to test different queries
search_query = "How do I get a refund?"
print(f"\nQuery: '{search_query}'")
print("-" * 80)

# Embed the query
query_vector = next(embedder.embed([search_query]))

# Convert numpy array to list of floats
query_vector_list = [float(x) for x in query_vector]

# Search for similar points
search_results = client.query_points(
    collection_name=COLLECTION,
    query=query_vector_list,
    limit=5  # Return top 5 results
).points

print(f"\nFound {len(search_results)} results:\n")

for idx, result in enumerate(search_results, 1):
    print(f"Result {idx}:")
    print(f"  Point ID: {result.id}")
    print(f"  Score: {result.score:.4f}")
    print(f"  Text: {result.payload.get('text')[:200]}...")  # First 200 chars
    print(f"  Source: {result.payload.get('source')}")
    print(f"  Title: {result.payload.get('title')}")
    print("-" * 80)

# 3) Get collection info
print("\n" + "=" * 80)
print("COLLECTION INFO")
print("=" * 80)

collection_info = client.get_collection(collection_name=COLLECTION)
print(f"\nCollection: {COLLECTION}")
print(f"Total points: {collection_info.points_count}")
print(f"Vector size: {collection_info.config.params.vectors.size}")
print(f"Distance: {collection_info.config.params.vectors.distance}")

print("\n" + "=" * 80)
print("CHECK COMPLETE")
print("=" * 80)

