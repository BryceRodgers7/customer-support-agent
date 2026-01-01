from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

# Example: many common embedding models are 384, 768, 1024, 1536, etc.
# If you use fastembed below, we'll infer size by generating one embedding.
from qdrant_client.http.models import PointStruct
from fastembed import TextEmbedding  # installed via qdrant-client[fastembed]
from chunker import packaged_chunks

client = QdrantClient(url="http://localhost:6333")
embedder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")  # fast + good demo choice
COLLECTION = "support_kb"

# Your corpus: list of dicts like:
# chunks = [{"id": 1, "text": "...", "source": "refund_policy.md", "title": "Refunds"}, ...]
chunks = packaged_chunks

# 1) get vector size from one embedding
sample_vec = next(embedder.embed([chunks[0]["text"]]))
VECTOR_SIZE = len(sample_vec)

# 2) create collection
client.recreate_collection(
    collection_name=COLLECTION,
    vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
)

# 3) upsert points
points = []
texts = [c["text"] for c in chunks]
vectors_iter = embedder.embed(texts)

def insert_first_point(chunks, vectors_iter):
    # Get the first vector from the generator
    first_vector = next(vectors_iter)

    #create the first point
    # Debug: check the type and content of first_vector
    print(f"Type of first_vector: {type(first_vector)}")
    print(f"First few elements: {first_vector[:5] if hasattr(first_vector, '__getitem__') else 'N/A'}")

    # Convert numpy array to list of floats
    vector_list = [float(x) for x in first_vector]
    print(f"Vector length: {len(vector_list)}")
    print(f"First few vector values: {vector_list[:5]}")

    points.append(
        PointStruct(
            id=chunks[0]["id"],
            vector=vector_list,
            payload={
                "text": chunks[0]["text"],
                "source": chunks[0].get("source"),
                "title": chunks[0].get("title"),
            },
        )
    )

    #insert the first point only
    client.upsert(collection_name=COLLECTION, points=[points[0]])
    print("Inserted 1 point successfully")

def insert_points(chunks, vectors_iter):
    points = []
    for c, v in zip(chunks, vectors_iter):
        # Convert numpy array to list of floats
        vector_list = [float(x) for x in v]
        points.append(
            PointStruct(
                id=c["id"],
                vector=vector_list,
                payload={
                    "text": c["text"],
                    "source": c.get("source"),
                    "title": c.get("title"),
                },
            )
        )

    # Write all points to points.txt for verification
    with open("points.txt", "w", encoding="utf-8") as f:
        for point in points:
            f.write(f"{point.id}\n")
            f.write(f"{point.vector}\n")
            f.write(f"{point.payload['text']}\n")
            f.write("\n")
    print("Created points.txt with all point data")

    client.upsert(collection_name=COLLECTION, points=points)
    print("Inserted:", len(points))

insert_points(chunks, vectors_iter)