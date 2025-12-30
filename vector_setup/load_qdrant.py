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

for c, v in zip(chunks, vectors_iter):
    points.append(
        PointStruct(
            id=c["id"],
            vector=list(v),
            payload={
                "text": c["text"],
                "source": c.get("source"),
                "title": c.get("title"),
            },
        )
    )

client.upsert(collection_name=COLLECTION, points=points)
print("Inserted:", len(points))
