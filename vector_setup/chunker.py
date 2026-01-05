import itertools

chunks = []
packaged_chunks = []

def chunk_text(
    text: str,
    chunk_size: int = 1000,     # characters
    overlap: int = 200,         # characters
):
    """
    Splits text into overlapping chunks.
    """
    assert overlap < chunk_size, "overlap must be smaller than chunk_size"

    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]

        last_period = chunk.rfind(".")
        if last_period > 0:
            chunk = chunk[: last_period + 1]

        chunks.append(chunk)

        if end >= text_length:
            break

        start = end - overlap

    return chunks

def package_chunks(chunks):
    for idx, chunk_text in enumerate(raw_chunks):
        packaged_chunks.append(
            {
                "id": idx + 1,                 # must be unique
                "text": chunk_text.strip(),
                "source": "knowledge_base.txt",
                "title": "Knowledge Base",
                "chunk_index": idx,
            }
        )

    return packaged_chunks
    


raw_chunks = chunk_text(
    open("knowledge_base.txt", "r", encoding="utf-8").read(),
    chunk_size=1000,
    overlap=200,
)
package_chunks(raw_chunks)

# sanity check
print(f"Total chunks: {len(packaged_chunks)}\n")
print(packaged_chunks[0]["text"][:500])
