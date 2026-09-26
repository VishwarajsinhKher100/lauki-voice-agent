from pathlib import Path

from docling.document_converter import DocumentConverter
from docling_core.transforms.chunker import HierarchicalChunker
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

from .config import settings


def load_document(source: str):
    """
    Parse a document using Docling. Accepts a local file path or URL.
    """
    print(f"  Parsing: {source}")
    converter = DocumentConverter()
    result = converter.convert(source)
    return result.document


def convert_chunk(doc_chunk) -> dict:
    """
    Convert a Docling DocChunk into a plain dictionary.
    Preserves headings as a breadcrumb trail prepended to content.
    """
    headings = getattr(doc_chunk.meta, "headings", []) or []
    content = getattr(doc_chunk, "text", "").strip()
    breadcrumb = " > ".join(headings)
    chunk_text = f"{breadcrumb}\n\n{content}" if breadcrumb else content

    return {
        "headings": headings,
        "content": content,
        "chunk_text": chunk_text,
    }


def ingest_documents(docs_dir: str) -> None:
    docs_path = Path(docs_dir)
    pdf_files = sorted(docs_path.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDF files found in '{docs_dir}'. Add PDFs and try again.")
        return

    print(f"\nFound {len(pdf_files)} PDF(s) in '{docs_dir}':")
    for f in pdf_files:
        print(f"  - {f.name}")

    print("\n--- Step 1: Parsing & Chunking ---")
    all_chunks = []
    chunker = HierarchicalChunker()

    for pdf_path in pdf_files:
        try:
            doc = load_document(str(pdf_path))
            doc_chunks = [convert_chunk(c) for c in chunker.chunk(doc)]
            print(f"  [OK] {pdf_path.name} -> {len(doc_chunks)} chunks")
            all_chunks.extend(doc_chunks)
        except Exception as e:
            print(f"  [FAIL] {pdf_path.name}: {e}")
            continue

    if not all_chunks:
        print("No chunks created. Check your documents and try again.")
        return

    print(f"\nTotal chunks: {len(all_chunks)}")

    print("\n--- Step 2: Generating Embeddings ---")
    embedder = SentenceTransformer(settings.embedding_model)
    chunk_texts = [c["chunk_text"] for c in all_chunks]
    embeddings = embedder.encode(chunk_texts, show_progress_bar=True)
    print(f"[OK] Embedding shape: {embeddings.shape}")

    print("\n--- Step 3: Indexing in Qdrant ---")
    client = QdrantClient(path=settings.qdrant_path)
    dim = embedder.get_sentence_embedding_dimension()

    if client.collection_exists(settings.collection_name):
        client.delete_collection(settings.collection_name)

    client.create_collection(
        collection_name=settings.collection_name,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )

    points = [
        PointStruct(
            id=idx,
            vector=embedding.tolist(),
            payload={
                "headings": chunk["headings"],
                "content": chunk["content"],
                "chunk_text": chunk["chunk_text"],
            },
        )
        for idx, (chunk, embedding) in enumerate(zip(all_chunks, embeddings))
    ]

    client.upsert(collection_name=settings.collection_name, points=points, wait=True)

    info = client.get_collection(settings.collection_name)
    print(f"[OK] Indexed {info.points_count} points (dim={dim})")

    client.close()
    print("\n[DONE] Ingestion complete. You can now start the voice agent.")


if __name__ == "__main__":
    print("Lauki Phones — Document Ingestion Pipeline")
    print(f"  Docs folder:  {settings.docs_dir}")
    print(f"  Embedding:    {settings.embedding_model}")
    print(f"  Qdrant path:  {settings.qdrant_path}")
    print(f"  Collection:   {settings.collection_name}")
    ingest_documents(settings.docs_dir)