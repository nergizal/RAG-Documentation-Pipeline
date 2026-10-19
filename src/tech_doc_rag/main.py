from pathlib import Path
from tech_doc_rag.ingestion.chunker import chunk_markdown_by_headers


def main():
    file_path = Path("data/raw/fastapi_tutorial.md")
    
    if not file_path.exists():
        print(f"Hata: {file_path} adresi altında dosya bulunamadı!")
        return

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    chunks = chunk_markdown_by_headers(
        document_id="doc-001",
        content=content,
        source_file=file_path.name
    )

    print("=" * 60)
    print(f" INGESTION BAŞARILI | Üretilen Chunk Sayısı: {len(chunks)}")
    print("=" * 60 + "\n")

    for chunk in chunks:
        print(f"📌 Chunk #{chunk.chunk_index}")
        print(f" Header : {chunk.metadata.get('section_header')}")
        print(f" Kod Var Mı?: {chunk.metadata.get('has_code_block')}")
        print(" Metin :")
        print(f"{chunk.content[:150]}...")
        print("-" * 60)


if __name__ == "__main__":
    main()