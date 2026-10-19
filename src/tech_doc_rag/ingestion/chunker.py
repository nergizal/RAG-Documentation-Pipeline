import re
import uuid
from typing import List
from tech_doc_rag.domain.models import DocumentChunk
from tech_doc_rag.ingestion.cleaner import clean_markdown_text


def chunk_markdown_by_headers(
    document_id: str, 
    content: str, 
    source_file: str = "unknown"
) -> List[DocumentChunk]:
    """
    Markdown dokümanını H1, H2, H3 başlıklarına göre parçalar.
    Her parçaya ait başlık hiyerarşisini metadata olarak ekler.
    """
    cleaned_content = clean_markdown_text(content)
    lines = cleaned_content.splitlines()
    
    chunks: List[DocumentChunk] = []
    current_headers = {"h1": "", "h2": "", "h3": ""}
    current_text_lines: List[str] = []
    chunk_index = 0

    header_pattern = re.compile(r"^(#{1,3})\s+(.+)$")

    def create_chunk_object() -> None:
        nonlocal chunk_index
        text_block = "\n".join(current_text_lines).strip()
        if not text_block:
            return

        headers = [h for h in [current_headers["h1"], current_headers["h2"], current_headers["h3"]] if h]
        section_header = " > ".join(headers) if headers else "Root"

        chunk = DocumentChunk(
            id=str(uuid.uuid4()),
            document_id=document_id,
            content=text_block,
            chunk_index=chunk_index,
            metadata={
                "source_file": source_file,
                "section_header": section_header,
                "h1": current_headers["h1"],
                "h2": current_headers["h2"],
                "h3": current_headers["h3"],
                "has_code_block": "```" in text_block
            }
        )
        chunks.append(chunk)
        chunk_index += 1

    for line in lines:
        match = header_pattern.match(line)
        if match:
            create_chunk_object()
            current_text_lines.clear()

            hashes, title = match.groups()
            level = len(hashes)

            if level == 1:
                current_headers["h1"] = title.strip()
                current_headers["h2"] = ""
                current_headers["h3"] = ""
            elif level == 2:
                current_headers["h2"] = title.strip()
                current_headers["h3"] = ""
            elif level == 3:
                current_headers["h3"] = title.strip()
            
            current_text_lines.append(line)
        else:
            current_text_lines.append(line)

    create_chunk_object()
    return chunks