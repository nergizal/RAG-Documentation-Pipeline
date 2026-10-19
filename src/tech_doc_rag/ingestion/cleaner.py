import re


def clean_markdown_text(text: str) -> str:
    """
    Ham Markdown metnini temizler:
    - Ardışık 3 veya daha fazla yeni satır karakterini 2 yeni satıra indirir.
    - Satır sonlarındaki gereksiz boşlukları siler.
    """
    if not text:
        return ""
    
    lines = [line.rstrip() for line in text.splitlines()]
    cleaned_text = "\n".join(lines)
    cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text)
    
    return cleaned_text.strip()
