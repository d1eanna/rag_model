import re
from dataclasses import dataclass
from typing import List


@dataclass
class Chunk:
    id: int
    title: str        
    text: str          
    start: int         
    end: int           


ARTICLE_RE = re.compile(r"(?=^\s*მუხლი\s+\d+[​]?\.)", re.MULTILINE)


def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def split_into_chunks(raw: str, fallback_chars: int = 1800, overlap: int = 200) -> List[Chunk]:
    parts = ARTICLE_RE.split(raw)

    if len(parts) >= 10:
        chunks: List[Chunk] = []
        cursor = 0
        cid = 0

        for part in parts:
            part = part.strip()
            if not part:
                continue

            m = re.search(r"მუხლი\s+(\d+[​]?)", part)
            title = f"მუხლი {m.group(1)}" if m else "უცნობი მუხლი"

            start = raw.find(part, cursor)
            if start == -1:
                start = cursor
            end = start + len(part)
            cursor = end

            chunks.append(Chunk(id=cid, title=title, text=part, start=start, end=end))
            cid += 1

        return chunks

    chunks = []
    cid = 0
    i = 0
    n = len(raw)
    while i < n:
        j = min(i + fallback_chars, n)
        text = raw[i:j].strip()
        chunks.append(Chunk(id=cid, title=f"Chunk {cid}", text=text, start=i, end=j))
        cid += 1
        i = j - overlap if (j - overlap) > i else j

    return chunks
