import re
from dataclasses import dataclass
from typing import List, Tuple
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from openai import OpenAI
import os
from text_loader import Chunk, load_text, split_into_chunks

load_dotenv()

def simple_tokenize(text: str) -> List[str]:
    text = text.lower()
    text = re.sub(r"[^\w\s\u10A0-\u10FF]", " ", text)  
    return [t for t in text.split() if t]


@dataclass
class Source:
    chunk_id: int
    title: str
    snippet: str


class RAGApp:
    def __init__(self, file_path: str, model: str):
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            raise RuntimeError("HF_TOKEN env var is missing. Set it before running.")

        self.client = OpenAI(
            base_url="https://router.huggingface.co/v1",
            api_key=hf_token,
        )
        self.model = model

        raw = load_text(file_path)
        self.chunks: List[Chunk] = split_into_chunks(raw)

        tokenized = [simple_tokenize(c.text) for c in self.chunks]
        self.bm25 = BM25Okapi(tokenized)

    def retrieve(self, query: str, k: int = 5) -> List[Chunk]:
        q_tokens = simple_tokenize(query)
        scores = self.bm25.get_scores(q_tokens)
        top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [self.chunks[i] for i in top if scores[i] > 0]

    def answer(self, question: str, k: int = 5) -> Tuple[str, List[Source]]:
        retrieved = self.retrieve(question, k=k)

        if not retrieved:
            return (
                "ფაილში ზუსტად შესაბამისი პასუხი ვერ ვიპოვე. სცადე კითხვა უფრო კონკრეტულად (მაგ: მუხლის ნომერი ან ტერმინი).",
                [],
            )

        context_blocks = []
        sources: List[Source] = []

        for c in retrieved:
            snip = c.text.strip().replace("\n", " ")
            snip = snip[:350] + ("..." if len(snip) > 350 else "")
            sources.append(Source(chunk_id=c.id, title=c.title, snippet=snip))

            context_blocks.append(
                f"[SOURCE {c.id} | {c.title}]\n{c.text.strip()}\n"
            )

        system = (
            "You are a legal Q&A assistant. Answer ONLY using the provided SOURCES.\n"
            "If the answer is not in the sources, say you couldn't find it in the file.\n"
            "Always include citations like (SOURCE 12) next to the relevant sentence.\n"
            "Keep it clear and short."
        )

        user = (
            f"QUESTION (in Georgian): {question}\n\n"
            "SOURCES:\n"
            + "\n---\n".join(context_blocks)
            + "\n\n"
            "Return:\n"
            "1) Answer in Georgian\n"
            "2) Add citations like (SOURCE X)\n"
        )

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.2,
        )

        answer_text = resp.choices[0].message.content.strip()
        return answer_text, sources
