from rag_core import RAGApp

FILE_PATH = "civil_code_raw.txt"

MODEL = "openai/gpt-oss-120b"

def main():
    rag = RAGApp(file_path=FILE_PATH, model=MODEL)

    print("Civil Code RAG ready. Type a question (or 'exit').\n")

    while True:
        q = input(" კითხვა: ").strip()
        if not q:
            continue
        if q.lower() in ("exit", "quit"):
            break

        ans, sources = rag.answer(q, k=5)

        print("\n პასუხი:\n")
        print(ans)

        if sources:
            print("\n Sources used:")
            for s in sources:
                print(f"- SOURCE {s.chunk_id} | {s.title}: {s.snippet}")
        print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    main()
