import json
import sys
import time
from pathlib import Path

from rag import NOT_FOUND, client
from lc_rag import answer      # evaluate the LangChain pipeline; use `from rag import answer` for the plain one

JUDGE_MODEL = "openai/gpt-oss-120b"      # the judge should be a larger model than the chatbot

JUDGE_PROMPT = """You are grading a study chatbot.
Compare the chatbot's answer with the reference answer.
Mark it correct if it contains the key facts of the reference, even if worded differently or in Roman Urdu.
Reply with JSON only: {"correct": true or false, "reason": "<one short sentence>"}"""


def judge(question, reference, reply):
    """Ask a larger LLM whether the chatbot's answer matches the reference."""
    response = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[
            {"role": "system", "content": JUDGE_PROMPT},
            {"role": "user", "content": f"Question: {question}\nReference answer: {reference}\nChatbot answer: {reply}"},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)


def evaluate(items):
    results = []
    for item in items:
        start = time.time()
        reply, chunks = answer(item["question"])
        latency = time.time() - start
        pages = [c["page"] for _, c in chunks]

        if item["page"] is None:                  # the answer is not in the notes
            retrieval_ok = None
            verdict = {"correct": NOT_FOUND in reply, "reason": "the chatbot should have refused"}
        else:
            retrieval_ok = item["page"] in pages  # is the correct page in the top 3?
            verdict = judge(item["question"], item["answer"], reply)

        results.append({**item, "reply": reply, "retrieved_pages": pages,
                        "retrieval_ok": retrieval_ok, "latency": round(latency, 2), **verdict})

        mark = "PASS" if verdict["correct"] else "FAIL"
        r = "-" if retrieval_ok is None else ("ok" if retrieval_ok else "MISS")
        print(f"[{mark}] retrieval={r:<4} {latency:4.1f}s  {item['question']}")
        time.sleep(2)                             # stay under the free-tier rate limit
    return results


def summary(results):
    in_notes = [r for r in results if r["page"] is not None]
    not_in_notes = [r for r in results if r["page"] is None]

    def pct(part, whole):
        if not whole:        # empty list: avoid dividing by zero
            return "n/a"
        return f"{len(part)}/{len(whole)} = {100 * len(part) / len(whole):.0f}%"

    print("\n========== RESULT ==========")
    print(f"Retrieval hit@3   : {pct([r for r in in_notes if r['retrieval_ok']], in_notes)}")
    print(f"Answer accuracy   : {pct([r for r in in_notes if r['correct']], in_notes)}")
    print(f"Correct refusals  : {pct([r for r in not_in_notes if r['correct']], not_in_notes)}")
    print(f"Overall           : {pct([r for r in results if r['correct']], results)}")
    print(f"Average latency   : {sum(r['latency'] for r in results) / len(results):.1f}s")

    failed = [r for r in results if not r["correct"]]
    if failed:
        print("\n---------- Failed questions ----------")
        for r in failed:
            print(f"\nQ: {r['question']}")
            print(f"   Expected page {r['page']}, retrieved {r['retrieved_pages']}")
            print(f"   Chatbot: {r['reply'][:200]}")
            print(f"   Judge  : {r['reason']}")


if __name__ == "__main__":
    items = json.loads(Path("eval_set.json").read_text(encoding="utf-8"))
    if len(sys.argv) > 1:                         # python evaluate.py 3  -> only the first 3 questions
        items = items[:int(sys.argv[1])]
    results = evaluate(items)
    summary(results)
    Path("eval_results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nFull details saved to eval_results.json.")
