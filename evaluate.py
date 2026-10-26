import json
import sys
import time
from pathlib import Path

from rag import client
from lc_rag import answer      # LangChain version ko test karne ke liye

NOT_FOUND = "Ye notes mein nahi hai."
JUDGE_MODEL = "openai/gpt-oss-120b"      # judge chatbot se bada model ho to behtar

JUDGE_PROMPT = """You are grading a study chatbot.
Compare the chatbot's answer with the reference answer.
Mark it correct if it contains the key facts of the reference, even if worded differently or in Roman Urdu.
Reply with JSON only: {"correct": true or false, "reason": "<one short sentence>"}"""


def judge(question, reference, reply):
    """Bada LLM check kare ke chatbot ka jawab reference se match karta hai ya nahi."""
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

        if item["page"] is None:                  # jawab notes mein hai hi nahi
            retrieval_ok = None
            verdict = {"correct": NOT_FOUND in reply, "reason": "chatbot ko mana karna tha"}
        else:
            retrieval_ok = item["page"] in pages  # sahi page top 3 mein aaya?
            verdict = judge(item["question"], item["answer"], reply)

        results.append({**item, "reply": reply, "retrieved_pages": pages,
                        "retrieval_ok": retrieval_ok, "latency": round(latency, 2), **verdict})

        mark = "PASS" if verdict["correct"] else "FAIL"
        r = "-" if retrieval_ok is None else ("ok" if retrieval_ok else "MISS")
        print(f"[{mark}] retrieval={r:<4} {latency:4.1f}s  {item['question']}")
        time.sleep(2)                             # free tier ki rate limit se bacho
    return results


def summary(results):
    in_notes = [r for r in results if r["page"] is not None]
    not_in_notes = [r for r in results if r["page"] is None]

    def pct(part, whole):
        if not whole:        # khali list: 0 se divide mat karo
            return "n/a"
        return f"{len(part)}/{len(whole)} = {100 * len(part) / len(whole):.0f}%"

    print("\n========== RESULT ==========")
    print(f"Retrieval hit@3   : {pct([r for r in in_notes if r['retrieval_ok']], in_notes)}")
    print(f"Answer accuracy   : {pct([r for r in in_notes if r['correct']], in_notes)}")
    print(f"Sahi mana kiya    : {pct([r for r in not_in_notes if r['correct']], not_in_notes)}")
    print(f"Overall           : {pct([r for r in results if r['correct']], results)}")
    print(f"Average latency   : {sum(r['latency'] for r in results) / len(results):.1f}s")

    failed = [r for r in results if not r["correct"]]
    if failed:
        print("\n---------- FAIL hue sawal ----------")
        for r in failed:
            print(f"\nQ: {r['question']}")
            print(f"   Expected page {r['page']}, search ne diye {r['retrieved_pages']}")
            print(f"   Chatbot: {r['reply'][:200]}")
            print(f"   Judge  : {r['reason']}")


if __name__ == "__main__":
    items = json.loads(Path("eval_set.json").read_text(encoding="utf-8"))
    if len(sys.argv) > 1:                         # python evaluate.py 3  -> sirf pehle 3 sawal
        items = items[:int(sys.argv[1])]
    results = evaluate(items)
    summary(results)
    Path("eval_results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nPoori detail eval_results.json mein save ho gayi.")