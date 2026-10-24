from embeddings import embed, cosine_similarity

sentences = [
    "Pipelining increases the throughput of a processor.",
    "Pipeline se processor zyada instructions per second chala leta hai.",
    "A cache stores recently used data close to the CPU.",
    "Biryani is cooked with rice, meat and spices.",
]

vectors = embed(sentences, task_type="SEMANTIC_SIMILARITY")

print(f"Vectors ka shape: {vectors.shape}")      # (4, 768)
print(f"Sentence 0 ke pehle 5 numbers: {vectors[0][:5]}\n")

print("Sentence 0 ka baaqi sab se similarity:")
for i in range(1, len(sentences)):
    score = cosine_similarity(vectors[0], vectors[i])
    print(f"  {score:.3f}  ->  {sentences[i]}")