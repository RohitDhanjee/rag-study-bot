from embeddings import embed, cosine_similarity

sentences = [
    "Pipelining increases the throughput of a processor.",
    "Pipeline se processor zyada instructions per second chala leta hai.",   # same meaning in Roman Urdu
    "A cache stores recently used data close to the CPU.",
    "Biryani is cooked with rice, meat and spices.",
]

vectors = embed(sentences, task_type="SEMANTIC_SIMILARITY")

print(f"Vector shape: {vectors.shape}")      # (4, 384)
print(f"First 5 numbers of sentence 0: {vectors[0][:5]}\n")

print("Similarity of sentence 0 to the others:")
for i in range(1, len(sentences)):
    score = cosine_similarity(vectors[0], vectors[i])
    print(f"  {score:.3f}  ->  {sentences[i]}")
