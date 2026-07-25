import chromadb
from sentence_transformers import SentenceTransformer
import ollama

RAG_DB_DIR = "rag_vector_db"

# 1. Load the model manually like the build script did
embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")

client = chromadb.PersistentClient(path=RAG_DB_DIR)

# 2. Get the collection WITHOUT passing embedding_function to avoid the conflict
collection = client.get_collection(name="plant_docs")

def ask_local_rag(query_text, model_name="gemma4"):
  # 3. Format query with the prefix the BGE model expects
  prefixed_query = f"Represent this sentence for searching relevant passages: {query_text}"
  
  # 4. Manually embed the query and normalize it (required for BGE cosine similarity)
  query_vector = embedder.encode(
      [prefixed_query], 
      normalize_embeddings=True
  ).tolist()

  # 5. Search using query_embeddings instead of query_texts
  results = collection.query(query_embeddings=query_vector, n_results=3)
  
  retrieved_chunks = results["documents"][0]
  print(f"Retrieved chunks: {retrieved_chunks}")

  # Format chunks with clear boundaries
  formatted_context = ""
  for i, chunk in enumerate(retrieved_chunks, 1):
    formatted_context += f"--- DOCUMENT CHUNK {i} ---\n{chunk}\n\n"
  print(f"Formatted context is {formatted_context}")

  prompt = f"""You are a precise industrial plant assistant. Answer the user's question using ONLY the factual information provided in the document chunks below. 

CRITICAL RULES:
1. Do not use any external knowledge or assumptions.
2. If the answer is not explicitly contained within the provided chunks, you must state: "I cannot find this information in the plant documents."
3. Cite the specific chunk number if you use its information.

Document Chunks:
{formatted_context}

User Question: {query_text}
"""

  response = ollama.chat(
      model=model_name,
      messages=[{"role": "user", "content": prompt}],
  )

  return response["message"]["content"], retrieved_chunks

if __name__ == "__main__":
  test_query = "Can you give me troubleshooting for Bed Inventory Loss"
  answer, sources = ask_local_rag(test_query)

  print("--- ANSWER ---")
  print(answer)
  print("\n--- SOURCES USED ---")
  for i, src in enumerate(sources, 1):
    print(f"[{i}] {src[:150]}...\n")