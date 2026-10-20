import sys
import os

sys.path.insert(0, os.path.abspath("."))

from src.claim_generation.generator import ClaimGenerator
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.sparse import BM25Retriever
from src.retrieval.dense import DenseRetriever

gen = ClaimGenerator(provider_type="mock")
rq = gen.generate_claims("What is the relationship between sleep duration and cognitive performance in adults?")

sparse = BM25Retriever()
sparse.load_index()

dense = DenseRetriever()
dense.load_index()

hybrid = HybridRetriever(sparse_retriever=sparse, dense_retriever=dense)

print("=== TASK 2: RETRIEVER OUTPUT VERIFICATION ===")
for c in rq.generated_claims:
    print(f"\nClaim ID: {c.claim_id}")
    print(f"Claim Text: '{c.claim_text}'")
    
    sparse_docs = sparse.retrieve(c.claim_text, top_k=20)
    dense_docs = dense.retrieve(c.claim_text, top_k=20)
    hybrid_docs = hybrid.retrieve(c.claim_text, mode="hybrid", top_k=5)
    
    print(f" -> Sparse (BM25) Count: {len(sparse_docs)}")
    print(f" -> Dense (FAISS) Count: {len(dense_docs)}")
    print(f" -> Hybrid (RRF) Count: {len(hybrid_docs)}")
    
    for doc in hybrid_docs:
        print(f"   [Rank #{doc['rank']}] Doc ID: {doc['doc_id']} | Score: {doc['score']} | RRF Score: {doc['rrf_score']}")
        print(f"     Title: {doc['title']}")
        print(f"     Text Excerpt: {doc['text'][:120]}...")
