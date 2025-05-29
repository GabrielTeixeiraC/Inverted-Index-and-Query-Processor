import heapq
import json
from processor.arg_parser import parse_processor_args
from shared.tokenizer import Tokenizer

class Processor:
  def __init__(self):
    self.index_path, self.queries_path, self.ranker = parse_processor_args()
    self.tokenizer = Tokenizer()
    self.queries = self.load_queries()
    self.index = self.load_index()

  def load_queries(self):
    with open(self.queries_path, 'r', encoding='utf-8') as f:
      queries = [line.strip() for line in f if line.strip()]
    return queries
  
  def load_index(self):
    index = {}
    with open(self.index_path, 'r', encoding='utf-8') as f:
      for line in f:
        doc = json.loads(line)
        index[doc['id']] = doc
    return index
  

  def process_queries(self):
    for query in self.queries:
      tokens = self.tokenizer.tokenize(query)
      results = self.rank_documents(tokens)
      print(f"Query: {query}")
      print("Results:")
      for doc_id, score in results:
        print(f"Document ID: {doc_id}, Score: {score}")
      print("\n")

  def rank_documents(self, tokens):
    if self.ranker == 'bm25':
      return bm25.bm25_ranking(tokens)
    elif self.ranker == 'tfidf':
      return tfidf.tfidf_ranking(tokens)
    
  def daat(self, query, index, k=10):
    results = []
    targets = {docid for term in self.tokenizer.tokenize(query) for docid in index.get(term, {})}
    lists = [index[term] for term in self.tokenizer.tokenize(query) if term in index]
    for target in targets:
      score = 0
      for posting in lists:
        for (docid, frequency) in posting:
          if docid == target:
            score += frequency
        results.heappush(results, (score, target))
    return heapq.nlargest(k, results)