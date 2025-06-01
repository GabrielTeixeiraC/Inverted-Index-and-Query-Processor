import heapq
import json
import os
from processor.arg_parser import parse_processor_args
from processor.scorer import Scorer
from shared.tokenizer import Tokenizer

class Processor:
  def __init__(self):
    self.index_file_path, self.queries_file_path, self.ranker = parse_processor_args()
    self.tokenizer = Tokenizer()
    self.index_path = os.path.dirname(self.index_file_path)
    indexing_statistics_path = os.path.join(self.index_path, "indexing_statistics.json")
    total_docs = 0
    with open(indexing_statistics_path, 'r', encoding='utf-8') as f:
      stats = json.load(f)
      total_docs = stats.get("Number of Documents", 0)

    self.queries = self.load_queries()

    needed_terms = set()
    for query in self.queries:
      terms = self.tokenizer.tokenize(query)
      needed_terms.update(terms)

    lexicon = self.load_lexicon(needed_terms)
    self.scorer = Scorer(lexicon, total_docs)
    self.index = self.load_index_for_queries(needed_terms)

  def load_queries(self):
    with open(self.queries_file_path, 'r', encoding='utf-8') as f:
      queries = [line.strip() for line in f if line.strip()]
    return queries
  
  def load_index_for_queries(self, needed_terms):
    index = {}
    with open(self.index_file_path, 'r', encoding='utf-8') as f:
      for line in f:
        term, postings = json.loads(line).values()
        if term in needed_terms:
          index[term] = postings
    return index
  
  def load_lexicon(self, needed_terms):
    lexicon = {}
    with open(os.path.join(self.index_path, "lexicon.jsonl"), 'r', encoding='utf-8') as f: 
      for line in f:
        token_info = json.loads(line)
        if token_info['token'] in needed_terms:
          lexicon[token_info['token']] = token_info
    return lexicon

  def score_document(self, docid, terms):
    score = 0
    for term in terms:
        postings = self.index.get(term, [])
        for posting_docid, frequency in postings:
            if posting_docid == docid:
                if self.ranker == "tfidf":
                    score += self.scorer.tfidf(term, frequency)
                elif self.ranker == "bm25":
                    score += self.scorer.bm25(term, frequency, docid)
    return score

  def display_results(self, query, results):
    result = {"Query": query, "Results": []}
    for score, docid in results:
      result["Results"].append({"ID": docid, "Score": score})

    print(json.dumps(result, indent=2, ensure_ascii=False))

  def process_queries(self):
    for query in self.queries:
        results = self.rank_daat(query)
        self.display_results(query, results)
    return results


  def rank_daat(self, query, k=10):
    results = []
    terms = self.tokenizer.tokenize(query)
    target_docids = set()
    for i, term in enumerate(terms):
      if i == 0:
        target_docids = {posting[0] for posting in self.index.get(term, [])}
      else:
        possible_docids = {posting[0] for posting in self.index.get(term, [])}
        target_docids.intersection_update(possible_docids)

      if not target_docids:
        break
    for target_docid in target_docids:
      score = self.score_document(target_docid, terms)
      heapq.heappush(results, (score, target_docid))
    return heapq.nlargest(k, results)
  

if __name__ == "__main__":
  processor = Processor()
  results = processor.process_queries()
  print(results)