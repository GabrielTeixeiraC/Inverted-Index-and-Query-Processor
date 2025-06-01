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
      average_tokens_per_document = stats.get("Average Tokens per Document", 0)



    self.queries = self.load_queries()
    self.queries_tokens = self.get_queries_tokens()
    unique_query_tokens = self.get_unique_query_tokens() 
    lexicon = self.load_lexicon(unique_query_tokens)
    self.index = self.load_index_for_queries(unique_query_tokens)
    self.target_docids = self.get_queries_docids()
    self.all_target_docids = set()
    for docids in self.target_docids:
      self.all_target_docids.update(docids)
  
    print(f"Target docids: {self.all_target_docids}")
    document_index = self.load_document_index()
    print(document_index)
    self.scorer = Scorer(lexicon, document_index, total_docs, average_tokens_per_document)

  def load_queries(self):
    with open(self.queries_file_path, 'r', encoding='utf-8') as f:
      queries = [line.strip() for line in f if line.strip()]
    return queries
  
  def get_unique_query_tokens(self):
    unique_query_tokens = set()
    for query in self.queries:
      tokens = self.tokenizer.tokenize(query)
      unique_query_tokens.update(tokens)
    return unique_query_tokens
  
  def get_queries_tokens(self):
    queries_tokens = []
    for query in self.queries:
      tokens = self.tokenizer.tokenize(query)
      queries_tokens.append(tokens)
    return queries_tokens

  def get_queries_docids(self):
    queries_docids = []
    for tokens_list in self.queries_tokens:
      queries_docids.append(self.get_query_target_docids(tokens_list))
    return queries_docids

  def load_index_for_queries(self, needed_tokens):
    index = {}
    with open(self.index_file_path, 'r', encoding='utf-8') as f:
      for line in f:
        token, postings = json.loads(line).values()
        if token in needed_tokens:
          index[token] = postings
    return index
  
  def load_lexicon(self, needed_tokens):
    lexicon = {}
    with open(os.path.join(self.index_path, "lexicon.jsonl"), 'r', encoding='utf-8') as f: 
      for line in f:
        token_info = json.loads(line)
        if token_info['token'] in needed_tokens:
          lexicon[token_info['token']] = token_info
    return lexicon
  
  def load_document_index(self):
    document_index = {}
    with open(os.path.join(self.index_path, "document_index.jsonl"), 'r', encoding='utf-8') as f:
      for i, line in enumerate(f):
        if str(i + 1).zfill(7) in self.all_target_docids:
          doc_info = json.loads(line)
          document_index[doc_info['id']] = doc_info
    return document_index
  
  def score_document(self, docid, tokens):
    score = 0
    for token in tokens:
        postings = self.index.get(token, [])
        for posting_docid, frequency in postings:
            if posting_docid == docid:
                if self.ranker == "tfidf":
                    score += self.scorer.tfidf(token, frequency)
                elif self.ranker == "bm25":
                    score += self.scorer.bm25(token, frequency, docid)
    return score

  def display_results(self, query, results):
    result = {"Query": query, "Results": []}
    for score, docid in results:
      result["Results"].append({"ID": docid, "Score": score})

    print(json.dumps(result, indent=2, ensure_ascii=False))

  def process_queries(self):
    for i, query in enumerate(self.queries):
        results = self.rank_daat(i)
        self.display_results(query, results)
    return results

  def get_query_target_docids(self, tokens):
    target_docids = set()
    for i, token in enumerate(tokens):
      if i == 0:
        target_docids = {posting[0] for posting in self.index.get(token, [])}
      else:
        possible_docids = {posting[0] for posting in self.index.get(token, [])}
        target_docids.intersection_update(possible_docids)

      if not target_docids:
        break
    return target_docids


  def rank_daat(self, query_index, k=10):
    results = []
    query_target_docids = self.target_docids[query_index]

    for target_docid in query_target_docids:
      score = self.score_document(target_docid, self.queries_tokens[query_index])
      heapq.heappush(results, (score, target_docid))
    return heapq.nlargest(k, results)
  

if __name__ == "__main__":
  processor = Processor()
  results = processor.process_queries()
  print(results)