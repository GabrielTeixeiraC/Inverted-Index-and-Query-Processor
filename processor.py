import heapq
import json
import os
from typing import List, Dict, Set, Tuple

from processor.arg_parser import parse_processor_args
from processor.scorer import Scorer
from shared.tokenizer import Tokenizer

class Processor:
  def __init__(self):
    """
    Initializes the Processor, loading the index, queries, tokenizer,
    lexicon, and inverted index. Also prepares the scorer.
    """
    self.index_file_path, self.queries_file_path, self.ranker = parse_processor_args()
    self.index_dir = os.path.dirname(self.index_file_path)

    self.tokenizer = Tokenizer()

    self.total_docs, self.avg_tokens_per_doc = self._load_indexing_statistics()
    self.queries = self._load_queries()
    self.query_tokens_list = [self.tokenizer.tokenize(q) for q in self.queries]

    needed_tokens = set().union(*self.query_tokens_list)

    self.lexicon = self._load_jsonl_with_filter("lexicon.jsonl", key='token', keys_filter=needed_tokens)
    self.inverted_index = self._load_inverted_index(needed_tokens)

    self.scorer = Scorer(
      self.lexicon,
      {},  # Will load document index later
      self.total_docs,
      self.avg_tokens_per_doc,
      ranker=self.ranker
    )

  def _load_indexing_statistics(self) -> Tuple[int, float]:
    """
    Loads indexing statistics including total document count and 
    average tokens per document.

    Returns:
      Tuple of (total_docs, avg_tokens_per_doc)
    """
    stats_path = os.path.join(self.index_dir, "indexing_statistics.json")
    with open(stats_path, 'r', encoding='utf-8') as f:
      stats = json.load(f)
    return stats.get("Number of Documents", 0), stats.get("Average Tokens per Document", 0)

  def _load_queries(self) -> List[str]:
    """
    Loads queries from the queries file.

    Returns:
      List of query strings.
    """
    with open(self.queries_file_path, 'r', encoding='utf-8') as f:
      return [line.strip() for line in f if line.strip()]

  def _load_inverted_index(self, needed_tokens: Set[str]) -> Dict[str, List[Tuple[str, int]]]:
    """
    Loads the inverted index filtering by needed tokens.

    Args:
      needed_tokens: Set of tokens required.

    Returns:
      Dictionary mapping token to list of (docid, frequency).
    """
    index = {}
    with open(self.index_file_path, 'r', encoding='utf-8') as f:
      for line in f:
        token, postings = json.loads(line).values()
        if token in needed_tokens:
          index[token] = postings
    return index

  def _load_jsonl_with_filter(
    self,
    filename: str,
    key: str,
    keys_filter: Set[str]
  ) -> Dict[str, Dict]:
    """
    Loads a JSONL file and filters its contents by key.

    Args:
      filename: Name of the JSONL file.
      key: Key to filter on.
      keys_filter: Set of keys to keep.

    Returns:
      Dictionary mapping the key to its JSON object.
    """
    path = os.path.join(self.index_dir, filename)
    result = {}
    with open(path, 'r', encoding='utf-8') as f:
      for line in f:
        item = json.loads(line)
        if item[key] in keys_filter:
          result[item[key]] = item
    return result

  def _get_matching_docids(self, tokens: List[str]) -> Set[str]:
    """
    Retrieves document IDs that contain all the tokens in the query
    (AND operation over posting lists).

    Args:
      tokens: List of tokens in the query.

    Returns:
      Set of matching document IDs.
    """
    postings = [set(docid for docid, _ in self.inverted_index.get(token, [])) for token in tokens]
    return set.intersection(*postings) if postings else set()

  def _score_document(self, docid: str, tokens: List[str]) -> float:
    """
    Computes the relevance score of a document for a given query.

    Args:
      docid: Document ID.
      tokens: List of tokens in the query.

    Returns:
      Document score as a float.
    """
    score = 0.0
    for token in tokens:
      postings = self.inverted_index.get(token, [])
      for posting_docid, frequency in postings:
        if posting_docid == docid:
          if self.ranker == "tfidf":
            score += self.scorer.compute_tfidf(token, frequency, docid)
          elif self.ranker == "bm25":
            score += self.scorer.compute_bm25(token, frequency, docid)
    return score

  def _rank_documents(
    self, 
    query_tokens: List[str], 
    docids: Set[str], 
    k: int = 10
  ) -> List[Tuple[float, str]]:
    """
    Ranks documents based on their scores for a given query.

    Args:
      query_tokens: List of tokens from the query.
      docids: Set of candidate document IDs.
      k: Number of top results to return.

    Returns:
      List of tuples (score, docid) sorted by score descending.
    """
    heap = []
    for docid in docids:
      score = self._score_document(docid, query_tokens)
      heapq.heappush(heap, (score, docid))
    return heapq.nlargest(k, heap)

  def _display_results(self, query: str, results: List[Tuple[float, str]]):
    """
    Prints the results for a query in JSON format.

    Args:
      query: The original query string.
      results: List of (score, docid) tuples.
    """
    output = {
      "Query": query,
      "Results": [{"ID": docid, "Score": score} for score, docid in results]
    }
    print(json.dumps(output, indent=2, ensure_ascii=False))

  def process_queries(self):
    """
    Main entry point to process all queries:
    - Finds matching documents.
    - Loads document metadata.
    - Ranks documents.
    - Displays the results.
    """
    all_docids = set().union(*(self._get_matching_docids(tokens) for tokens in self.query_tokens_list))

    self.scorer.document_index = self._load_jsonl_with_filter(
      "document_index.jsonl",
      key='id',
      keys_filter=all_docids
    )

    for i, query in enumerate(self.queries):
      tokens = self.query_tokens_list[i]
      docids = self._get_matching_docids(tokens)

      if not docids:
        self._display_results(query, [])
        continue

      results = self._rank_documents(tokens, docids)
      self._display_results(query, results)

if __name__ == "__main__":
  processor = Processor()
  processor.process_queries()
