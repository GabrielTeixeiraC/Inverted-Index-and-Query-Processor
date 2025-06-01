import heapq
import json
import os

class IndexMerger:
  def __init__(self, index_path: str):
    self.index_path = index_path
    self.file_pointers = []
    self.heap = []
    self.lexicon = {}

  def read_next_term(self, fp):
    line = fp.readline()
    if not line:
      return None
    data = json.loads(line)
    return data["token"], data["postings"]

  def merge(self):
    partial_index_files = [os.path.join(self.index_path, f) for f in os.listdir(self.index_path) if f.startswith('index_') and f.endswith('.jsonl')]
    self.file_pointers = [open(f, 'r') for f in partial_index_files]
    current_terms = []

    for fp in self.file_pointers:
      next_term = self.read_next_term(fp)
      if next_term:
        current_terms.append((*next_term, fp))

    self.heap = [(term, postings, fp) for (term, postings, fp) in current_terms]
    heapq.heapify(self.heap)

    output_path = os.path.join(self.index_path, 'final_inverted_index.jsonl')
    with open(output_path, 'w', encoding='utf-8') as out_fp, open(os.path.join(self.index_path, 'lexicon.jsonl'), 'w', encoding='utf-8') as lexicon_fp:
      while self.heap:
        term, postings, fp = heapq.heappop(self.heap)
        merged_postings = postings

        while self.heap and self.heap[0][0] == term:
            _, more_postings, other_fp = heapq.heappop(self.heap)
            merged_postings.extend(more_postings)

            next_term = self.read_next_term(other_fp)
            if next_term:
                heapq.heappush(self.heap, (*next_term, other_fp))

        doc_frequency = len(merged_postings)
        term_frequency_corpus = sum(freq for _, freq in merged_postings)

        # Write merged term to output index
        out_fp.write(json.dumps({"token": term, "postings": merged_postings}) + '\n')

        # Write lexicon entry line-by-line
        lexicon_entry = {
            "token": term,
            "doc_frequency": doc_frequency,
            "term_frequency_corpus": term_frequency_corpus
        }
        lexicon_fp.write(json.dumps(lexicon_entry) + '\n')

        next_term = self.read_next_term(fp)
        if next_term:
            heapq.heappush(self.heap, (*next_term, fp))

    for fp in self.file_pointers:
      fp.close()
      os.remove(fp.name)
