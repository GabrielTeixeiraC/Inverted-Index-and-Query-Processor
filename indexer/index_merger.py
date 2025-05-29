import heapq
import json
import os

class IndexMerger:
  def __init__(self, index_path: str):
    self.index_path = index_path
    self.file_pointers = []
    self.heap = []

  def read_next_term(self, fp):
    line = fp.readline()
    if not line:
      print(f"End of file reached for {fp.name}.")
      return None
    term, postings = json.loads(line).values()
    return term, postings

  def merge(self):
    self.partial_index_files = [os.path.join(self.index_path, f) for f in os.listdir(self.index_path) if f.startswith('index_') and f.endswith('.jsonl')]
    self.file_pointers = [open(f, 'r') for f in self.partial_index_files]
    current_terms = []

    for fp in self.file_pointers:
      next_term = self.read_next_term(fp)
      if next_term:
        current_terms.append((*next_term, fp))

    self.heap = [(term, postings, fp) for (term, postings, fp) in current_terms]
    heapq.heapify(self.heap)

    output_path = os.path.join(self.index_path, 'final_index.jsonl')
    with open(output_path, 'w') as out_fp:
      while self.heap:
        term, postings, fp = heapq.heappop(self.heap)
        merged_postings = postings

        # Merge other occurrences of the same term
        while self.heap and self.heap[0][0] == term:
          _, more_postings, other_fp = heapq.heappop(self.heap)
          merged_postings.extend(more_postings)

          next_term = self.read_next_term(other_fp)
          if next_term:
            heapq.heappush(self.heap, (*next_term, other_fp))

        # Write merged term to output file
        out_fp.write(json.dumps({term: merged_postings}) + '\n')

        next_term = self.read_next_term(fp)
        if next_term:
          heapq.heappush(self.heap, (*next_term, fp))

    for fp in self.file_pointers:
      fp.close()
      os.remove(fp.name)
