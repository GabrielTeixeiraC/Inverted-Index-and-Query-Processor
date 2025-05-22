import os
import json
import psutil
from indexer.arg_parser import parse_indexer_args 
from indexer.tokenizer import Tokenizer

ONE_MB = 1024 * 1024
class Indexer:
  """
  A class to handle the indexing of documents.
  """

  def __init__(self):
    """
    Initializes the Indexer class.
    """
    memory_limit, corpus_path, index_path = parse_indexer_args()
    self.memory_limit = memory_limit
    self.corpus_path = corpus_path
    self.index_path = index_path

    self.tokenizer = Tokenizer()

  def index(self):
    """
    Indexes the documents in the corpus.
    """
    print(psutil.Process(os.getpid()).memory_info().rss / ONE_MB)

    with open(self.corpus_path, 'r', encoding='utf-8') as f:
      for line in f:
        parsed_line = json.loads(line)
        text = parsed_line['text']
        tokens = self.tokenizer.tokenize(text)
        print(tokens)

if __name__ == "__main__":
  indexer = Indexer()
  indexer.index()
  print("Indexing completed.")