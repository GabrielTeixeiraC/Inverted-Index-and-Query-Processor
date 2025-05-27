import os
import json
import sys
from threading import Thread

import psutil
from indexer.tokenizer import Tokenizer
from indexer.arg_parser import parse_indexer_args 
from indexer.in_memory_indexer import InMemoryIndexer
from indexer.index_writer import IndexWriter
from indexer.index_merger import IndexMerger

def log(message: str):
    """
    Logs a message to stderr and to a file with a given level.

    Args:
      message (str): The message to log.
      level (str): Log level ("INFO", "WARNING", "ERROR", "CRITICAL").
    """
    formatted_message = f"{message}"

    # Print to stderr
    print(formatted_message, file=sys.stderr)

    # Write to file
    with open("indexer_log.txt", "a", encoding="utf-8") as log_file:
        log_file.write(formatted_message + "\n")

def index_worker(documents_chunk, tokenizer, memory_limit, index_path, thread_id):
  log(f"Thread {thread_id} started indexing {len(documents_chunk)} documents.")
  indexer = InMemoryIndexer(thread_memory_limit_mb=memory_limit)
  writer = IndexWriter(index_path)

  for i, doc in enumerate(documents_chunk):
    if i % 1000 == 0:
      log(f"Thread {thread_id} indexed {i} documents so far.")
    text = doc['text']
    tokens_freq = tokenizer.tokenize(text)
    index, mem_limit_reached = indexer.index_document(doc['id'], tokens_freq)

    if mem_limit_reached:
      log(f"Thread {thread_id} memory limit reached, writing index to disk.")
      writer.write_to_disk(index)
      indexer.reset_index()
      index = {}

  if index:
    writer.write_to_disk(indexer.index)

ONE_MB = 1024 * 1024

class Indexer:
  """
  A class to handle the indexing of documents.
  """

  def __init__(self):
    """
    Initializes the Indexer class.
    """
   
    self.memory_limit, self.corpus_path, self.index_path = parse_indexer_args()
    self.soft_memory_threshold = 0.8 * self.memory_limit 
    self.memory_budget = int(self.soft_memory_threshold - self.get_memory_usage())
    if self.memory_budget <= 0:
      raise ValueError("Memory budget is too low, please increase the memory limit.")
    
    self.tokenizer = Tokenizer()
    
    with open("indexer_log.txt", 'w', encoding='utf-8') as f:
      f.write("Indexer log started.\n")
  
  def get_memory_usage(self) -> int:
    """
    Returns the current memory usage of the process in MB.
    """  
    return psutil.Process(os.getpid()).memory_info().rss / ONE_MB
  
  def run_multithreaded_indexing(self):
    with open(self.corpus_path, 'r', encoding='utf-8') as f:
      lines = [json.loads(line) for line in f.readlines()]
    log(f"Loaded {len(lines)} documents from {self.corpus_path}.")
    num_threads = 4 
    chunk_size = len(lines) // num_threads
    thread_memory_limit = self.memory_budget // num_threads
    threads = []
    for i in range(num_threads):
      chunk = lines[i * chunk_size: (i + 1) * chunk_size]
      t = Thread(target=index_worker, args=(chunk, self.tokenizer, thread_memory_limit, self.index_path, i))
      t.start()
      threads.append(t)

    for t in threads:
      t.join()

if __name__ == "__main__":
  indexer = Indexer()
  indexer.run_multithreaded_indexing()
  print("Indexing completed.")