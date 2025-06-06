import os
import json
import sys
import time
import psutil
import collections
from multiprocessing import Event, Process, Queue, cpu_count
from typing import List, Dict, Optional

from indexer.arg_parser import parse_indexer_args
from indexer.in_memory_indexer import InMemoryIndexer
from indexer.index_merger import IndexMerger
from indexer.partial_index_writer import PartialIndexWriter
from shared.tokenizer import Tokenizer

ONE_MB = 1024 * 1024

def index_worker(
  index_dir: str,
  memory_budget_mb: int,
  input_queue: Queue,
  worker_id: int,
  number_of_workers: int,
  stop_event
) -> None:
  """
  Worker function that indexes documents from the input queue.

  Args:
    index_dir (str): Path to save index files.
    memory_budget_mb (int): Max memory for in-memory indexer (in MB).
    input_queue (Queue): Queue with document batches.
    worker_id (int): ID of the worker.
    number_of_workers (int): Total number of workers. Just for logging.
    stop_event: Event to signal workers to stop.
  """
  print(f"Worker {worker_id} started with memory budget: {memory_budget_mb} MB")
  indexer = InMemoryIndexer(memory_budget_mb)
  writer = PartialIndexWriter(index_dir, worker_id)
  tokenizer = Tokenizer()

  # Timing measurements
  tokenization_time = 0.0
  indexing_time = 0.0
  writing_time = 0.0
  
  total_tokens = 0
  total_documents = 0
  # Create a document index file for this worker
  document_index_path = os.path.join(index_dir, f'document_index_worker_{worker_id}.jsonl')
  with open(document_index_path, 'a', encoding='utf-8') as document_index_fp:
    while not stop_event.is_set():
      try:
        # Get a batch of documents from the input queue
        batch = input_queue.get(timeout=1)
      except Exception:
        continue
      if batch is None:
        break

      for doc in batch:
        if worker_id == 0 and total_documents % 1000 == 0:
          # Print progress assuming equal distribution of documents across workers
          print(f"Approximately {total_documents * number_of_workers} documents indexed so far")

        total_documents += 1
        docid = doc["id"]
        text = doc["text"]

        # Measure tokenization time
        tokenization_start = time.time()
        tokens = tokenizer.tokenize(text)
        tokenization_time += time.time() - tokenization_start

        token_count = len(tokens)
        total_tokens += token_count

        # Create a document metadata entry
        document_metadata = {
          "id": docid,
          "character_count": len(text),
          "token_count": token_count
        }
        # Write the document metadata to the document index file
        document_index_fp.write(json.dumps(document_metadata) + "\n")

        tokens_counter = collections.Counter(tokens)

        # Measure indexing time
        indexing_start = time.time()
        memory_limit_reached = indexer.index_document(docid, tokens_counter)
        indexing_time += time.time() - indexing_start

        # Check if the memory limit is reached
        if memory_limit_reached:
          writing_start = time.time()
          writer.write_to_disk(indexer.index)
          writing_time += time.time() - writing_start
          indexer.reset_index()

    # Write any remaining index data to disk
    if indexer.index:
      writing_start = time.time()
      writer.write_to_disk(indexer.index)
      writing_time += time.time() - writing_start

  # Write worker statistics to a JSON file. This is done here to avoid tokenizing twice.
  with open(os.path.join(index_dir, f'worker_{worker_id}_stats.json'), 'w') as stats_fp:
    json.dump({
      "total_tokens": total_tokens,
    }, stats_fp)
  
  print(f"Worker {worker_id} finished. Total documents indexed: {total_documents}, Total tokens: {total_tokens}")
  if worker_id == 0:
    print("Workers approximate timing statistics:")
    print(f"Tokenization: {tokenization_time:.2f}s, Indexing: {indexing_time:.2f}s, Writing: {writing_time:.2f}s")

class Indexer:
  """
  Orchestrates the indexing pipeline: document reading, parallel indexing,
  merging partial indices, and collecting statistics.
  """
  def __init__(self):
    self.memory_limit_mb, self.corpus_path, self.index_dir = parse_indexer_args()

    # Safeguard against using too much memory
    self.memory_limit_threshold_mb = 0.8 * self.memory_limit_mb
    self.memory_budget_mb = int(self.memory_limit_threshold_mb - self._get_memory_usage())

    if self.memory_budget_mb <= 0:
      raise ValueError("Memory budget is too low. Increase memory limit.")

    os.makedirs(self.index_dir, exist_ok=True)

    self.final_index_path = os.path.join(self.index_dir, 'final_inverted_index.jsonl')
    self.indexing_statistics_path = os.path.join(self.index_dir, 'indexing_statistics.json')
    self.document_index_path = os.path.join(self.index_dir, 'document_index.jsonl')
    self.lexicon_path = os.path.join(self.index_dir, 'lexicon.jsonl')

    self.index_merger = IndexMerger(self.index_dir, self.final_index_path, self.document_index_path, self.lexicon_path)

  def _get_memory_usage(self) -> float:
    """
    Returns the current process memory usage in MB.
    """
    return psutil.Process(os.getpid()).memory_info().rss / ONE_MB

  def _stream_documents(
    self,
    queue: Queue,
    batch_size: int,
    number_of_workers: int
  ) -> int:
    """
    Streams documents from the corpus into the input queue in batches.

    Args:
      queue (Queue): Queue to send document batches to.
      batch_size (int): Number of documents per batch.

    Returns:
      int: Total number of documents streamed.
    """
    total_documents = 0
    batch: List[Dict[str, str]] = []
    
    # Open the corpus file and read it line by line
    with open(self.corpus_path, 'r', encoding='utf-8') as corpus_fp:
      for line in corpus_fp:
        doc = json.loads(line)

        # Append the document to the batch
        batch.append({"id": doc["id"], "text": doc["text"]})
        total_documents += 1

        # If the batch size is reached, put it in the queue
        if len(batch) >= batch_size:
          queue.put(batch)
          batch = []

    # If there are any remaining documents in the batch, put them in the queue
      if batch:
        queue.put(batch)

    # Signal the workers that there are no more documents
    for _ in range(number_of_workers):
      queue.put(None)

    return total_documents
  
  def _collect_statistics(self, total_postings: int, number_of_lists: int, elapsed_time: float, total_documents: int) -> None:
    """
    Collects indexing statistics and saves them to disk.

    Args:
      total_postings (int): Total number of postings in the index.
      number_of_lists (int): Total number of unique token lists in the index.
      elapsed_time (float): Total indexing time in seconds.
      total_documents (int): Total number of documents indexed.
    """
    stats = {}
    index_size = os.path.getsize(self.final_index_path) / ONE_MB
    stats["Index Size"] = round(index_size, 2)

    stats["Elapsed Time"] = round(elapsed_time, 2)
    stats["Number of Lists"] = number_of_lists

    average_list_size = total_postings / number_of_lists if number_of_lists else 0
    stats["Average List Size"] = round(average_list_size, 2)

    # Print the assignment's required statistics
    print(json.dumps(stats, indent=2))

    total_tokens = 0
    for file in os.listdir(self.index_dir):
      if file.startswith('worker_') and file.endswith('_stats.json'):
        with open(os.path.join(self.index_dir, file), 'r') as stats_fp:
          worker_stats = json.load(stats_fp)
          total_tokens += worker_stats.get("total_tokens", 0)
    
    average_tokens_per_document = round(total_tokens / total_documents) if total_documents else 0

    # Save other useful statistics
    stats["Number of Documents"] = total_documents
    stats["Average Tokens per Document"] = average_tokens_per_document

    # Write the statistics to a JSON file
    with open(self.indexing_statistics_path, 'w', encoding='utf-8') as stats_fp:
      json.dump(stats, stats_fp, indent=2)

    # Clean worker statistics files
    for file in os.listdir(self.index_dir):
      if file.startswith('worker_') and file.endswith('_stats.json'):
        os.remove(os.path.join(self.index_dir, file))

  def run(self, number_of_workers: Optional[int] = None) -> None:
    """
    Runs the full indexing pipeline using multiprocessing.

    Args:
      number_of_workers (Optional[int]): Number of worker processes. Defaults to CPU count or max 8.
    """
    number_of_workers = number_of_workers or min(cpu_count(), 8)

    input_queue = Queue(maxsize=16)
    stop_event = Event()
    processes: List[Process] = []

    OVERHEAD_MB_PER_PROCESS = 40
    functional_memory_budget_per_worker = (self.memory_budget_mb // number_of_workers) - OVERHEAD_MB_PER_PROCESS
    if functional_memory_budget_per_worker <= 0:
      raise ValueError("Memory budget too low.")

    # Start worker processes
    for worker_id in range(number_of_workers):
      process = Process(
        target=index_worker,
        args=(self.index_dir, functional_memory_budget_per_worker, input_queue, worker_id, number_of_workers, stop_event)
      )
      process.start()
      processes.append(process)

    # Measure overall timing
    total_start_time = time.time()
    
    total_documents = self._stream_documents(input_queue, batch_size=1000, number_of_workers=number_of_workers)

    for process in processes:
      process.join()

    # Merge the partial indexes
    merge_start_time = time.time()
    print("Merging inverted indexes...")
    total_postings, number_of_lists = self.index_merger.merge()
    inverted_merge_time = time.time() - merge_start_time
    print(f"Inverted index merge time: {inverted_merge_time:.2f} seconds")

    # Merge the document indexes
    doc_merge_start_time = time.time()
    print("Merging document indexes...")
    self.index_merger.merge_document_indexes()
    document_merge_time = time.time() - doc_merge_start_time
    print(f"Document index merge time: {document_merge_time:.2f} seconds")

    total_elapsed_time = time.time() - total_start_time

    print(f"Total indexing time: {total_elapsed_time:.2f} seconds")

    # Signal workers to stop
    stop_event.set()
    print("Collecting statistics...")
    self._collect_statistics(total_postings, number_of_lists, total_elapsed_time, total_documents)

if __name__ == "__main__":
  indexer = Indexer()
  indexer.run()
