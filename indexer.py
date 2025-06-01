import collections
import os
import sys
import json
import time
import psutil
from multiprocessing import Event, Process, Queue, cpu_count
from threading import Thread
from indexer.arg_parser import parse_indexer_args
from indexer.in_memory_indexer import InMemoryIndexer
from indexer.index_merger import IndexMerger
from indexer.index_writer import IndexWriter
from shared.tokenizer import Tokenizer

ONE_MB = 1024 * 1024

def log(message: str):
  formatted_message = f"{message}"
  print(formatted_message, file=sys.stderr)
  with open("indexer_log.txt", "a", encoding="utf-8") as log_file:
    log_file.write(formatted_message + "\n")

def index_worker(index_path, memory_limit, input_queue, stop_event, worker_id):
    log(f"Process {worker_id} started indexing.")
    indexer = InMemoryIndexer(memory_limit)
    writer = IndexWriter(index_path, worker_id)
    tokenizer = Tokenizer()

    docs_processed = 0
    total_tokens = 0

    while not stop_event.is_set():
        try:
            batch = input_queue.get(timeout=1)
        except Exception:
            continue
        if batch is None:
            break

        for doc_info in batch:
            docid, text = doc_info["id"], doc_info["text"]
            tokens = tokenizer.tokenize(text)
            total_tokens += len(tokens)

            tokens_counter = collections.Counter(tokens)
            limit_reached = indexer.index_document(docid, tokens_counter)
            docs_processed += 1

            if limit_reached:
                log(f"Process {worker_id}: Memory limit reached, writing index {writer.index_id} to disk. Current memory: {psutil.Process(os.getpid()).memory_info().rss / ONE_MB:.2f} MB")
                writer.write_to_disk(indexer.index)
                indexer.reset_index()

        log(f"Process {worker_id}: Documents processed so far: {docs_processed}")

    if indexer.index:
        writer.write_to_disk(indexer.index)

    # ✅ Write worker stats
    worker_stats = {
        "docs_processed": docs_processed,
        "total_tokens": total_tokens
    }
    with open(os.path.join(index_path, f'worker_{worker_id}_stats.json'), 'w') as f:
        json.dump(worker_stats, f)

    log(f"Process {worker_id} finished indexing. Total documents processed: {docs_processed}")


def memory_monitor(stop_event, process, interval=1.0, log_file="memory_usage_log.txt"):
  log(f"Starting memory monitor for process {process.pid} with interval {interval} seconds.")
  with open(log_file, "w", encoding="utf-8") as f:
    f.write(f"time_seconds,parent_{process.pid}_mb,")

    # Write headers for each child process (by pid)
    child_pids = []
    for child in process.children(recursive=True):
      child_pids.append(child.pid)
      f.write(f"child_{child.pid}_mb,")
    f.write("total_mb\n")

    start_time = time.time()
    while not stop_event.is_set():
      mem_main = process.memory_info().rss / ONE_MB
      mem_children = []
      current_children = process.children(recursive=True)
      # Update child_pids if new children appear
      for child in current_children:
        if child.pid not in child_pids:
          child_pids.append(child.pid)
          # Add new column header
          f.seek(0, os.SEEK_END)
          # Not updating header row for new children after start
      mem_by_pid = {child.pid: child.memory_info().rss / ONE_MB for child in current_children}
      total_mb = mem_main + sum(mem_by_pid.get(pid, 0) for pid in child_pids)
      elapsed = time.time() - start_time
      f.write(f"{elapsed:.2f},{mem_main:.2f},")
      for pid in child_pids:
        f.write(f"{mem_by_pid.get(pid, 0):.2f},")
      f.write(f"{total_mb:.2f}\n")
      f.flush()
      time.sleep(interval)

class Indexer:
  def __init__(self):
    self.memory_limit, self.corpus_path, self.index_path = parse_indexer_args()
    self.soft_memory_threshold = 0.8 * self.memory_limit
    self.memory_budget = int(self.soft_memory_threshold - self.get_memory_usage())
    if self.memory_budget <= 0:
      raise ValueError("Memory budget is too low, please increase the memory limit.")
    with open("indexer_log.txt", 'w', encoding='utf-8') as f:
      f.write("Indexer log started.\n")
    os.makedirs(self.index_path, exist_ok=True)
    self.index_merger = IndexMerger(self.index_path)

  def get_memory_usage(self) -> int:
    return psutil.Process(os.getpid()).memory_info().rss / ONE_MB

  def stream_documents(self, queue, batch_size: int = 1000, num_workers: int = 4):
    total_docs = 0

    doc_index_path = os.path.join(self.index_path, 'document_index.jsonl')
    with open(self.corpus_path, 'r', encoding='utf-8') as f_corpus, \
         open(doc_index_path, 'w', encoding='utf-8') as f_doc_index:

        batch = []
        for line in f_corpus:
            doc = json.loads(line)

            doc_metadata = {
                "id": doc["id"],
                "doc_length": len(doc["text"]),
            }
            f_doc_index.write(json.dumps(doc_metadata) + "\n")

            batch.append({"id": doc["id"], "text": doc["text"]})
            total_docs += 1

            if len(batch) >= batch_size:
                queue.put(batch)
                batch = []

        if batch:
            queue.put(batch)

    for _ in range(num_workers):
        queue.put(None)

    return total_docs


  def collect_statistics(self, start_time, total_docs):
    index_file = os.path.join(self.index_path, 'final_inverted_index.jsonl')
    index_size = os.path.getsize(index_file) / ONE_MB  # in MB

    num_lists = 0
    total_postings = 0

    with open(index_file, 'r', encoding='utf-8') as f:
        for line in f:
            num_lists += 1
            postings = json.loads(line)['postings']
            total_postings += len(postings)

    avg_list_size = total_postings / num_lists if num_lists > 0 else 0

    total_tokens = 0
    for filename in os.listdir(self.index_path):
        if filename.startswith('worker_') and filename.endswith('_stats.json'):
            with open(os.path.join(self.index_path, filename), 'r') as f:
                stats = json.load(f)
                total_tokens += stats.get("total_tokens", 0)

    average_doc_length = round(total_tokens / total_docs) if total_docs > 0 else 0
    elapsed_time = time.time() - start_time

    stats = {
        "Index Size (MB)": round(index_size, 2),
        "Elapsed Time (s)": round(elapsed_time, 2),
        "Number of Lists": num_lists,
        "Average List Size": round(avg_list_size, 2),
        "Number of Documents": total_docs,
        "Average Tokens per Document": average_doc_length,
    }

    with open(os.path.join(self.index_path, 'indexing_statistics.json'), 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)

    for filename in os.listdir(self.index_path):
        if filename.startswith('worker_') and filename.endswith('_stats.json'):
            os.remove(os.path.join(self.index_path, filename))

  def run_multiprocessing_indexing(self, process, num_workers=4):
    input_queue = Queue(maxsize=8)
    stop_event = Event()
    processes = []

    NEW_PROCESS_MB = 40
    thread_functional_memory_limit_mb = (self.memory_budget // num_workers) - NEW_PROCESS_MB
    if (thread_functional_memory_limit_mb <= 0):
        raise ValueError("Memory budget is too low, please increase the memory limit.")

    for i in range(num_workers):
        p = Process(target=index_worker, args=(self.index_path, thread_functional_memory_limit_mb, input_queue, stop_event, i))
        log(f"Starting worker process {i} with total memory limit {thread_functional_memory_limit_mb + NEW_PROCESS_MB} MB and functional memory of {thread_functional_memory_limit_mb} MB.")
        p.start()
        processes.append(p)

    monitor_proc = Thread(target=memory_monitor, args=(stop_event, process))
    monitor_proc.start()

    start_time = time.time()
    total_docs = self.stream_documents(input_queue, batch_size=1000, num_workers=num_workers)

    for p in processes:
        p.join()

    self.index_merger.merge()

    stop_event.set()
    monitor_proc.join()

    self.collect_statistics(start_time, total_docs)
if __name__ == "__main__":
  indexer = Indexer()
  stop_event = Event()
  main_process = psutil.Process(os.getpid())

  num_workers = min(cpu_count(), 8)
  indexer.run_multiprocessing_indexing(main_process, num_workers=num_workers)
  stop_event.set()
  print("Indexing completed.")
