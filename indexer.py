import os
import json
import sys
from threading import Thread
import time
import psutil
from multiprocessing import Process, Queue, cpu_count, Event
from indexer.tokenizer import Tokenizer
from indexer.arg_parser import parse_indexer_args
from indexer.in_memory_indexer import InMemoryIndexer
from indexer.index_writer import IndexWriter

ONE_MB = 1024 * 1024

def log(message: str):
  formatted_message = f"{message}"
  print(formatted_message, file=sys.stderr)
  with open("indexer_log.txt", "a", encoding="utf-8") as log_file:
    log_file.write(formatted_message + "\n")

def index_worker(tokenizer, writer, memory_limit, input_queue, stop_event, worker_id):
  log(f"Process {worker_id} started indexing.")
  indexer = InMemoryIndexer(memory_limit)
  docs_processed = 0
  while not stop_event.is_set():
    try:
      batch = input_queue.get(timeout=1)
    except Exception:
      continue
    if batch is None:
      break
    for doc in batch:
      tokens = tokenizer.tokenize(doc['text'])
      limit_reached = indexer.index_document(doc['id'], tokens)
      docs_processed += 1
      if limit_reached:
        log(f"Process {worker_id}: Memory limit reached, writing index {writer.index_id} to disk.")
        writer.write_to_disk(indexer.index)
        indexer.reset_index()
    log(f"Process {worker_id}: Documents processed so far: {docs_processed}")
  if indexer.index:
    writer.write_to_disk(indexer.index)
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
    self.tokenizer = Tokenizer()
    self.index_writer = IndexWriter(self.index_path)
    with open("indexer_log.txt", 'w', encoding='utf-8') as f:
      f.write("Indexer log started.\n")

  def get_memory_usage(self) -> int:
    return psutil.Process(os.getpid()).memory_info().rss / ONE_MB

  def stream_documents(self, batch_size: int = 1000):
    with open(self.corpus_path, 'r', encoding='utf-8') as f:
      batch = []
      for line in f:
        batch.append(json.loads(line))
        if len(batch) >= batch_size:
          yield batch
          batch = []
      if batch:
        yield batch

  def run_multiprocessing_indexing(self, process, num_workers=4):
    input_queue = Queue(maxsize=8)
    stop_event = Event()
    processes = []
    thread_memory_limit = self.memory_budget // num_workers
    for i in range(num_workers):
      p = Process(target=index_worker, args=(self.tokenizer, self.index_writer, thread_memory_limit, input_queue, stop_event, i))
      p.start()
      log(f"Starting worker process {p.pid} with memory limit {thread_memory_limit} MB.")
      processes.append(p)
    monitor_proc = Thread(target=memory_monitor, args=(stop_event, main_process, ))
    monitor_proc.start()
    self.stream_documents(input_queue, batch_size=1000, num_workers=num_workers)
    for p in processes:
      p.join()
    stop_event.set()
    monitor_proc.join()

if __name__ == "__main__":
  indexer = Indexer()
  stop_event = Event()
  main_process = psutil.Process(os.getpid())

  num_workers = min(cpu_count(), 8)
  indexer.run_multiprocessing_indexing(main_process, num_workers=num_workers)
  stop_event.set()
  print("Indexing completed.")
