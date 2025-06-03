import os
import json
from typing import Dict, List

class PartialIndexWriter:
  """
  A class to handle writing the partial index to disk for each worker.
  It creates a new index file every time it's flushed.
  """

  def __init__(self, index_path: str, worker_id: int) -> None:
    """
    Args:
      index_path (str): Directory where index files will be saved.
      worker_id (int): ID of the worker process.
    """
    self.index_path = index_path
    self.worker_id = worker_id
    # Counter to track the number of partial indexes written by this worker
    self.counter = 0  
    os.makedirs(self.index_path, exist_ok=True)

  def write_to_disk(self, index: Dict[str, List[Dict[str, int]]]) -> None:
    """
    Writes the index to disk in a new file for each flush.

    Args:
      index (dict[str, list]): The index to write to disk.
    """
    # Create a new file name for each flush
    index_file_path = os.path.join(
      self.index_path, f"partial_index_{self.worker_id}_{self.counter}.jsonl"
    )
    
    with open(index_file_path, "w", encoding="utf-8") as file:
      for token in sorted(index.keys()):
        postings = index[token]
        json_line = json.dumps({"token": token, "postings": postings})
        file.write(json_line + "\n")
    
    # Increment flush counter for next time
    self.counter += 1