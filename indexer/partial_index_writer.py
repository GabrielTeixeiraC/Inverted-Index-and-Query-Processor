import os
import json
from typing import Dict, List

class PartialIndexWriter:
  """
  A class to handle writing the partial index to disk for each worker.
  It creates exactly one index file per worker.
  """

  def __init__(self, index_path: str, worker_id: int) -> None:
    """
    Args:
      index_path (str): Directory where index files will be saved.
      worker_id (int): ID of the worker process.
    """
    self.index_path = index_path
    self.worker_id = worker_id
    os.makedirs(self.index_path, exist_ok=True)

    self.index_file_path = os.path.join(
      self.index_path, f"index_{self.worker_id}.jsonl"
    )
    
    self.file = open(self.index_file_path, "w", encoding="utf-8")

  def write_to_disk(self, index: Dict[str, List[Dict[str, int]]]) -> None:
    """
    Writes the index to disk in a thread-safe way.

    Args:
      index (dict[str, list]): The index to write to disk.
    """
    for token in sorted(index.keys()):
      postings = index[token]
      json_line = json.dumps({"token": token, "postings": postings})
      self.file.write(json_line + "\n")

    # Write the data from the buffer to disk
    self.file.flush()
    
  def close(self) -> None:
    """
    Closes the index file. Should be called after finishing writing.
    """
    self.file.close()