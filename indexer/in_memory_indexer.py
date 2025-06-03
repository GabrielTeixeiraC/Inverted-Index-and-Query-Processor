import gc
import os
import sys
import psutil
from collections import defaultdict
from typing import Dict

ONE_MB = 1024 * 1024

class InMemoryIndexer:
  """
  Temporary in-memory indexer that maps tokens to (docid, frequency) pairs.

  The index automatically flushes when the memory limit is reached.
  """
  def __init__(self, memory_budget_mb: int):
    """
    Initialize the indexer with a memory usage limit.

    Args:
      memory_budget_mb (int): The memory budget in megabytes for the indexer process.
    """
    self.memory_budget_mb = memory_budget_mb
    self.index = defaultdict(list)
    
    # Each entry is a tuple (docid, frequency). Since there will be considerably more append 
    # operations than insertions of new tokens, we assume an amortized size for each entry.
    # For instance, in 64-bit Python, the size of an integer is 28 bytes. The size of a 
    # tuple with two integers is 56 bytes 
    # (40 bytes for the tuple itself and 2 * 8 bytes for the integers pointers).
    # Therefore, the total size in bytes is: 56 + 2 * 28 = 112 bytes.
    self.entry_size = sys.getsizeof((0, 0)) + 2 * sys.getsizeof(0)
    
    # Calculate the approximate maximum number of entries that can be stored in the
    # index before reaching the memory budget.
    self.entry_count = 0
    self.max_entries = (memory_budget_mb * ONE_MB) // self.entry_size

  def index_document(self, docid: str, tokens_frequency_dict: Dict[str, int]) -> bool:
    """
    Index a single document into the in-memory index.

    Args:
      docid (str): Document identifier.
      token_frequencies (Dict[str, int]): Mapping from token to its frequency in the document.

    Returns:
      bool: True if the memory budget was reached and the index should be flushed.
    """
    for token in tokens_frequency_dict:
      self.index[token].append((docid, tokens_frequency_dict[token]))
      self.entry_count += 1

      if self.entry_count >= self.max_entries:
        self.entry_count = 0
        return True

    return False

  def reset_index(self):
    """
    Reset the in-memory index, clearing all indexed data.

    Should be called after the index is flushed to disk.
    """
    self.index = defaultdict(list)
    self.entry_count = 0
    gc.collect()
