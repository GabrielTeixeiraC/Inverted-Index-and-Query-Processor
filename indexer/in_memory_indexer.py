from collections import defaultdict
import gc
from typing import Dict, List, Tuple

ONE_MB = 1024 * 1024 

class InMemoryIndexer:
    def __init__(self, thread_memory_limit_mb: int):
        self.index = defaultdict(list)
        self.thread_memory_limit_mb = thread_memory_limit_mb
        self.memory_per_entry_bytes = 82
        self.max_entries = int((self.thread_memory_limit_mb * ONE_MB) / self.memory_per_entry_bytes)
        self.entry_count = 0

    def index_document(self, docid: int, tokens_frequency_dict: Dict[str, int]) -> Tuple[Dict[str, List[Tuple[int, int]]], bool]:
        for token in tokens_frequency_dict:
            self.index[token].append((docid, tokens_frequency_dict[token]))
            self.entry_count += 1

            if self.entry_count >= self.max_entries:
                self.entry_count = 0
                return self.index, True

        return self.index, False

    def reset_index(self):
        self.index = defaultdict(list)
        self.entry_count = 0
        gc.collect()
