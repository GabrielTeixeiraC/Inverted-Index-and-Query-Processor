from collections import defaultdict
import gc
import os
from typing import Dict, List, Tuple

import psutil

ONE_MB = 1024 * 1024 

class InMemoryIndexer:
    def __init__(self, thread_memory_limit_mb: int):
        self.index = defaultdict(list)
        self.entry_size = 112
        self.thread_memory_limit_mb = thread_memory_limit_mb
        self.entry_count = 0
        self.max_entries = (thread_memory_limit_mb * ONE_MB) // self.entry_size

    def index_document(self, docid: int, tokens_frequency_dict: Dict[str, int]) -> bool:
        for token in tokens_frequency_dict:
            self.index[token].append((docid, tokens_frequency_dict[token]))
            self.entry_count += 1

            if self.entry_count >= self.max_entries and psutil.Process(os.getpid()).memory_info().rss / ONE_MB > self.thread_memory_limit_mb:
                self.entry_count = 0
                return True
                    
        return False

    def reset_index(self):
        self.index = defaultdict(list)
        self.entry_count = 0
        gc.collect()
