import os
import json
from typing import Dict, List

class IndexWriter:
    """
    A class to handle writing the partial indices to disk in a thread-safe manner.
    """

    def __init__(self, index_path: str, worker_id: int):
        self.index_path = index_path
        self.worker_id = worker_id
        self.index_id = 0
        os.makedirs(self.index_path, exist_ok=True)

    def write_to_disk(self, index: Dict[str, List[Dict[str, int]]]) -> str:
        """
        Writes the index to disk in a thread-safe way.

        Args:
            index (dict[str, list]): The index to write to disk.

        Returns:
            str: The path to the written index file.
        """
        file_id = self.index_id
        print(f"Writing index {file_id} of process_id {self.worker_id} to file")
        self.index_id += 1

        index_file_path = os.path.join(
            self.index_path, f"index_{self.worker_id}_{file_id}.json"
        )

        with open(index_file_path, "w", encoding="utf-8") as f:
            json.dump(index, f)

        return index_file_path
