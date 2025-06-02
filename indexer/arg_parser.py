import argparse
def parse_indexer_args() -> tuple:
  """
  Parses the indexer's command-line arguments.
  Returns:
    tuple: (memory_limit_mb, corpus_path, index_dir)
  """
  # Initialize the argument parser
  parser = argparse.ArgumentParser(description="Indexer Argument Parser")

  parser.add_argument("-m", "--memory_limit_mb", type=int, required=True, help="The memory available to the indexer in MB")
  parser.add_argument("-c", "--corpus_path", type=str, required=True, help="The path to the corpus file to be indexed")
  parser.add_argument("-i", "--index_dir", type=str, required=True, help="The path to the directory where indexes should be written")

  # Parse the command-line arguments
  args = parser.parse_args()

  # Validate that memory is positive 
  if args.memory_limit_mb <= 0:
    parser.error("Memory must be a positive integer.")

  # Validate that the corpus file is a .jsonl file
  if not args.corpus_path.endswith(".jsonl"):
    parser.error("Corpus file must be a .jsonl file.")  

  return args.memory_limit_mb, args.corpus_path, args.index_dir