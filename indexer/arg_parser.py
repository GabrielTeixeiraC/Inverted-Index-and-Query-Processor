import argparse
def parse_indexer_args() -> tuple:
  """
  Parses the indexer's command-line arguments.
  Returns:
    tuple: (memory_limit, corpus_path, index_path)
  """
  # Initialize the argument parser
  parser = argparse.ArgumentParser(description="Web Crawler Argument Parser")

  parser.add_argument("-m", "--memory_limit", type=int, required=True, help="The memory available to the indexer in MB")
  parser.add_argument("-c", "--corpus_path", type=str, required=True, help="The path to the corpus file to be indexed")
  parser.add_argument("-i", "--index_path", type=str, required=True, help="The path to the directory where indexes should be written")

  # Parse the command-line arguments
  args = parser.parse_args()

  # Validate that memory is positive 
  if args.memory_limit <= 0:
    parser.error("Memory must be a positive integer.")

  # Validate that the corpus file is a .jsonl file
  if not args.corpus_path.endswith(".jsonl"):
    parser.error("Corpus file must be a .jsonl file.")  

  return args.memory_limit, args.corpus_path, args.index_path