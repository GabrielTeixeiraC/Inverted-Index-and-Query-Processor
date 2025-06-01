import argparse
def parse_processor_args() -> tuple:
  """
  Parses the processor's command-line arguments.
  Returns:
    tuple: (index_file_path, queries_file_path, ranker)
  """
  # Initialize the argument parser
  parser = argparse.ArgumentParser(description="Processor Argument Parser")

  parser.add_argument("-i", "--index_file_path", type=str, required=True, help="The path to the index file.")
  parser.add_argument("-q", "--queries_file_path", type=str, required=True, help="The path to a file with the list of queries to be processed.")
  parser.add_argument("-r", "--ranker", type=str, required=True, help="A string informing the ranking function to be used to score documents for each query. Options are: 'bm25', 'tfidf'.")

  # Parse the command-line arguments
  args = parser.parse_args()

  # Validate that the index file is a .jsonl file
  if not args.index_file_path.endswith(".jsonl"):
    parser.error("Index file must be a .jsonl file.")  
   
  if args.ranker not in ['bm25', 'tfidf']:
    parser.error("Ranker must be one of the following: 'bm25', 'tfidf'.")

  return args.index_file_path, args.queries_file_path, args.ranker