# Parallel Inverted Index Construction and Query Processing System

A scalable, memory-efficient, and parallelized inverted index builder with support for ranked query processing using TF-IDF and BM25. Designed for large-scale document collections, it handles millions of documents under tight memory constraints with multiprocessing and external memory algorithms.

## 📌 Features

- ⚙️ **Parallel Indexing** — Scales with multiple processes
- 🧠 **Memory-aware In-Memory Indexing** — Flushes to disk based on memory usage
- 🔀 **Efficient Index Merging** — External k-way merge using min-heaps
- 🔍 **Ranked Retrieval** — Supports TF-IDF and BM25 ranking functions
- 📑 **Lexicon and Document Index** — Provides detailed corpus and term-level statistics
- 🚀 **Fast Query Processing** — Selective loading of postings and metadata for efficiency

## 📂 Project Structure

```
.
├── indexer/
│   ├── indexer.py              # Main indexing orchestrator
│   ├── in_memory_indexer.py    # Memory-aware inverted index builder
│   ├── partial_index_writer.py # Handles partial index file writing
│   ├── index_merger.py         # K-way merge for index consolidation
│   └── arg_parser.py          # Command-line argument parser for indexer
├── processor/
│   ├── processor.py           # Query processing engine
│   ├── scorer.py             # TF-IDF and BM25 scoring algorithms
│   └── arg_parser.py         # Command-line argument parser for processor
├── shared/
│   └── tokenizer.py          # Text preprocessing and tokenization
├── docs/
│   └── report.tex           # Technical report with complexity analysis
├── indexer.py              # Indexing entry point
├── processor.py            # Query processing entry point
└── README.md

## ⚙️ Installation

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/Inverted-Index-and-Query-Processor.git
cd Inverted-Index-and-Query-Processor
```

2. **Install dependencies:**
```bash
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

## 🚀 Usage

### Indexing Phase
Build an inverted index from a document corpus:

```bash
python indexer.py --memory_limit_mb 2048 \
                  --corpus_path path/to/corpus.jsonl \
                  --index_dir ./index_output
```

### Query Processing Phase
Process queries using the built index:

```bash
python processor.py --index_file_path ./index_output/final_inverted_index.jsonl \
                    --queries_file_path queries.txt \
                    --ranker bm25
```

## ✅ Arguments

### Indexer
| Argument | Short | Description | Required |
|----------|-------|-------------|----------|
| `--memory_limit_mb` | `-m` | Memory available to the indexer in MB | Yes |
| `--corpus_path` | `-c` | Path to the corpus file (must be .jsonl format) | Yes |
| `--index_dir` | `-i` | Directory where index files will be written | Yes |

**Example:**
```bash
python indexer.py -m 1024 -c documents.jsonl -i ./indexes
```

### Processor
| Argument | Short | Description | Required |
|----------|-------|-------------|----------|
| `--index_file_path` | `-i` | Path to the final inverted index file (must be .jsonl) | Yes |
| `--queries_file_path` | `-q` | Path to file containing queries (one per line) | Yes |
| `--ranker` | `-r` | Ranking function: `bm25` or `tfidf` | Yes |

**Example:**
```bash
python processor.py -i ./indexes/final_inverted_index.jsonl -q queries.txt -r bm25
```
