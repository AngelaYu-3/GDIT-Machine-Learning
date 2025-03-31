# Enron Email Dataset Retrieval Augmented Generation System

An advanced system for processing, indexing, and semantically querying the Enron Email Dataset using enhanced chunking strategies, named entity recognition, and large language models.

## Project Overview

This project implements a sophisticated Retrieval Augmented Generation (RAG) system specifically designed for email corpora like the Enron Email Dataset. It features:

- **Enhanced Semantic Chunking**: Text is chunked based on semantic coherence while preserving entity boundaries
- **Named Entity Recognition**: GLiNER integration to extract and utilize entities in the text
- **Overlap Management**: Ensures context continuity between chunks with controlled sentence overlap
- **Vector Search**: FAISS and Qdrant integration for efficient similarity search
- **LLM Integration**: Groq LLM integration for natural language generation
- **Optimized Processing Pipeline**: Batch processing and caching for efficient handling of large datasets

## Installation

### Prerequisites

- Python 3.8+
- Pip

### Setup

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd EnronEmailDataset
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the Enron Email Dataset and place it in the `data` directory or update the path in the code.

4. (Optional) Download or prepare a subset of the dataset in CSV format.

## Usage

### Processing and Indexing Emails

The system can be run in either interactive (Jupyter) or script mode:

#### Jupyter Notebook Mode

1. Open `emailrag3.ipynb` in Jupyter:
   ```bash
   jupyter notebook emailrag.ipynb
   ```

2. Execute the cells sequentially to process the data, create chunks, and build the vector index.

#### Script Mode

1. Run the complete processing pipeline after exporting emailrag3.ipynb to a .py script:
   ```bash
   python emailrag3.py
   ```

### Querying the System

Once the emails are processed and indexed, you can query the system using:

```python
# Example query
query = "Find emails related to energy trading policies"
results = retrieval_chain.invoke({"input": query})
print(results["answer"])
```

## Key Components

### EnhancedSemanticChunker

The core of the system is the `EnhancedSemanticChunker` class which extends LangChain's `SemanticChunker` with:

- Named entity recognition to preserve entity mentions across chunk boundaries
- Sentence-level overlap to maintain context between chunks
- Entity metadata enrichment for better retrieval

```python
chunker = EnhancedSemanticChunker(
    embeddings=embedding_model,
    gliner_model=ner_model,
    breakpoint_threshold_amount=5,
    min_chunk_size=3,
    overlap_sentences=1
)
```

### Groq LLM Integration

The system uses Groq's high-performance LLMs via the `initialize_groq.py` module:

```python
from initialize_groq import init_groq

client, llm = init_groq(model_name="llama-3.3-70b-versatile")
```

### Retrieval Strategies

Different retrieval strategies are implemented:

- **Top-K Similarity**: Find the most similar documents by vector similarity
- **Maximum Marginal Relevance (MMR)**: Balance relevance with diversity
- **Dynamic Threshold Retrieval**: Only retrieve documents above a similarity threshold

## Advanced Features

### Optimized Batching

Process large datasets efficiently with batching and parallel processing:

```python
# Process in batches with parallel execution
enhanced_docslist = optimized_process_emails(email_texts, batch_size=50)
```

### Entity-Enhanced Retrieval

The system includes entities directly in the document content to improve vector search for entity-focused queries:

```
"passage: Meeting scheduled for tomorrow... 
Entities found: date: tomorrow, Thursday; person: John Smith, Sarah Jones; event: meeting, conference call."
```

## Performance Considerations

- Processing large email datasets is resource-intensive. Start with a small subset for testing.
- GPU acceleration significantly speeds up named entity recognition.
- Consider using caching for embeddings and NER to avoid redundant processing.

## File Structure

- `emailrag.py` - Main processing script
- `emailrag.ipynb` - Jupyter notebook version
- `initialize_groq.py` - Groq LLM integration
- `data/` - Directory for email data
- `email_faiss_normalized_e5/` - FAISS vector storage

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The Enron Email Dataset
- LangChain for the foundational RAG components
- GLiNER for Named Entity Recognition
- Groq for LLM API access