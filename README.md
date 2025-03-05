# HCIRAG

A Retrieval Augmented Generation (RAG) system using Elasticsearch for document storage and retrieval of Human Computer Interaction (HCI) required reading.

## Overview

HCIRAG is a powerful RAG system designed to enhance your AI applications by providing efficient document storage, retrieval, and generation capabilities. By leveraging Elasticsearch's advanced search capabilities including vector search, this system offers a robust solution for knowledge management and AI-powered content generation.

## Features

- **Document Storage**: Store and manage documents in Elasticsearch with robust indexing
- **Vector Search**: Find relevant documents using semantic vector search for accurate retrieval
- **Full-text Search**: Leverage Elasticsearch's powerful text search capabilities
- **Content Generation**: Generate high-quality content augmented with retrieved information
- **Scalable Architecture**: Built to handle growing document collections

## Getting Started

### Prerequisites

- Python 3.8+
- Elasticsearch 8.0+
- pip (Python package manager)

### Installation

```bash
git clone https://github.com/n8mauer/HCIRAG.git
cd HCIRAG
pip install -r requirements.txt
```

### Setting up Elasticsearch

1. **Install Elasticsearch**:

   - **Ubuntu/Debian**:
     ```bash
     # Import Elasticsearch GPG key
     wget -qO - https://artifacts.elastic.co/GPG-KEY-elasticsearch | sudo gpg --dearmor -o /usr/share/keyrings/elasticsearch-keyring.gpg
     
     # Add Elasticsearch repository
     echo "deb [signed-by=/usr/share/keyrings/elasticsearch-keyring.gpg] https://artifacts.elastic.co/packages/8.x/apt stable main" | sudo tee /etc/apt/sources.list.d/elastic-8.x.list
     
     # Install Elasticsearch
     sudo apt-get update && sudo apt-get install elasticsearch
     ```

   - **macOS** (using Homebrew):
     ```bash
     brew tap elastic/tap
     brew install elastic/tap/elasticsearch-full
     ```

   - **Windows**: Download installer from the [Elasticsearch website](https://www.elastic.co/downloads/elasticsearch)

2. **Start Elasticsearch**:

   - **Ubuntu/Debian**:
     ```bash
     sudo systemctl start elasticsearch
     sudo systemctl enable elasticsearch
     ```

   - **macOS**:
     ```bash
     elasticsearch
     ```
     Or as a service:
     ```bash
     brew services start elastic/tap/elasticsearch-full
     ```

3. **Verify Elasticsearch is running**:
   ```bash
   curl http://localhost:9200
   ```

### Configuration

1. Create a `.env` file in the root directory based on the provided `.env.example`:
   ```
   ES_HOSTS=http://localhost:9200
   ES_INDEX_PREFIX=rag
   CHUNK_SIZE=1000
   ```

## Importing Documents

```bash
# Import a single file
python src/import_utils.py path/to/document.pdf

# Import all documents in a directory
python src/import_utils.py path/to/documents/ --recursive

# Set custom chunk size
python src/import_utils.py path/to/documents/ --chunk-size 1500

# Use a custom Elasticsearch host
python src/import_utils.py path/to/documents/ --es-hosts http://elasticsearch:9200
```

Supported document formats:
- PDF (.pdf)
- Word Documents (.docx)
- Plain Text (.txt)

## Vector Search

The system includes support for vector embeddings in the `rag_chunks` index. To use this feature:

1. Generate vector embeddings for your text chunks using a model like Sentence Transformers
2. Update the chunks in Elasticsearch with the vector embeddings
3. Use Elasticsearch's vector search capabilities for semantic retrieval

Example vector search query:
```python
# Using the Elasticsearch Python client
from elasticsearch import Elasticsearch

es = Elasticsearch("http://localhost:9200")
query = {
    "knn": {
        "vector": {
            "vector": your_query_vector,
            "k": 10
        }
    }
}
results = es.search(index="rag_chunks", body=query)
```

## Project Structure

```
HCIRAG/
├── data/              # Directory for storing document files to be processed
├── models/            # Directory for storing vector embeddings and model files
├── notebooks/         # Jupyter notebooks for experimentation and demonstrations
├── src/               # Source code
│   ├── import_utils.py  # Utilities for importing documents into Elasticsearch
└── requirements.txt   # Project dependencies
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Elasticsearch](https://www.elastic.co/) for the powerful search and vector capabilities
- [PyPDF2](https://pypi.org/project/PyPDF2/) and [python-docx](https://python-docx.readthedocs.io/) for document processing
- [LangChain](https://www.langchain.com/) for RAG integration