# HCIRAG

A Retrieval Augmented Generation (RAG) system using MongoDB for document storage and retrieval of Human Computer Interaction (HCI) required reading.

## Overview

HCIRAG is a powerful RAG system designed to enhance your AI applications by providing efficient document storage, retrieval, and generation capabilities. By leveraging MongoDB's document storage capabilities, this system offers a robust solution for knowledge management and AI-powered content generation.

## Features

- **Document Storage**: Efficiently store and manage documents in MongoDB
- **Semantic Retrieval**: Find relevant documents based on semantic similarity
- **Content Generation**: Generate high-quality content augmented with retrieved information
- **Scalable Architecture**: Built to handle growing document collections

## Getting Started

### Prerequisites

- Python 3.8+
- MongoDB 4.4+
- pip (Python package manager)

### Installation

```bash
git clone https://github.com/yourusername/HCIRAG.git
cd HCIRAG
pip install -r requirements.txt
```

### Configuration

1. Create a `.env` file in the root directory with the following variables:
   ```
   MONGODB_URI=your_mongodb_connection_string
   API_KEY=your_api_key_for_llm_provider
   ```

2. Customize the configuration in `config.py` according to your needs.

## Usage

### Basic Example

```python
from hcirag import HCIRAG

# Initialize the system
rag = HCIRAG()

# Add documents
rag.add_documents("path/to/documents/")

# Query the system
results = rag.query("Your question here")
print(results.generated_response)
```

### Advanced Usage

See the `examples/` directory for more advanced usage scenarios.

## Project Structure

```
HCIRAG/
├── src/                # Source code
│   ├── retriever/      # Document retrieval components
│   ├── generator/      # Content generation components
│   └── storage/        # MongoDB storage interfaces
├── examples/           # Example usage scripts
├── tests/              # Unit and integration tests
├── docs/               # Documentation
└── README.md           # This file
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

- [MongoDB](https://www.mongodb.com/) for the powerful document database
- [LLM Provider] for the generation capabilities
