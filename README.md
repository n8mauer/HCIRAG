# HCIRAG

A Retrieval Augmented Generation (RAG) system using MongoDB for document storage and retrieval.

## Project Structure

```
HCIRAG/
   data/              # Directory for storing document files to be processed
   models/            # Directory for storing vector embeddings and model files
   notebooks/         # Jupyter notebooks for experimentation and demonstrations
   src/               # Source code
      import_utils.py  # Utilities for importing documents into MongoDB
   requirements.txt   # Project dependencies
```

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Install MongoDB:

- For Ubuntu/Debian:
  ```bash
  sudo apt-get update
  sudo apt-get install -y mongodb
  ```

- For macOS (using Homebrew):
  ```bash
  brew tap mongodb/brew
  brew install mongodb-community
  ```

- For Windows, download and install from the [MongoDB website](https://www.mongodb.com/try/download/community).

3. Start MongoDB service:

- For Ubuntu/Debian:
  ```bash
  sudo systemctl start mongodb
  ```

- For macOS:
  ```bash
  brew services start mongodb-community
  ```

- For Windows, MongoDB is installed as a service and should be running automatically.

4. Create a `.env` file from the example:

```bash
cp .env.example .env
```

5. Edit the `.env` file to configure your MongoDB connection and application settings.

## Importing Documents

Use the document import utility to import documents into the MongoDB database:

```bash
# Import a single file
python src/import_utils.py path/to/document.pdf

# Import all documents in a directory
python src/import_utils.py path/to/documents/ --recursive

# Set custom chunk size
python src/import_utils.py path/to/documents/ --chunk-size 1500
```

Supported file formats:
- PDF (.pdf)
- Word Documents (.docx)
- Plain Text (.txt)

## Usage

1. Import your documents as described above.
2. Use the MongoDB collections to retrieve documents and chunks for your RAG application.
3. Implement vector embeddings and similarity search for more advanced RAG capabilities.

## MongoDB Collections

The system uses two main collections:

1. `documents` - Stores metadata about each document.
2. `chunks` - Stores the actual text chunks extracted from documents, linked to their source document.

## License

[MIT License](LICENSE)