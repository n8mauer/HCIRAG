<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>HCIRAG - MongoDB-based RAG System</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
            line-height: 1.6;
            color: #24292e;
            max-width: 980px;
            margin: 0 auto;
            padding: 20px;
        }
        h1, h2, h3 {
            border-bottom: 1px solid #eaecef;
            padding-bottom: 0.3em;
        }
        code, pre {
            font-family: SFMono-Regular, Consolas, "Liberation Mono", Menlo, monospace;
            background-color: #f6f8fa;
            border-radius: 3px;
        }
        pre {
            padding: 16px;
            overflow: auto;
            line-height: 1.45;
        }
        code {
            padding: 0.2em 0.4em;
        }
        .container {
            display: flex;
            flex-direction: column;
            gap: 20px;
        }
        .highlight pre {
            background-color: #f6f8fa;
            border-radius: 3px;
        }
        a {
            color: #0366d6;
            text-decoration: none;
        }
        a:hover {
            text-decoration: underline;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>HCIRAG</h1>
        
        <p>A Retrieval Augmented Generation (RAG) system using MongoDB for document storage and retrieval.</p>
        
        <h2>Overview</h2>
        <p>HCIRAG is a powerful RAG system designed to enhance your AI applications by providing efficient document storage, retrieval, and generation capabilities. By leveraging MongoDB's document storage capabilities, this system offers a robust solution for knowledge management and AI-powered content generation.</p>
        
        <h2>Features</h2>
        <ul>
            <li><strong>Document Storage</strong>: Efficiently store and manage documents in MongoDB</li>
            <li><strong>Semantic Retrieval</strong>: Find relevant documents based on semantic similarity</li>
            <li><strong>Content Generation</strong>: Generate high-quality content augmented with retrieved information</li>
            <li><strong>Scalable Architecture</strong>: Built to handle growing document collections</li>
        </ul>
        
        <h2>Getting Started</h2>
        
        <h3>Prerequisites</h3>
        <ul>
            <li>Python 3.8+</li>
            <li>MongoDB 4.4+</li>
            <li>pip (Python package manager)</li>
        </ul>
        
        <h3>Installation</h3>
        <div class="highlight">
            <pre><code>git clone https://github.com/yourusername/HCIRAG.git
cd HCIRAG
pip install -r requirements.txt</code></pre>
        </div>
        
        <h3>Configuration</h3>
        <ol>
            <li>
                <p>Create a <code>.env</code> file in the root directory with the following variables:</p>
                <div class="highlight">
                    <pre><code>MONGODB_URI=your_mongodb_connection_string
API_KEY=your_api_key_for_llm_provider</code></pre>
                </div>
            </li>
            <li>
                <p>Customize the configuration in <code>config.py</code> according to your needs.</p>
            </li>
        </ol>
        
        <h2>Usage</h2>
        
        <h3>Basic Example</h3>
        <div class="highlight">
            <pre><code>from hcirag import HCIRAG

# Initialize the system
rag = HCIRAG()

# Add documents
rag.add_documents("path/to/documents/")

# Query the system
results = rag.query("Your question here")
print(results.generated_response)</code></pre>
        </div>
        
        <h3>Advanced Usage</h3>
        <p>See the <code>examples/</code> directory for more advanced usage scenarios.</p>
        
        <h2>Project Structure</h2>
        <div class="highlight">
            <pre><code>HCIRAG/
├── src/                # Source code
│   ├── retriever/      # Document retrieval components
│   ├── generator/      # Content generation components
│   └── storage/        # MongoDB storage interfaces
├── examples/           # Example usage scripts
├── tests/              # Unit and integration tests
├── docs/               # Documentation
└── README.md           # This file</code></pre>
        </div>
        
        <h2>Contributing</h2>
        <p>Contributions are welcome! Please feel free to submit a Pull Request.</p>
        <ol>
            <li>Fork the repository</li>
            <li>Create your feature branch (<code>git checkout -b feature/amazing-feature</code>)</li>
            <li>Commit your changes (<code>git commit -m 'Add some amazing feature'</code>)</li>
            <li>Push to the branch (<code>git push origin feature/amazing-feature</code>)</li>
            <li>Open a Pull Request</li>
        </ol>
        
        <h2>License</h2>
        <p>This project is licensed under the MIT License - see the LICENSE file for details.</p>
        
        <h2>Acknowledgments</h2>
        <ul>
            <li><a href="https://www.mongodb.com/">MongoDB</a> for the powerful document database</li>
            <li>[LLM Provider] for the generation capabilities</li>
        </ul>
    </div>
</body>
</html>
