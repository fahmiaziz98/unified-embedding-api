---
title: Api Embedding
emoji: 🐠
colorFrom: green
colorTo: purple
sdk: docker
pinned: false
---

# Unified Embedding API

**🧩 A self-hosted embedding service for dense, sparse, and reranking models with OpenAI-compatible API.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green.svg)](https://fastapi.tiangolo.com/)
[![Hugging Face](https://img.shields.io/badge/🤗-Hugging%20Face-yellow)](https://huggingface.co/spaces/fahmiaziz/api-embedding)

---

## Overview

The Unified Embedding API is a modular, self-hosted solution designed to simplify the development and management of embedding models for Retrieval-Augmented Generation (RAG) and semantic search applications. Built on FastAPI and Sentence Transformers, this API provides a unified interface for dense embeddings, sparse embeddings (SPLADE), and document reranking through CrossEncoder models.

**Key Differentiation:** Unlike traditional embedding services that require separate infrastructure for each model type, this API consolidates all embedding operations into a single, configurable endpoint with OpenAI-compatible responses.

### Project Motivation

During the development of RAG and agentic systems for production environments and portfolio projects, several operational challenges emerged:

1. **Development Environment Overhead:** Each experiment required setting up isolated environments with PyTorch, Transformers, and associated dependencies (often 5-10GB per environment)
2. **Model Experimentation Costs:** Testing different models for optimal precision, MRR, and recall metrics necessitated downloading multiple model versions, consuming significant disk space and compute resources
3. **Hardware Limitations:** Running models locally on CPU-only machines frequently resulted in thermal throttling and system instability

**Solution Approach:** After evaluating Hugging Face's Text Embeddings Inference (TEI), the need for a more flexible, configuration-driven solution became apparent. This project addresses these challenges by:

- Providing a single API endpoint that can serve multiple model types
- Enabling model switching through configuration files without code changes
- Leveraging Hugging Face Spaces for free, serverless hosting
- Maintaining compatibility with OpenAI's client libraries for seamless integration

---

## Technical Motivation

### Architecture Decisions

#### 1. Framework Selection: SentenceTransformers + FastAPI

**SentenceTransformers** was chosen as the core embedding library for several technical reasons:

- **Unified Model Interface:** Provides consistent APIs across diverse model architectures (BERT, RoBERTa, SPLADE, CrossEncoders)
- **Model Ecosystem:** Direct compatibility with 5,000+ pre-trained models on Hugging Face Hub

**FastAPI** serves as the web framework due to:

- **Async-First Architecture:** Non-blocking I/O operations critical for handling concurrent embedding requests
- **Automatic API Documentation:** OpenAPI/Swagger generation reduces documentation overhead
- **Type Safety:** Pydantic integration ensures request validation at the schema level

#### 2. Hosting Strategy: Hugging Face Spaces

Deploying on Hugging Face Spaces provides several operational advantages:

- Zero infrastructure cost for CPU-based workloads (2vCPU, 16GB RAM)
- Eliminates need for dedicated VPS or cloud compute instances
- No egress fees for model weight downloads from HF Hub
- Built-in CI/CD through git-based deployments
- Easy transition to paid GPU instances for larger models
- Native support for Docker-based deployments

---

## Features

### Core Capabilities

- **Multi-Model Support:** Serve dense embeddings (transformers), sparse embeddings (SPLADE), and reranking models (CrossEncoders) from a single API
- **OpenAI Compatibility:** Drop-in replacement for OpenAI's embedding API with client library support
- **Configuration-Driven:** Switch models through YAML configuration without code modifications
- **Batch Processing:** Automatic optimization for single and batch requests
- **Type Safety:** Full Pydantic validation with OpenAPI schema generation
- **Async Operations:** Non-blocking request handling with FastAPI's async/await

---

## Architecture

### System Components

```
┌─────────────────────────────────────────────────────────┐
│                     FastAPI Server                      │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐         │
│  │ Embeddings │  │  Reranking │  │   Models   │         │
│  │  Endpoint  │  │  Endpoint  │  │  Endpoint  │         │
│  └────────────┘  └────────────┘  └────────────┘         │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│                   Model Manager                         │
│  • Configuration Loading                                │
│  • Model Lifecycle Management                           │
│  • Thread-Safe Model Access                             │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│              Embedding Implementations                  │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐     │
│  │    Dense     │ │    Sparse    │ │   Reranking  │     │
│  │(Transformer) │ │   (SPLADE)   │ │(CrossEncoder)│     │
│  └──────────────┘ └──────────────┘ └──────────────┘     │
└─────────────────────────────────────────────────────────┘
```

### Project Structure

```
unified-embedding-api/
├── src/
│   ├── api/                    # API layer
│   │   ├── dependencies.py     # Dependency injection
│   │   └── routes/
│   │       ├── embeddings.py   # Dense/sparse endpoints
│   │       ├── model_list.py   # Model management
│   │       └── health.py       # Health checks
│   │       └── rerank.py       # Reranking endpoint
│   ├── core/                   # Business logic
│   │   ├── base.py             # Abstract base classes
│   │   ├── config.py           # Configuration models
│   │   ├── exceptions.py       # Custom exceptions     
│   │   └── manager.py          # Model lifecycle management
│   ├── models/                 # Domain models
│   │   ├── embeddings/
│   │   │   ├── dense.py        # Dense embedding implementation
│   │   │   ├── sparse.py       # Sparse embedding implementation
│   │   │   └── rank.py         # Reranking implementation
│   │   └── schemas/
│   │       ├── common.py       # Shared schemas
│   │       ├── requests.py     # Request models
│   │       └── responses.py    # Response models
│   ├── config/
│   │   ├── settings.py         # Application settings
│   │   └── models.yaml         # Model configuration
│   └── utils/
│       ├── logger.py           # Logging configuration
│       └── validators.py       # Validation kwrags, token etc
├── app.py                      # Application entry point
├── requirements.txt            # Development dependencies
└── Dockerfile                  # Container definition
```

---

## Quick Start

### Deployment on Hugging Face Spaces

**Prerequisites:**
- Hugging Face account
- Git installed locally

**Steps:**

1. **Duplicate Space**
   - Navigate to [fahmiaziz/api-embedding](https://huggingface.co/spaces/fahmiaziz/api-embedding)
   - Click the three-dot menu → "Duplicate this Space"

2. **Configure Environment**
   - In Space settings, add `HF_TOKEN` as a repository secret (for private model access)
   - Ensure Space visibility is set to "Public"

3. **Clone Repository**
   ```bash
   git clone https://huggingface.co/spaces/YOUR_USERNAME/api-embedding
   cd api-embedding
   ```

4. **Configure Models**
   Edit `src/config/models.yaml`:
   ```yaml
   models:
     custom-model:
       name: "organization/model-name"
       type: "embeddings"  # Options: embeddings, sparse-embeddings, rerank
   ```

5. **Deploy Changes**
   ```bash
   git add src/config/models.yaml
   git commit -m "Configure custom models"
   git push
   ```

6. **Access API**
   - Base URL: `https://YOUR_USERNAME-api-embedding.hf.space`
   - Documentation: `https://YOUR_USERNAME-api-embedding.hf.space/docs`

### Local Development (NOT RECOMMENDED)

**System Requirements:**
- Python 3.10+
- 8GB RAM minimum
- 10GB++ disk space 

**Setup:**

```bash
# Clone repository
git clone https://github.com/fahmiaziz98/unified-embedding-api.git
cd unified-embedding-api

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start server
python app.py
```

Server will be available at `http://localhost:7860`

### Docker Deployment

```bash
# Build image
docker build -t unified-embedding-api .

# Run container
docker run -p 7860:7860 unified-embedding-api

```

---

## Usage

### Native API (requests)

```python
import requests

BASE_URL = "https://fahmiaziz-api-embedding.hf.space/api/v1"

# Generate embeddings
response = requests.post(
    f"{BASE_URL}/embeddings",
    json={
        "input": "Natural language processing",
        "model": "qwen3-0.6b"
    }
)

data = response.json()
embedding = data["data"][0]["embedding"]
print(f"Embedding dimensions: {len(embedding)}")
```

### OpenAI Client Integration

The API implements OpenAI's embedding API specification, enabling direct integration with OpenAI's Python client:

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://fahmiaziz-api-embedding.hf.space/api/v1",
    api_key="not-required"  # Placeholder required by client
)

# Single text embedding
response = client.embeddings.create(
    input="Text to embed",
    model="qwen3-0.6b"
)

embedding_vector = response.data[0].embedding
```

**Async Operations:**

```python
from openai import AsyncOpenAI

client = AsyncOpenAI(
    base_url="https://fahmiaziz-api-embedding.hf.space/api/v1",
    api_key="not-required"
)

async def generate_embeddings(texts: list[str]):
    response = await client.embeddings.create(
        input=texts,
        model="qwen3-0.6b"
    )
    return [item.embedding for item in response.data]

# Usage in async context
embeddings = await generate_embeddings(["text1", "text2"])
```

### Document Reranking

```python
import requests

response = requests.post(
    f"{BASE_URL}/rerank",
    json={
        "query": "machine learning frameworks",
        "documents": [
            "TensorFlow is a comprehensive ML platform",
            "React is a JavaScript UI library",
            "PyTorch provides flexible neural networks"
        ],
        "model": "bge-v2-m3",
        "top_k": 2
    }
)

results = response.json()["results"]
for result in results:
    print(f"Score: {result['score']:.3f} - {result['text']}")
```

---

## API Reference

### Endpoints

| Endpoint | Method | Description | OpenAI Compatible |
|----------|--------|-------------|-------------------|
| `/api/v1/embeddings` | POST | Generate embeddings | Yes |
| `/api/v1/embed_sparse` | POST | Generate sparse embeddings | No |
| `/api/v1/rerank` | POST | Rerank documents | No |
| `/api/v1/models` | GET | List available models | Partial |
| `/health` | GET | Health check | No |

**Detailed API documentation:** [docs/API.md](docs/API.md)

### Request Format

**Embeddings (OpenAI-compatible):**
```json
{
  "input": "text" | ["text1", "text2"],
  "model": "model-identifier",
  "encoding_format": "float"
}
```

**Sparse Embeddings:**
```json
{
  "input": "text" | ["text1", "text2"],
  "model": "splade-model-id"
}
```

**Reranking:**
```json
{
  "query": "search query",
  "documents": ["doc1", "doc2"],
  "model": "reranker-id",
  "top_k": 10
}
```

### Response Format

**Standard Embedding Response:**
```json
{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "embedding": [0.123, -0.456, ...],
      "index": 0
    }
  ],
  "model": "qwen3-0.6b",
  "usage": {
    "prompt_tokens": 5,
    "total_tokens": 5
  }
}
```

---

## Configuration

### Model Configuration
Default configuration is optimized for **CPU 2vCPU / 16GB RAM**. See [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard) for model recommendations and memory usage reference.

Edit `src/config/models.yaml` to add or modify models:

```yaml
models:
  # Dense embedding model
  custom-dense:
    name: "sentence-transformers/all-MiniLM-L6-v2"
    type: "embeddings"

  # Sparse embedding model
  custom-sparse:
    name: "prithivida/Splade_PP_en_v1"
    type: "sparse-embeddings"

  # Reranking model
  custom-reranker:
    name: "BAAI/bge-reranker-base"
    type: "rerank"
```

**Model Type Reference:**

| Type | Description | Use Case |
|------|-------------|----------|
| `embeddings` | Dense vector embeddings | Semantic search, similarity |
| `sparse-embeddings` | Sparse vectors (SPLADE) | Keyword + semantic hybrid |
| `rerank` | CrossEncoder scoring | Precision reranking |

⚠️ If you plan to use larger models like `Qwen2-embedding-8B`, please upgrade your Space.

### Application Settings

Configure through `src/config/settings.py` file:

```bash
# Application
APP_NAME="Unified Embedding API"
VERSION="3.0.0"

# Server
HOST=0.0.0.0
PORT=7860  # don't change port
WORKERS=1

# Models
MODEL_CONFIG_PATH=src/config/models.yaml
PRELOAD_MODELS=true
DEVICE=cpu

# Logging
LOG_LEVEL=INFO
```

---

## Performance Optimization

### Recommended Practices

1. **Batch Processing**
   - Always send multiple texts in a single request when possible
   - Batch size of 16-32 provides optimal throughput/latency balance

2. **Normalization**
   - Enable `normalize_embeddings` for cosine similarity operations
   - Reduces downstream computation in vector databases

3. **Model Selection**
   - Dense models: Best for semantic similarity
   - Sparse models: Better for keyword matching + semantics
   - Reranking: Use as second-stage after initial retrieval

---

## Migration from OpenAI

Replace OpenAI embedding calls with minimal code changes:

**Before (OpenAI):**
```python
from openai import OpenAI
client = OpenAI(api_key="sk-...")

response = client.embeddings.create(
    input="Hello world",
    model="text-embedding-3-small"
)
```

**After (Self-hosted):**
```python
from openai import OpenAI
client = OpenAI(
    base_url="https://your-space.hf.space/api/v1",
    api_key="not-required"
)

response = client.embeddings.create(
    input="Hello world",
    model="qwen3-0.6b"  # Your configured model
)
```

**Compatibility Matrix:**

| Feature | Supported | Notes |
|---------|-----------|-------|
| `input` (string) | ✓ | Converted to list internally |
| `input` (list) | ✓ | Batch processing |
| `model` parameter | ✓ | Use configured model IDs |
| `encoding_format` | Partial | Always returns float |
| `dimensions` | ✗ | Returns model's native dimensions |
| `user` parameter | ✗ | Ignored |

---
## ⚠️ **Note:** This is a development API.  
For production deployment, host it on cloud platforms such as **Hugging Face TEI**, **AWS**, **GCP**, or any cloud provider of your choice.

---

## Contributing

Contributions are welcome. Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request
---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## References

- **Sentence Transformers Documentation:** https://www.sbert.net/
- **FastAPI Documentation:** https://fastapi.tiangolo.com/
- **OpenAI API Specification:** https://platform.openai.com/docs/api-reference/embeddings
- **MTEB Benchmark:** https://huggingface.co/spaces/mteb/leaderboard
- **Hugging Face Spaces:** https://huggingface.co/docs/hub/spaces

---

## Support

- **Issues:** [GitHub Issues](https://github.com/fahmiaziz98/unified-embedding-api/issues)
- **Discussions:** [GitHub Discussions](https://github.com/fahmiaziz98/unified-embedding-api/discussions)
- **Live Demo:** [Hugging Face Space](https://huggingface.co/spaces/fahmiaziz/api-embedding)

---

**Maintained by:** [Fahmi Aziz](https://github.com/fahmiaziz98)  
**Project Status:** Active Development

> ✨ "Unify your embeddings. Simplify your AI stack."