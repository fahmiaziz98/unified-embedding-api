# 📖 Unified Embedding API Documentation

Complete API reference for the Unified Embedding API v3.0.0.

**Features:** Dense Embeddings, Sparse Embeddings, and Document Reranking

---

## 🌐 Base URL

```
https://fahmiaziz-api-embedding.hf.space
```

For local development:
```
http://localhost:7860
```

---

## 🔑 Authentication

**Currently no authentication required.** 

---

## 📊 Endpoints Overview

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/embeddings/embed` | POST | Generate document embeddings |
| `/api/v1/embeddings/query` | POST | Generate query embeddings |
| `/api/v1/rerank` | POST | Rerank documents by relevance |
| `/api/v1/models` | GET | List available models |
| `/api/v1/models/{model_id}` | GET | Get model information |
| `/health` | GET | Health check |
| `/` | GET | API information |

---

## 🚀 Embedding Endpoints

### 1. Generate Document Embeddings

**`POST /api/v1/embeddings/embed`**

Generate embeddings for document texts. Supports both single and batch processing.

#### Request Body

```json
{
  "texts": ["string"],           // Required: List of texts (1-100 items)
  "model_id": "string",          // Required: Model identifier
  "prompt": "string",            // Optional: Instruction prompt
  "options": {                   // Optional: Embedding parameters
    "normalize_embeddings": true,
    "batch_size": 32,
    "max_length": 512,
    "show_progress_bar": false
  }
}
```

#### Parameters

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `texts` | array[string] | ✅ Yes | List of texts to embed (min: 1, max: 100) |
| `model_id` | string | ✅ Yes | Model identifier (e.g., "qwen3-0.6b") |
| `prompt` | string | ❌ No | Instruction prompt for the model |
| `options` | object | ❌ No | Additional embedding parameters |

#### Options Parameters

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `normalize_embeddings` | boolean | false | L2 normalize output embeddings |
| `batch_size` | integer | 32 | Processing batch size (1-256) |
| `max_length` | integer | 512 | Maximum sequence length (1-8192) |
| `show_progress_bar` | boolean | false | Display progress during encoding |
| `precision` | string | float32 | Precision ("float32", "int8", "binary") |

#### Response - Single Text (Dense)

```json
{
  "embedding": [0.123, -0.456, 0.789, ...],
  "dimension": 768,
  "model_id": "qwen3-0.6b",
  "processing_time": 0.0523
}
```

#### Response - Batch (Dense)

```json
{
  "embeddings": [
    [0.123, -0.456, ...],
    [0.234, 0.567, ...],
    [0.345, -0.678, ...]
  ],
  "dimension": 768,
  "count": 3,
  "model_id": "qwen3-0.6b",
  "processing_time": 0.1245
}
```

#### Response - Single Text (Sparse)

```json
{
  "sparse_embedding": {
    "text": "Hello world",
    "indices": [10, 25, 42, 100],
    "values": [0.85, 0.62, 0.91, 0.73]
  },
  "model_id": "splade-pp-v2",
  "processing_time": 0.0421
}
```

#### Response - Batch (Sparse)

```json
{
  "embeddings": [
    {
      "text": "First doc",
      "indices": [10, 25, 42],
      "values": [0.85, 0.62, 0.91]
    },
    {
      "text": "Second doc",
      "indices": [15, 30, 50],
      "values": [0.73, 0.88, 0.65]
    }
  ],
  "count": 2,
  "model_id": "splade-pp-v2",
  "processing_time": 0.0892
}
```

#### Examples

**Single Text (Dense Model):**
```bash
curl -X 'POST' \
  'https://fahmiaziz-api-embedding.hf.space/api/v1/embeddings/embed' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "texts": ["What is artificial intelligence?"],
  "model_id": "qwen3-0.6b"
}'
```

**Single Text (Sparse Model):**
```bash
curl -X 'POST' \
  'https://fahmiaziz-api-embedding.hf.space/api/v1/embeddings/embed' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "texts": ["Hello world"],
  "model_id": "splade-pp-v2"
}'
```

**Batch (with Options):**
```bash
curl -X 'POST' \
  'https://fahmiaziz-api-embedding.hf.space/api/v1/embeddings/embed' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "texts": [
    "First document to embed",
    "Second document to embed",
    "Third document to embed"
  ],
  "model_id": "qwen3-0.6b",
  "options": {
    "normalize_embeddings": true,
    "batch_size": 32
  }
}'
```

**Python Example:**
```python
import requests

url = "https://fahmiaziz-api-embedding.hf.space/api/v1/embeddings/embed"

payload = {
    "texts": ["Hello world"],
    "model_id": "qwen3-0.6b"
}

response = requests.post(url, json=payload)
data = response.json()

print(f"Embedding dimension: {data['dimension']}")
print(f"Processing time: {data['processing_time']:.3f}s")
```

---

### 2. Generate Query Embeddings

**`POST /api/v1/embeddings/query`**

Generate embeddings optimized for search queries. Some models differentiate between query and document embeddings.

#### Request Body

Same as `/embed` endpoint.

```json
{
  "texts": ["string"],
  "model_id": "string",
  "prompt": "string",
  "options": {}
}
```

#### Response

Same format as `/embed` endpoint.

#### Examples

**Single Query:**
```bash
curl -X 'POST' \
  'https://fahmiaziz-api-embedding.hf.space/api/v1/embeddings/query' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "texts": ["What is machine learning?"],
  "model_id": "qwen3-0.6b",
  "prompt": "Represent this query for retrieval",
  "options": {
    "normalize_embeddings": true
  }
}'
```

**Batch Queries:**
```bash
curl -X 'POST' \
  'https://fahmiaziz-api-embedding.hf.space/api/v1/embeddings/query' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "texts": [
    "First query",
    "Second query",
    "Third query"
  ],
  "model_id": "qwen3-0.6b"
}'
```

**Python Example:**
```python
import requests

url = "https://fahmiaziz-api-embedding.hf.space/api/v1/embeddings/query"

payload = {
    "texts": ["What is AI?"],
    "model_id": "qwen3-0.6b",
    "options": {
        "normalize_embeddings": True
    }
}

response = requests.post(url, json=payload)
embedding = response.json()["embedding"]
```

---

### 3. Rerank Documents

**`POST /api/v1/rerank`**

Rerank documents based on their relevance to a query using CrossEncoder models.

#### Request Body

```json
{
  "query": "string",             // Required: Search query
  "documents": ["string"],       // Required: List of documents (min: 1)
  "model_id": "string",          // Required: Reranking model identifier
  "top_k": integer,              // Required: Number of top results to return
}
```

#### Parameters

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `query` | string | ✅ Yes | Search query text |
| `documents` | array[string] | ✅ Yes | List of documents to rerank (min: 1) |
| `model_id` | string | ✅ Yes | Reranking model identifier |
| `top_k` | integer | ✅ Yes | Maximum number of results to return |

#### Response

```json
{
  "model_id": "jina-reranker-v3",
  "processing_time": 0.56,
  "query": "Python for data science",
  "results": [
    {
      "index": 0,
      "score": 0.95,
      "text": "Python is excellent for data science"
    },
    {
      "index": 2,
      "score": 0.73,
      "text": "R is also used in data science"
    }
  ]
}
```

#### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `model_id` | string | Model identifier used |
| `processing_time` | float | Processing time in seconds |
| `query` | string | Original search query |
| `results` | array | Reranked documents with scores |
| `results[].index` | integer | Original index in input documents |
| `results[].score` | float | Relevance score (0-1, normalized) |
| `results[].text` | string | Document text |

#### Examples

**Basic Reranking:**
```bash
curl -X 'POST' \
  'https://fahmiaziz-api-embedding.hf.space/api/v1/rerank' \
  -H 'Content-Type: application/json' \
  -d '{
  "query": "Python for data science",
  "documents": [
    "Python is great for data science",
    "Java is used for enterprise applications",
    "R is also used in data science",
    "JavaScript is for web development"
  ],
  "model_id": "jina-reranker-v3",
  "top_k": 2
}'
```


**Python Example:**
```python
import requests

url = "https://fahmiaziz-api-embedding.hf.space/api/v1/rerank"

payload = {
    "query": "best programming language for beginners",
    "documents": [
        "Python is beginner-friendly with simple syntax",
        "C++ is powerful but complex for beginners",
        "JavaScript is essential for web development",
        "Rust offers memory safety but steep learning curve"
    ],
    "model_id": "jina-reranker-v3",
    "top_k": 2
}

response = requests.post(url, json=payload)
data = response.json()

print(f"Top result: {data['results'][0]['text']}")
print(f"Score: {data['results'][0]['score']:.3f}")
```

**JavaScript Example:**
```javascript
const url = "https://fahmiaziz-api-embedding.hf.space/api/v1/rerank";

const response = await fetch(url, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    query: "AI applications",
    documents: [
      "Computer vision for image recognition",
      "Recipe for chocolate cake",
      "Natural language processing for chatbots",
      "Travel guide to Paris"
    ],
    model_id: "jina-reranker-v3",
    top_k: 2
  })
});

const { results } = await response.json();
console.log("Top results:", results);
```

---

## 🤖 Model Management

### 3. List Available Models

**`GET /api/v1/models`**

Get a list of all available embedding models.

#### Response

```json
{
  "models": [
    {
      "id": "qwen3-0.6b",
      "name": "Qwen/Qwen3-Embedding-0.6B",
      "type": "embeddings",
      "loaded": true,
      "repository": "https://huggingface.co/Qwen/Qwen3-Embedding-0.6B"
    },
    {
      "id": "splade-pp-v2",
      "name": "prithivida/Splade_PP_en_v2",
      "type": "sparse-embeddings",
      "loaded": true,
      "repository": "https://huggingface.co/prithivida/Splade_PP_en_v2"
    }
  ],
  "total": 2
}
```

#### Example

```bash
curl -X 'GET' \
  'https://fahmiaziz-api-embedding.hf.space/api/v1/models' \
  -H 'accept: application/json'
```

---

### 4. Get Model Information

**`GET /api/v1/models/{model_id}`**

Get detailed information about a specific model.

#### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `model_id` | string | ✅ Yes | Model identifier |

#### Response

```json
{
  "id": "qwen3-0.6b",
  "name": "Qwen/Qwen3-Embedding-0.6B",
  "type": "embeddings",
  "loaded": true,
  "repository": "https://huggingface.co/Qwen/Qwen3-Embedding-0.6B"
}
```

#### Example

```bash
curl -X 'GET' \
  'https://fahmiaziz-api-embedding.hf.space/api/v1/models/qwen3-0.6b' \
  -H 'accept: application/json'
```

---

## 🏥 System Endpoints

### 5. Health Check

**`GET /health`**

Check API health status.

#### Response

```json
{
  "status": "ok",
  "total_models": 2,
  "loaded_models": 2,
  "startup_complete": true
}
```

#### Example

```bash
curl -X 'GET' \
  'https://fahmiaziz-api-embedding.hf.space/health' \
  -H 'accept: application/json'
```

---

### 6. API Information

**`GET /`**

Get basic API information.

#### Response

```json
{
  "message": "Unified Embedding API - Dense & Sparse Embeddings",
  "version": "3.0.0",
  "docs_url": "/docs"
}
```

---

## ❌ Error Responses

All errors follow this format:

```json
{
  "detail": "Error message description"
}
```

### HTTP Status Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 400 | Bad Request - Invalid input |
| 404 | Not Found - Model not found |
| 422 | Unprocessable Entity - Validation error |
| 500 | Internal Server Error |
| 503 | Service Unavailable - Server not ready |

### Common Errors

**Model Not Found (404):**
```json
{
  "detail": "Model 'unknown-model' not found in configuration"
}
```

**Validation Error (422):**
```json
{
  "detail": [
    {
      "loc": ["body", "texts"],
      "msg": "texts list cannot be empty",
      "type": "value_error"
    }
  ]
}
```

**Batch Too Large (422):**
```json
{
  "detail": "Batch size (150) exceeds maximum (100)"
}
```

---

## 📦 Available Models

### Dense Embedding Models

| Model ID | Name | Dimension | Description |
|----------|------|-----------|-------------|
| `qwen3-0.6b` | Qwen/Qwen3-Embedding-0.6B | 768 | Efficient multilingual embeddings |

### Sparse Embedding Models

| Model ID | Name | Type | Description |
|----------|------|------|-------------|
| `splade-pp-v2` | prithivida/Splade_PP_en_v2 | Sparse | SPLADE++ English v2 |

### Reranking Models

| Model ID | Name | Type | Description |
|----------|------|------|-------------|
| `jina-reranker-v3` | jinaai/jina-reranker-v3-base-en | CrossEncoder | High-quality reranking (English) |
| `bge-v2-m3` | BAAI/bge-reranker-v2-m3 | CrossEncoder | Multilingual reranking |

---

## 🔧 Rate Limits

**Current Limits:**
- Max text length: 8,192 characters
- Max batch size: 100 texts per request
- No rate limiting (subject to server resources)

---

## 💡 Best Practices

### 1. Batch Processing
Always batch multiple texts together for better performance:
```python
# ❌ Bad - Multiple requests
for text in texts:
    response = requests.post(url, json={"texts": [text], ...})

# ✅ Good - Single batch request
response = requests.post(url, json={"texts": texts, ...})
```

### 2. Normalize Embeddings for Similarity
For cosine similarity, always normalize:
```python
payload = {
    "texts": ["text"],
    "model_id": "qwen3-0.6b",
    "options": {"normalize_embeddings": True}
}
```

### 3. Model Selection
- **Dense models** (qwen3-0.6b): Best for semantic similarity
- **Sparse models** (splade-pp-v2): Best for keyword matching + semantic
- **Rerank models** (jina-reranker-v3): Best for re-scoring top candidates

### 4. Two-Stage Retrieval (Recommended for RAG)
```python
# Stage 1: Fast retrieval with embeddings (top 100)
query_embedding = embed_query(query)
candidates = vector_search(query_embedding, top_k=100)

# Stage 2: Precise reranking (top 10)
reranked = rerank(
    query=query,
    documents=[c["text"] for c in candidates],
    model_id="jina-reranker-v3",
    top_k=10
)
```

### 5. Error Handling
Always handle errors gracefully:
```python
try:
    response = requests.post(url, json=payload)
    response.raise_for_status()
    data = response.json()
except requests.exceptions.HTTPError as e:
    print(f"HTTP error: {e}")
except requests.exceptions.RequestException as e:
    print(f"Request failed: {e}")
```

---

## 🐛 Troubleshooting

### Empty Response
- Check `texts` field is not empty
- Validate `model_id` exists

### Slow Performance
- Use batch requests instead of multiple single requests
- Reduce `batch_size` in options if memory issues
- Check model is preloaded (first request is slower)

### Connection Errors
- Verify base URL is correct
- Check network connectivity
- Ensure server is running (`/health` endpoint)

---

## 📞 Support

- **Documentation**: [GitHub README](https://github.com/fahmiaziz/unified-embedding-api)
- **Issues**: [GitHub Issues](https://github.com/fahmiaziz/unified-embedding-api/issues)
- **Hugging Face Space**: [fahmiaziz/api-embedding](https://huggingface.co/spaces/fahmiaziz/api-embedding)

---

## 🔄 Changelog

### v3.0.0 (Current)
- ✨ Added reranking endpoint (`/api/v1/rerank`)
- ✨ Support for CrossEncoder models
- ✨ Unified batch-only response format
- ✨ Flexible kwargs support
- ✨ In-memory caching
- ✨ Improved error handling
- ✨ Comprehensive documentation
- 🐛 Fixed type hint errors in RerankModel
- 🐛 Fixed duplicate parameter errors in rerank endpoint

---

**Last Updated**: 2025-11-02