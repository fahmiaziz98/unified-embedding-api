---
title: Api Embedding
emoji: 🐠
colorFrom: green
colorTo: purple
sdk: docker
pinned: false
---

# 🧠 Unified Embedding API

> 🧩 Unified API for all your Embedding, Sparse & Reranking Models — plug and play with any model from Hugging Face or your own fine-tuned versions.

---

## 🚀 Overview

**Unified Embedding API** is a modular and open-source **RAG-ready API** built for developers who want a simple, unified way to access **dense**, **sparse**, and **reranking** models.

It's designed for **vector search**, **semantic retrieval**, and **AI-powered pipelines** — all controlled from a single `config.yaml` file.

⚠️ **Note:** This is a development API.  
For production deployment, host it on cloud platforms such as **Hugging Face TEI**, **AWS**, **GCP**, or any cloud provider of your choice.

---

## 🧩 Features

- 🧠 **Unified Interface** — One API to handle dense, sparse, and reranking models
- ⚡ **Batch Processing** — Automatic single/batch detection
- 🔧 **Flexible Parameters** — Full control via kwargs and options
- 🔌 **OpenAI Compatible** — Works with OpenAI client libraries
- 📈 **RAG Support** — Perfect base for Retrieval-Augmented Generation systems
- ⚡ **Fast & Lightweight** — Powered by FastAPI and optimized with async processing
- 🧰 **Extendable** — Switch models instantly via `config.yaml` and add your own models effortlessly

---

## 📁 Project Structure

```
unified-embedding-api/
├── src/
│   ├── api/
│   │   ├── dependencies.py
│   │   └── routes/
│   │       ├── embeddings.py  # endpoint sparse & dense   
│   │       ├── models.py
│   │       ├── health.py
│   │       └── rerank.py      # endpoint reranking
│   ├── core/
│   │   ├── base.py
│   │   ├── config.py
│   │   ├── exceptions.py
│   │   └── manager.py
│   ├── models/
│   │   ├── embeddings/
│   │   │   ├── dense.py       # dense model
│   │   │   ├── sparse.py      # sparse model
│   │   │   └── rank.py        # reranking model
│   │   └── schemas/
│   │       ├── common.py
│   │       ├── requests.py       
│   │       └── responses.py
│   ├── config/
│   │   ├── settings.py
│   │   └── models.yaml        # add/change models here
│   └── utils/
│       ├── logger.py
│       └── validators.py
│
├── app.py                         
├── requirements.txt
├── LICENSE
├── Dockerfile
└── README.md
```

---

## 🧩 Model Selection

Default configuration is optimized for **CPU 2vCPU / 16GB RAM**. See [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard) for model recommendations and memory usage reference.

**Add More Models:** Edit `src/config/models.yaml`

```yaml
models:
  your-model-name:
    name: "org/model-name"
    type: "embeddings"  # or "sparse-embeddings" or "rerank"
```

⚠️ If you plan to use larger models like `Qwen2-embedding-8B`, please upgrade your Space.

---

## ☁️ How to Deploy (Free 🚀)

Deploy your **Custom Embedding API** on **Hugging Face Spaces** — free, fast, and serverless.

### **1️⃣ Deploy on Hugging Face Spaces (Free!)**

1. **Duplicate this Space:**  
   👉 [fahmiaziz/api-embedding](https://huggingface.co/spaces/fahmiaziz/api-embedding)  
   Click **⋯** (three dots) → **Duplicate this Space**

2. **Add HF_TOKEN environment variable**. Make sure your space is public

3. **Clone your Space locally:**  
   Click **⋯** → **Clone repository**
   ```bash
   git clone https://huggingface.co/spaces/YOUR_USERNAME/api-embedding
   cd api-embedding
   ```

4. **Edit `src/config/models.yaml`** to customize models:
   ```yaml
   models:
     your-model:
       name: "org/model-name"
       type: "embeddings"  # or "sparse-embeddings" or "rerank"
   ```

5. **Commit and push changes:**
   ```bash
   git add src/config/models.yaml
   git commit -m "Update models configuration"
   git push
   ```

6. **Access your API:**  
   Click **⋯** → **Embed this Space** → copy **Direct URL**
   ```
   https://YOUR_USERNAME-api-embedding.hf.space
   https://YOUR_USERNAME-api-embedding.hf.space/docs  # Interactive docs
   ```

That's it! You now have a live embedding API endpoint powered by your models.

### **2️⃣ Run Locally (NOT RECOMMENDED)**

```bash
# Clone repository
git clone https://github.com/fahmiaziz98/unified-embedding-api.git
cd unified-embedding-api

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run server
python app.py
```

API available at: `http://localhost:7860`

### **3️⃣ Run with Docker**

```bash
# Build and run
docker-compose up --build

# Or with Docker only
docker build -t embedding-api .
docker run -p 7860:7860 embedding-api
```

---

## 📖 Usage Examples

### **Python with Native API**

```python
import requests

base_url = "https://fahmiaziz-api-embedding.hf.space/api/v1"

# Single embedding
response = requests.post(f"{base_url}/embeddings", json={
    "input": "What is artificial intelligence?",
    "model": "qwen3-0.6b"
})
embeddings = response.json()["data"]

# Batch embeddings with options
response = requests.post(f"{base_url}/embeddings", json={
    "input": ["First document", "Second document", "Third document"],
    "model": "qwen3-0.6b",
    "options": {
        "normalize_embeddings": True
    }
})
batch_embeddings = response.json()["data"]
```

### **cURL**

```bash
# Dense embeddings
curl -X POST "https://fahmiaziz-api-embedding.hf.space/api/v1/embeddings" \
  -H "Content-Type: application/json" \
  -d '{
    "input": ["Hello world"],
    "model": "qwen3-0.6b"
  }'

# Sparse embeddings
curl -X POST "https://fahmiaziz-api-embedding.hf.space/api/v1/embed_sparse" \
  -H "Content-Type: application/json" \
  -d '{
    "input": ["First doc", "Second doc", "Third doc"],
    "model": "splade-pp-v2"
  }'

# Reranking
curl -X POST "https://fahmiaziz-api-embedding.hf.space/api/v1/rerank" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Python for data science",
    "documents": [
      "Python is great for data science",
      "Java is used for enterprise apps",
      "R is for statistical analysis"
    ],
    "model": "bge-v2-m3",
    "top_k": 2
  }'
```

### **JavaScript/TypeScript**

```typescript
const baseUrl = "https://fahmiaziz-api-embedding.hf.space/api/v1";

// Using fetch
const response = await fetch(`${baseUrl}/embeddings`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    texts: ["Hello world"],
    model_id: "qwen3-0.6b",
  }),
});

const { embeddings } = await response.json();
console.log(embeddings);
```

---

## 📊 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/embeddings` | POST | Generate embeddings (OpenAI compatible) |
| `/api/v1/embed_sparse` | POST | Generate sparse embeddings |
| `/api/v1/rerank` | POST | Rerank documents by relevance |
| `/api/v1/models` | GET | List available models |
| `/api/v1/models/{model_id}` | GET | Get model information |
| `/health` | GET | Health check |
| `/` | GET | API information |
| `/docs` | GET | Interactive API documentation |

---

## 🔌 OpenAI Client Compatibility

This API is **fully compatible** with OpenAI's client libraries, making it a drop-in replacement for OpenAI's embedding API.

### **Why use OpenAI client?**

✅ **Familiar API** — Same interface as OpenAI  
✅ **Type Safety** — Full type hints and IDE support  
✅ **Error Handling** — Built-in retry logic and error handling  
✅ **Async Support** — Native async/await support  
✅ **Easy Migration** — Switch between OpenAI and self-hosted seamlessly

### **Supported Features**

| Feature | Supported | Notes |
|---------|-----------|-------|
| `embeddings.create()` | ✅ Yes | Single and batch inputs |
| `input` as string | ✅ Yes | Auto-converted to list |
| `input` as list | ✅ Yes | Batch processing |
| `model` parameter | ✅ Yes | Use your model IDs |
| `encoding_format` | ⚠️ Partial | Always returns `float` |

### **Example with OpenAI Client (Compatible!)**

```python
from openai import OpenAI

# Initialize client with your API endpoint
client = OpenAI(
    base_url="https://fahmiaziz-api-embedding.hf.space/api/v1",
    api_key="-"  # API key not required, but must be present
)

# Generate embeddings
embedding = client.embeddings.create(
    input="Hello",
    model="qwen3-0.6b"
)

# Access results
for item in embedding.data:
    print(f"Embedding: {item.embedding[:5]}...")  # First 5 dimensions
    print(f"Index: {item.index}")
```

### **Async OpenAI Client**

```python
from openai import AsyncOpenAI

# Initialize async client
client = AsyncOpenAI(
    base_url="https://fahmiaziz-api-embedding.hf.space/api/v1",
    api_key="-"
)

# Generate embeddings asynchronously
async def get_embeddings():
    try:
        embedding = await client.embeddings.create(
            input=["Hello", "World", "AI"],
            model="qwen3-0.6b"
        )
        return embedding
    except Exception as e:
        print(f"Error: {e}")

# Use in async context
embeddings = await get_embeddings()
```

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📚 Resources

- [API Documentation](API.md)
- [Sentence Transformers](https://www.sbert.net/)
- [FastAPI Docs](https://fastapi.tiangolo.com/)
- [OpenAI Python Client](https://github.com/openai/openai-python)
- [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard)
- [Hugging Face Spaces](https://huggingface.co/docs/hub/spaces)
- [Deploy Applications on Hugging Face Spaces](https://huggingface.co/blog/HemanthSai7/deploy-applications-on-huggingface-spaces)
- [Sync HF Spaces with GitHub](https://github.com/ruslanmv/How-to-Sync-Hugging-Face-Spaces-with-a-GitHub-Repository)
- [Duplicate & Clone Spaces](https://huggingface.co/docs/hub/spaces-overview#duplicating-a-space)

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Sentence Transformers** for the embedding models
- **FastAPI** for the excellent web framework
- **Hugging Face** for model hosting and Spaces
- **OpenAI** for the client library design
- **Open Source Community** for inspiration and support

---

## 📞 Support

- **Issues:** [GitHub Issues](https://github.com/fahmiaziz98/unified-embedding-api/issues)
- **Discussions:** [GitHub Discussions](https://github.com/fahmiaziz98/unified-embedding-api/discussions)
- **Hugging Face Space:** [fahmiaziz/api-embedding](https://huggingface.co/spaces/fahmiaziz/api-embedding)

---

<div align="center">

Made with ❤️ by the Open-Source Community

> ✨ "Unify your embeddings. Simplify your AI stack."

</div>