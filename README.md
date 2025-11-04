---
title: Api Embedding
emoji: 🐠
colorFrom: green
colorTo: purple
sdk: docker
pinned: false
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

# 🧠 Unified Embedding API

> 🧩 Unified API for all your Embedding, Sparse & Reranking Models — plug and play with any model from Hugging Face or your own fine-tuned versions.

---

## 🚀 Overview

**Unified Embedding API** is a modular and open-source **RAG-ready API** built for developers who want a simple, unified way to access **dense**, **sparse**, and **reranking** models.

It’s designed for **vector search**, **semantic retrieval**, and **AI-powered pipelines** — all controlled from a single `config.yaml` file.

⚠️ **Note:** This is a development API.  
For production deployment, host it on cloud platforms such as **Hugging Face TEI**, **AWS**, **GCP**, or any cloud provider of your choice.

---

## 🧩 Features

- 🧠 **Unified Interface** — One API to handle dense, sparse, and reranking models.
- ⚡ **Batch Processing** — Automatic single/batch.
- 🔧 **Flexible Parameters** — Full control via kwargs and options
- 🔍 **Vector DB Ready** — Easily integrates with FAISS, Chroma, Qdrant, Milvus, etc.
- 📈 **RAG Support** — Perfect base for Retrieval-Augmented Generation systems.
- ⚡ **Fast & Lightweight** — Powered by FastAPI and optimized with async processing.
- 🧰 **Extendable** —  Switch models instantly via `config.yaml` and add your own models or pipelines effortlessly.

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
│   │       |── health.py
│   │       └── rerank.py       # endpoint reranking
│   ├── core/
│   │   ├── base.py
│   │   ├── config.py
│   │   ├── exceptions.py
│   │   └── manager.py
│   ├── models/
│   │   ├── embeddings/
│   │   │   ├── dense.py        # dense model
│   │   │   └── sparse.py       # sparse model
│   │   │   └── rank.py         # reranking model
│   │   └── schemas/
│   │       ├── common.py
│   │       ├── requests.py       
│   │       └── responses.py
│   ├── config/
│   │   ├── settings.py
│   │   └── models.yaml         # add/change models here
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

2. **Add HF_TOKEN environment variable**  Make sure your space is public

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
  Click **⋯** →  **Embed this Space** -> copy **Direct URL**
   ```
   https://YOUR_USERNAME-api-embedding.hf.space
   https://YOUR_USERNAME-api-embedding.hf.space/docs  # Interactive docs
   ```

That’s it! You now have a live embedding API endpoint powered by your models.

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

## 📖 Usage Examples

### **Python**

```python
import requests

url = "http://localhost:7860/api/v1/embeddings/embed"

# Single embedding
response = requests.post(url, json={
    "texts": ["What is artificial intelligence?"],
    "model_id": "qwen3-0.6b"
})
print(response.json())

# Batch embeddings
response = requests.post(url, json={
    "texts": [
        "First document",
        "Second document", 
        "Third document"
    ],
    "model_id": "qwen3-0.6b",
    "options": {
        "normalize_embeddings": True
    }
})
embeddings = response.json()["embeddings"]
```

### **cURL**

```bash
# Single embedding (Dense)
curl -X POST "http://localhost:7860/api/v1/embeddings/embed" \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["Hello world"],
    "prompt": "add instructions here",
    "model_id": "qwen3-0.6b"
  }'

# Batch embeddings (Sparse)
curl -X POST "http://localhost:7860/api/v1/embeddings/embed" \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["First doc", "Second doc", "Third doc"],
    "model_id": "splade-pp-v2"
  }'

# Reranking
curl -X POST "http://localhost:7860/api/v1/rerank" \
  -H "Content-Type: application/json" \
  -d '{
  "documents": [
    "Python is a popular language for data science due to its extensive libraries.",
    "R is widely used in statistical computing and data analysis.",
    "Java is a versatile language used in various applications, including data science.",
    "SQL is essential for managing and querying relational databases.",
    "Julia is a high-performance language gaining popularity for numerical computing and data science."
  ],
  "model_id": "bge-v2-m3",
  "query": "Python best programming languages for data science",
  "top_k": 3
}'

# Query embedding with options
curl -X POST "http://localhost:7860/api/v1/embeddings/query" \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["What is machine learning?"],
    "model_id": "qwen3-0.6b",
    "options": {
      "normalize_embeddings": true,
      "batch_size": 32
    }
  }'
```

### **JavaScript/TypeScript**

```typescript
const url = "http://localhost:7860/api/v1/embeddings/embed";

const response = await fetch(url, {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
  },
  body: JSON.stringify({
    texts: ["Hello world"],
    model_id: "qwen3-0.6b",
  }),
});

const data = await response.json();
console.log(data.embedding);
```

---

## 📊 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/embeddings/embed` | POST | Generate document embeddings (single/batch) |
| `/api/v1/embeddings/query` | POST | Generate query embeddings (single/batch) |
| `/api/v1/rerank` | POST | Rerank documents based on a query |
| `/api/v1/models` | GET | List available models |
| `/api/v1/models/{model_id}` | GET | Get model information |
| `/health` | GET | Health check |
| `/` | GET | API information |
| `/docs` | GET | Interactive API documentation |


### 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

**Development Setup:**

```bash
git clone https://github.com/fahmiaziz/unified-embedding-api.git
cd unified-embedding-api
pip install -r requirements-dev.txt
pre-commit install  # (optional)
```

---

## 📚 Resources

- [API Documentation](API.md)
- [Sentence Transformers](https://www.sbert.net/)
- [FastAPI Docs](https://fastapi.tiangolo.com/)
- [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard)
- [Hugging Face Spaces](https://huggingface.co/docs/hub/spaces)
- [Deploy Applications on Hugging Face Spaces (Official Guide)](https://huggingface.co/blog/HemanthSai7/deploy-applications-on-huggingface-spaces)
- [How-to-Sync-Hugging-Face-Spaces-with-a-GitHub-Repository by Ruslanmv](https://github.com/ruslanmv/How-to-Sync-Hugging-Face-Spaces-with-a-GitHub-Repository?tab=readme-ov-file)
- [Duplicate & Clone space to local machine](https://huggingface.co/docs/hub/spaces-overview#duplicating-a-space)
---

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Sentence Transformers** for the embedding models
- **FastAPI** for the excellent web framework
- **Hugging Face** for model hosting and Spaces
- **Open Source Community** for inspiration and support

---

## 📞 Support

- **Issues:** [GitHub Issues](https://github.com/fahmiaziz/unified-embedding-api/issues)
- **Discussions:** [GitHub Discussions](https://github.com/fahmiaziz/unified-embedding-api/discussions)
- **Hugging Face Space:** [fahmiaziz/api-embedding](https://huggingface.co/spaces/fahmiaziz/api-embedding)

---

> ✨ “Unify your embeddings. Simplify your AI stack.”

<div align="center">

**⭐ Star this repo if you find it useful!**

Made with ❤️ by the Open-Source Community

</div>



