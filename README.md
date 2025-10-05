# 🧠 Unified Embedding API

> 🧩 Unified API for all your Embedding & Reranking needs — plug and play with any model from Hugging Face or your own fine-tuned versions. This official repository from huggingface space

---

## 🚀 Overview

**Unified Embedding API** is a modular and open-source **RAG-ready API** built for developers who want a simple, unified way to access **dense**, and **sparse** models.

It’s designed for **vector search**, **semantic retrieval**, and **AI-powered pipelines** — all controlled from a single `config.yaml` file.

⚠️ **Note:** This is a development API.  
For production deployment, host it on cloud platforms such as **Hugging Face TGI**, **AWS**, or **GCP**.

---

## 🧩 Features

- 🧠 **Unified Interface** — One API to handle dense, sparse, and reranking models.
- ⚙️ **Configurable** — Switch models instantly via `config.yaml`.
- 🔍 **Vector DB Ready** — Easily integrates with FAISS, Chroma, Qdrant, Milvus, etc.
- 📈 **RAG Support** — Perfect base for Retrieval-Augmented Generation systems.
- ⚡ **Fast & Lightweight** — Powered by FastAPI and optimized with async processing.
- 🧰 **Extendable** — Add your own models or pipelines effortlessly.

---

## 📁 Project Structure

```

unified-embedding-api/
│
├── core/
│   ├── embedding.py         
│   └── model_manager.py     
├── models/
|   └──model.py
├── app.py                   # Entry point (FastAPI server)
|── config.yaml              # Model + system configuration
├── Dockerfile                 
├── requirements.txt
└── README.md

```
---

## ☁️ How to Deploy (Free 🚀)

Deploy your **custom Embedding API** on **Hugging Face Spaces** — free, fast, and serverless.

### 🔧 Steps:

1. **Clone this Space Template:**
   👉 [Hugging Face Space — fahmiaziz/api-embedding](https://huggingface.co/spaces/fahmiaziz/api-embedding)
2. **Edit `config.yaml`** to set your own model names and backend preferences.
3. **Push your code** — Spaces will automatically rebuild and host your API.

That’s it! You now have a live embedding API endpoint powered by your models.

📘 **Tutorial Reference:**
[Deploy Applications on Hugging Face Spaces (Official Guide)](https://huggingface.co/blog/HemanthSai7/deploy-applications-on-huggingface-spaces)

---


## 🧑‍💻 Contributing

Contributions are welcome!
Please open an issue or submit a pull request to discuss changes.

---

## ⚠️ License

MIT License © 2025
Developed with ❤️ by the Open-Source Community.

---

> ✨ “Unify your embeddings. Simplify your AI stack.”

