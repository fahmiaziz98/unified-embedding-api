import time
from typing import List, Dict
from loguru import logger
from pinecone import Pinecone, ServerlessSpec

from .embedding_client import EmbeddingAPIClient


class PineconeVectorDB:
    """A client for interacting with Pinecone for hybrid (dense-sparse) vector search."""

    def __init__(
        self,
        api_key: str,
        embedding_api_url: str,
        cloud: str = "aws",
        region: str = "us-east-1",
    ) -> None:
        """
        Initializes the PineconeVectorDB client.

        Args:
            api_key (str): Your Pinecone API key.
            embedding_api_url (str): The base URL for the embedding API service.
            cloud (str): The cloud provider for the Pinecone index. Defaults to "aws".
            region (str): The region for the Pinecone index. Defaults to "us-east-1".
        """
        self.pc = Pinecone(api_key=api_key)
        self.api_client = EmbeddingAPIClient(embedding_api_url)
        self.cloud = cloud
        self.region = region

    def create_index_db(
        self,
        index_name: str,
        dimension: int,
    ) -> None:
        """
        Creates a new Pinecone index if it doesn't already exist.

        Args:
            index_name (str): The name of the index to create.
            dimension (int): The dimension of the dense vectors.
        """
        if index_name not in self.pc.list_indexes().names():
            logger.info(f"📦 Creating index: {index_name}")
            self.pc.create_index(
                name=index_name,
                dimension=dimension,
                metric="dotproduct",
                spec=ServerlessSpec(cloud=self.cloud, region=self.region),
            )

            while not self.pc.describe_index(index_name).status["ready"]:
                logger.debug("⏳ Waiting for index to be ready...")
                time.sleep(1)

            logger.success(f"✅ Index {index_name} created successfully")
        else:
            logger.info(f"ℹ️  Index {index_name} already exists")

        index = self.pc.Index(index_name)
        stats = index.describe_index_stats()
        logger.info(f"📊 Index stats: {stats}")

    async def push_data_to_index(
        self, documents: List[Dict[str, str]], index_name: str, batch_size: int = 8
    ) -> None:
        """
        Uploads documents to a Pinecone index in batches.

        Args:
            documents (List[Dict[str, str]]): A list of documents, where each document is a dictionary
                                              with 'id', 'question', and 'context' keys.
            index_name (str): The name of the Pinecone index.
            batch_size (int): The size of each batch for processing. Defaults to 8.
        """
        index = self.pc.Index(index_name)
        total_docs = len(documents)

        logger.info(
            f"📤 Uploading {total_docs} documents in batches of {batch_size}..."
        )

        for i in range(0, total_docs, batch_size):
            batch = documents[i : i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (total_docs + batch_size - 1) // batch_size

            logger.debug(
                f"\n🔄 Processing batch {batch_num}/{total_batches} ({len(batch)} docs)..."
            )

            texts = [doc["context"] for doc in batch]
            ids = [doc["id"] for doc in batch]

            await self._push_hybrid_batch(index, batch, texts, ids)

            logger.info(f"✅ Batch {batch_num}/{total_batches} uploaded")

        logger.success(f"\n🎉 All {total_docs} documents uploaded successfully!")

    async def _push_hybrid_batch(
        self,
        index,
        batch: List[Dict],
        texts: List[str],
        ids: List[str],
    ) -> None:
        """
        A helper method to generate and upload a batch of hybrid vectors.

        Args:
            index: The Pinecone index object.
            batch (List[Dict]): The batch of original documents.
            texts (List[str]): The list of texts ('context') to embed.
            ids (List[str]): The list of document IDs.
        """

        embeddings = await self.api_client.get_dense_embeddings(texts)
        logger.info(
            f"  ✓ Generated {len(embeddings)} dense embeddings (dim: {len(embeddings[0])})"
        )

        sparse_embeddings = await self.api_client.get_sparse_embeddings(texts)
        logger.info(f"  ✓ Generated {len(sparse_embeddings)} sparse embeddings")

        vectors = []
        for doc_id, doc, embedding, sparse_emb in zip(
            ids, batch, embeddings, sparse_embeddings
        ):
            vectors.append(
                {
                    "id": doc_id,
                    "values": embedding,
                    "sparse_values": {
                        "indices": sparse_emb["indices"],
                        "values": sparse_emb["values"],
                    },
                    "metadata": {
                        "question": doc["question"],
                        "context": doc["context"],
                    },
                }
            )

        index.upsert(vectors=vectors)
        logger.info(f"  ✓ Uploaded {len(vectors)} hybrid vectors")

    async def query(
        self,
        query: str,
        index_name: str,
        alpha: float = 0.5,
        top_k: int = 5,
        include_metadata: bool = True,
    ) -> Dict:
        """
        Performs a hybrid search query on a Pinecone index.

        Args:
            query (str): The query string.
            index_name (str): The name of the Pinecone index.
            alpha (float): The weight for hybrid search, between 0 and 1.
                           1 for pure dense search, 0 for pure sparse search. Defaults to 0.5.
            top_k (int): The number of results to return. Defaults to 5.
            include_metadata (bool): Whether to include metadata in the response. Defaults to True.

        Returns:
            Dict: The query response from Pinecone.
        """
        index = self.pc.Index(index_name)

        logger.info("🎯 Performing hybrid search...")
        logger.info("Generate sparse & dense embeddings...")

        query_embedding = await self.api_client.get_dense_embeddings([query])
        query_sparse_embedding = await self.api_client.get_sparse_embeddings([query])

        sparse_vec, dense_vec = self.hybrid_scale(
            query_embedding[0], query_sparse_embedding[0], alpha
        )

        query_response = index.query(
            vector=dense_vec,
            sparse_vector=sparse_vec,
            top_k=top_k,
            include_metadata=include_metadata,
        )

        return query_response

    async def query_with_rerank(
        self,
        query: str,
        index_name: str,
        alpha: float = 0.5,
        initial_top_k: int = 20,
        final_top_k: int = 5,
    ) -> List[Dict]:
        """
        Performs a query and then reranks the results for improved accuracy.

        Args:
            query (str): The query string.
            index_name (str): The name of the Pinecone index.
            alpha (float): The weight for the initial hybrid search. Defaults to 0.5.
            initial_top_k (int): The number of documents to retrieve from the initial vector search.
                                 Defaults to 20.
            final_top_k (int): The number of documents to return after reranking. Defaults to 5.

        Returns:
            List[Dict]: A list of reranked documents with their scores and metadata.
        """
        search_results = await self.query(
            query=query, index_name=index_name, alpha=alpha, top_k=initial_top_k
        )

        contexts = []
        metadata_map = {}

        for match in search_results["matches"]:
            context = match["metadata"].get("context", "")
            contexts.append(context)
            metadata_map[context] = {
                "id": match["id"],
                "score": match["score"],
                "question": match["metadata"].get("question", ""),
                "metadata": match["metadata"],
            }

        if not contexts:
            return []

        logger.info(f"🎯 Reranking top {initial_top_k} results to {final_top_k}...")
        reranked = await self.api_client.rerank_documents(
            query=query, documents=contexts, top_k=final_top_k
        )

        final_results = []
        for item in reranked:
            context = item["text"]
            original_data = metadata_map.get(context, {})

            final_results.append(
                {
                    "id": original_data.get("id"),
                    "rerank_score": item["score"],
                    "original_score": original_data.get("score"),
                    "question": original_data.get("question"),
                    "context": context,
                    "metadata": original_data.get("metadata", {}),
                }
            )

        logger.success("✅ Reranking complete!")
        return final_results

    def delete_index(self, index_name: str) -> None:
        """
        Deletes a Pinecone index.

        Args:
            index_name (str): The name of the index to delete.
        """
        if index_name in self.pc.list_indexes().names():
            self.pc.delete_index(index_name)
            logger.success(f"🗑️  Deleted index: {index_name}")
        else:
            logger.warning(f"⚠️  Index {index_name} not found")

    def hybrid_scale(
        self, dense: List[float], sparse: Dict[str, List], alpha: float
    ) -> tuple:
        """
        Scales dense and sparse vectors according to the alpha weight.

        Args:
            dense (List[float]): The dense vector.
            sparse (Dict[str, List]): The sparse vector, containing 'indices' and 'values'.
            alpha (float): The weighting factor, between 0 and 1.
                           alpha=1 gives full weight to dense, alpha=0 gives full weight to sparse.

        Returns:
            tuple: A tuple containing the scaled sparse vector and the scaled dense vector.
        """
        if alpha < 0 or alpha > 1:
            raise ValueError("Alpha must be between 0 and 1")

        # Scale sparse values
        hsparse = {
            "indices": sparse["indices"],
            "values": [v * (1 - alpha) for v in sparse["values"]],
        }

        # Scale dense values
        hdense = [v * alpha for v in dense]

        return hsparse, hdense

    async def close(self):
        """
        Closes the underlying EmbeddingAPIClient.

        This should be called to ensure that the HTTP client session is properly
        terminated and resources are released.
        """
        await self.api_client.close()
