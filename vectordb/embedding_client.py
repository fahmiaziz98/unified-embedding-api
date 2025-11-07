from httpx import AsyncClient, HTTPError, TimeoutException
from typing import List, Dict
import asyncio


class EmbeddingAPIClient:
    """
    A client for interacting with an embedding API.

    Attributes:
        base_url (str): The base URL of the embedding API.
        timeout (int): The timeout duration for requests in seconds.
        max_retries (int): The maximum number of retry attempts for failed requests.
        client (AsyncClient): An instance of AsyncClient for making HTTP requests.
    """

    def __init__(self, base_url: str, timeout: int = 60, max_retries: int = 3) -> None:
        """
        Initializes the EmbeddingAPIClient with the specified parameters.
        """

        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.client = AsyncClient(base_url=base_url, timeout=timeout)

    async def _make_request_with_retry(
        self, endpoint: str, payload: Dict, retry_count: int = 0
    ):
        """
        Helper method to make a POST request with retry logic.

        Args:
            endpoint (str): The endpoint URL to which the request is sent.
            payload (Dict): The JSON data to be sent in the request.
            retry_count (int, optional): The current retry attempt count. Defaults to 0.

        Returns:
            Dict: The JSON response from the API.

        Raises:
            Exception: If the request fails after the maximum number of retries.
        """
        try:
            response = await self.client.post(endpoint, json=payload)
            response.raise_for_status()
            return response.json()

        except (HTTPError, TimeoutException) as e:
            if retry_count < self.max_retries:
                wait_time = 2**retry_count
                print(
                    f"⚠️  Request failed, retrying in {wait_time}s... (attempt {retry_count + 1}/{self.max_retries})"
                )
                await asyncio.sleep(wait_time)
                return await self._make_request_with_retry(
                    endpoint, payload, retry_count + 1
                )
            else:
                raise Exception(f"❌ Failed after {self.max_retries} retries: {str(e)}")

    async def get_dense_embeddings(
        self, texts: List[str], model: str = "qwen3-0.6b"
    ) -> List[List[float]]:
        """
        Retrieve dense embeddings from the API.

        Args:
            texts (List[str]): A list of texts for which to retrieve embeddings.
            model (str): The model to use for generating embeddings. Defaults to "qwen3-0.6b".

        Returns:
            List[List[float]]: A list of dense embeddings corresponding to the input texts.
        """
        data = await self._make_request_with_retry(
            "/embeddings", {"input": texts, "model": model}
        )
        return [item["embedding"] for item in data["data"]]

    async def get_sparse_embeddings(
        self, texts: List[str], model: str = "splade-large-query"
    ) -> List[Dict[str, List]]:
        """
        Retrieve sparse embeddings from the API.

        Args:
            texts (List[str]): A list of texts for which to retrieve embeddings.
            model (str): The model to use for generating embeddings. Defaults to "splade-large-query".

        Returns:
            List[Dict[str, List]]: A list of sparse embeddings corresponding to the input texts.
        """
        data = await self._make_request_with_retry(
            "/embed_sparse", {"input": texts, "model": model}
        )
        return data["embeddings"]

    async def rerank_documents(
        self, query: str, documents: List[str], top_k: int = 5, model: str = "bge-v2-m3"
    ) -> List[Dict]:
        """
        Rerank a list of documents based on a query.

        Args:
            query (str): The query string used for reranking.
            documents (List[str]): A list of documents to be reranked.
            top_k (int): The number of top documents to return. Defaults to 5.
            model (str): The model to use for reranking. Defaults to "bge-v2-m3".

        Returns:
            List[Dict]: A list of reranked documents with their scores.
        """
        data = await self._make_request_with_retry(
            "/rerank",
            {"query": query, "documents": documents, "top_k": top_k, "model": model},
        )
        return data["results"]

    async def close(self):
        """
        Close the HTTP client.

        This method closes the AsyncClient instance to free up resources.
        """
        await self.client.aclose()
