import hashlib
import logging
import os
from typing import Optional

from elasticsearch import AsyncElasticsearch


class AsyncElasticsearchCache:
    def __init__(self, es: AsyncElasticsearch, index_name: str):
        """Ensure the Elasticsearch index exists before querying or inserting data."""
        self.es = es
        self.index_name = index_name
        self.is_refresh_required = bool(os.environ.get('FORCE_CACHE_REFRESH'))
        logging.info(f'Elastic Index Name: {self.index_name}')

        if self.is_refresh_required:
            logging.warning('FORCE_CACHE_REFRESH is enabled. All queries will miss.')

    @staticmethod
    async def maybe_from_env_var(index_name: str) -> 'AsyncElasticsearchCache | None':
        """Initialize from env var.

        Return None if any of the required env var is missing.
        """
        index_name = index_name.lower()

        required_env_keys = ['ELASTIC_SEARCH_HOST', 'ELASTIC_SEARCH_API_KEY']
        if not all((_key in os.environ) for _key in required_env_keys):
            logging.warning(
                'All of these are required to enable ElasticsearchCache: '
                f'{required_env_keys}.'
                ' Not enabling ElasticsearchCache since some keys are not set.'
            )
            return None

        es = AsyncElasticsearch(
            os.environ['ELASTIC_SEARCH_HOST'],
            api_key=os.environ['ELASTIC_SEARCH_API_KEY'],
            request_timeout=None,
        )

        # Ensure the index exists at startup
        # parse the name if / is present
        index_name = index_name.replace('/', '_')
        exists = await es.indices.exists(index=index_name)
        if not exists:
            await es.indices.create(index=index_name)

        return AsyncElasticsearchCache(es=es, index_name=index_name)

    @staticmethod
    def _get_query_hash(query: str, nonce: Optional[str] = None) -> str:
        """Obtain query hash given query and- optionally- nonce."""
        query_key = query
        if nonce is not None:
            query_key += f'\n{nonce}'

        return hashlib.sha256(query_key.encode()).hexdigest()

    async def get(self, query: str, nonce: Optional[str] = None) -> 'str | None':
        """Try reading response from cache.

        Args:
            query (str): The query to fetch from cache.
            nonce (str, optional): An optional nonce to differentiate cache entries.

        Returns:
            str | None: Cached result if available.
        """
        if self.is_refresh_required:
            return None

        # Cache lookup
        query_hash = self._get_query_hash(query=query, nonce=nonce)
        try:
            response = await self.es.get(index=self.index_name, id=query_hash)
            if response.get('found'):
                logging.info(f'Cache hit: {query_hash}')
                return response['_source']['result']
        except Exception:
            logging.debug(f'Cache miss: {query_hash}')

        return None  # Cache miss or index doesn't exist

    async def set(self, query: str, value: str, nonce: Optional[str] = None) -> None:
        """Set/Update cache.

        Args:
            query (str): The query whose result is to be cached.
            value (str): The value to store in cache.
            nonce (str, optional): An optional nonce to differentiate cache entries.
        """
        query_hash = self._get_query_hash(query=query, nonce=nonce)
        doc = {'query': query, 'result': value, 'nonce': nonce}
        await self.es.index(index=self.index_name, id=query_hash, document=doc)

    async def close(self) -> None:
        """Close Elasticsearch connection."""
        await self.es.close()
