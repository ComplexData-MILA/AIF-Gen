`BACKOFF_RETRIES`: Change the number of retries with exponential backoff incase of inference request failure. Defaults to 3 retries.

`ELASTIC_SEARCH_API_KEY`: ElasticSearch API key. Helpful to cache model responses across generations.

`ELASTIC_SEARCH_HOST`: API Endpoint for ElasticSearch instance, if enabled. Helpful to cache model responses across generations.

`FORCE_CACHE_REFRESH`: Force refresh the cache entry for a generation of validation request.

`HF_TOKEN`: Access token for access to HuggingFace hub. Needed for uploading or downloading datasets to HuggingFace.

`OPENAI_API_KEY`: OpenAI API key. Can be arbitrary if using vLLM. Needed for generation or LLM-based validation metrics

`OPENAI_BASE_URL`: API Endpoint for vLLM or OpenAI inference server. Needed for generation or LLM-based validation metrics
