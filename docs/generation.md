## Generation

### LLM Inference Engine Service

::: aif_gen.generate.engine

#### Prompt Mapper

![image](./img/prompt_mapper.svg)

::: aif_gen.generate.mappers.base.PromptMapperBase
::: aif_gen.generate.mappers.prompt_mapper.PromptMapper

#### Response Mapper

![image](./img/response_mapper.svg)

::: aif_gen.generate.mappers.base.ResponseMapperBase
::: aif_gen.generate.mappers.response_mapper.ResponseMapper

### Caching

::: aif_gen.generate.caching.AsyncElasticsearchCache
