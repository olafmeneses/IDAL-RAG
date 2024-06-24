import os

## Base configuration
### LLM
embeddings_name = 'dariolopez/roberta-base-bne-finetuned-msmarco-qa-es-mnrl-mn'

os.environ["TAVILY_API_KEY"] = "YOUR_TAVILY_API_KEY"
### Tracing cfg
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = "YOUR_LANGCHAIN_API_KEY"
os.environ["GROQ_API_KEY"] = "YOUR_GROQ_API_KEY"