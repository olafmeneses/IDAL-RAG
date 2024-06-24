from langchain_community.chat_models import ChatOllama
from langchain_groq import ChatGroq

def ChatLLM(model, local=False, json=False, temperature=0):
    format=None
    if json:
        format="json"
    if local:
        return ChatOllama(model=model, format=format, temperature=temperature)
    else:
        return ChatGroq(model=model, temperature=temperature)