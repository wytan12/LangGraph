from langchain_openai import AzureChatOpenAI
from langchain_openai.embeddings import AzureOpenAIEmbeddings
import os

azure_embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=os.environ['AZURE_OPENAI_ENDPOINT'],
    api_key=os.environ['AZURE_OPENAI_KEY'],
    azure_deployment=os.environ['DEPLOYMENT_NAME-TEXTEMBEDDING'],
    model=os.environ['MODEL_NAME-TEXTEMBEDDING']
)

azure_model = AzureChatOpenAI(
    api_version=os.environ['OPENAI_API_VERSION'],
    azure_endpoint=os.environ['AZURE_OPENAI_ENDPOINT'],
    api_key=os.environ['AZURE_OPENAI_KEY'],
    azure_deployment=os.environ['DEPLOYMENT_NAME_GPT35TURBO'],
    temperature=0
)