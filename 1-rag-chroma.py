import os
import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb.utils.embedding_functions as embedding_functions
from openai import OpenAI

BOOK_PATH = "books/back-to-the-future-script.txt"
BOOK_NAME = "Back to the Future"
COLLECTION_NAME = BOOK_NAME.lower().replace(" ", "-")
VECTOR_STORE = "./vector-store-chroma"
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

client = chromadb.PersistentClient(path=VECTOR_STORE)

default_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=OPENAI_API_KEY, model_name="text-embedding-3-small"
)

try:
    collection = client.get_collection(
        name=COLLECTION_NAME, embedding_function=default_ef
    )
except:
    collection = None


if collection is None:
    collection = client.create_collection(
        name=COLLECTION_NAME, embedding_function=default_ef
    )

    f = open(BOOK_PATH)
    text = f.read()

    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=80)

    documents = splitter.split_text(text)
    ids = [f"id_{i}" for i in range(len(documents))]
    metadatas = [{"book_name": BOOK_NAME} for _ in range(len(documents))]

    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids,
    )

result = collection.query(query_texts=["Who is Doc?"], n_results=3)

client = OpenAI(api_key=OPENAI_API_KEY)

# context is the concatenation of the retrieved documents
context = "\n\n".join([doc for doc in result["documents"][0]])

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "system",
            "content": "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.",
        },
        {
            "role": "assistant",
            "content": f"Context: {context}",
        },
        {
            "role": "user",
            "content": "Who is Doc?",
        },
    ],
    model="gpt-4-turbo",
)

print(chat_completion.choices[0].message.content)
