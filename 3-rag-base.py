import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
import dspy
import chromadb
from dspy.retrieve.chromadb_rm import ChromadbRM
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

BOOK_PATH = "books/stand-by-me-script.txt"
BOOK_NAME = "Stand By Me"
VECTOR_STORE = "./vector-store"
COLLECTION_NAME = "vector-books"

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# Set up the LM
gpt3_turbo = dspy.OpenAI(model="gpt-3.5-turbo", max_tokens=300, api_key=OPENAI_API_KEY)


chroma_client = chromadb.PersistentClient(path=VECTOR_STORE)
embedding_function = OpenAIEmbeddingFunction(
    api_key=OPENAI_API_KEY, model_name="text-embedding-3-small"
)
collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME, embedding_function=embedding_function
)

if collection.count() == 0:
    f = open(BOOK_PATH)
    text = f.read()

    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=80)

    documents = splitter.split_text(text)
    ids = [f"id_{i}" for i in range(len(documents))]
    book_name = BOOK_NAME
    metadatas = [{"book_name": book_name} for _ in range(len(documents))]

    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids,
    )


retriever_model = ChromadbRM(
    collection_name=COLLECTION_NAME,
    persist_directory=VECTOR_STORE,
    embedding_function=embedding_function,
    k=5,
)

dspy.settings.configure(lm=gpt3_turbo, rm=retriever_model)


class GenerateAnswer(dspy.Signature):
    """Answer questions with short factoid answers."""

    context = dspy.InputField(desc="the content of a book")
    question = dspy.InputField()
    answer = dspy.OutputField(desc="answer about the content of the book")


predictor = dspy.Predict(GenerateAnswer)

question = "What is the name of the main character in Stand By Me?"
prediction = predictor(question=question)
print(prediction.answer)

question = "What happens in the movie Stand By Me?"
prediction = predictor(question=question)
print(prediction.answer)

gpt3_turbo.inspect_history(n=1)
