#####################
# Imports
#####################
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
import dspy
import chromadb
from dspy.retrieve.chromadb_rm import ChromadbRM
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction


#####################
# Constants
#####################
BOOK_PATH = "books/back-to-the-future-script.txt"
BOOK_NAME = "Back to the Future"
COLLECTION_NAME = BOOK_NAME.lower().replace(" ", "-")
VECTOR_STORE = "./vector-store-dspy"
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")


#####################
# ChromaDB setup
#####################
chroma_client = chromadb.PersistentClient(path=VECTOR_STORE)
embedding_function = OpenAIEmbeddingFunction(
    api_key=OPENAI_API_KEY, model_name="text-embedding-3-small"
)

collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME, embedding_function=embedding_function
)

collection_documents = collection.get()

if len(collection_documents["ids"]) == 0:
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


retriever_model = ChromadbRM(
    collection_name=COLLECTION_NAME,
    persist_directory=VECTOR_STORE,
    embedding_function=embedding_function,
    k=5,
)


#####################
# DSPy setup
#####################
gpt3_turbo = dspy.OpenAI(model="gpt-3.5-turbo", max_tokens=300, api_key=OPENAI_API_KEY)

dspy.settings.configure(lm=gpt3_turbo, rm=retriever_model)


#####################
# Signature
#####################
class GenerateAnswer(dspy.Signature):
    """Answer questions with short factoid answers."""

    context = dspy.InputField(desc="Content of a book")
    question = dspy.InputField(desc="Question about the content of the book")
    answer = dspy.OutputField(desc="Answer about the content of the book")


#####################
# Prediction
#####################
predictor = dspy.Predict(GenerateAnswer)

question = "What is the name of the main character?"
prediction = predictor(question=question)
print(prediction.answer)

question = "What happens in the movie?"
prediction = predictor(question=question)
print(prediction.answer)

# gpt3_turbo.inspect_history(n=1)
