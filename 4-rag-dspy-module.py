import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
import dspy
import chromadb
from dspy.retrieve.chromadb_rm import ChromadbRM
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction


BOOK_PATH = "books/back-to-the-future-script.txt"
BOOK_NAME = "Back to the Future"
COLLECTION_NAME = BOOK_NAME.lower().replace(" ", "-")
VECTOR_STORE = "./vector-store-dspy"
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")


# Set up the LM
gpt4_turbo = dspy.OpenAI(model="gpt-4-turbo", max_tokens=1000, api_key=OPENAI_API_KEY)


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

dspy.settings.configure(lm=gpt4_turbo, rm=retriever_model)


#####################
# Signature
#####################
class GenerateAnswer(dspy.Signature):
    """Answer questions with short factoid answers."""

    context = dspy.InputField(desc="Content of a book")
    question = dspy.InputField(desc="Question about the content of the book")
    answer = dspy.OutputField(desc="Answer about the content of the book")


#####################
# Build RAG module
#####################
class RAG(dspy.Module):
    def __init__(self, num_passages=3):
        super().__init__()

        self.retrieve = dspy.Retrieve(k=num_passages)
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)

    def forward(self, question):
        context = self.retrieve(question).passages
        prediction = self.generate_answer(context=context, question=question)
        return dspy.Prediction(context=context, answer=prediction.answer)


uncompiled_rag = RAG()

question = "What is the name of the main character?"
print(uncompiled_rag(question).answer)

question = "What happens in the movie?"
print(uncompiled_rag(question).answer)

question = "Create 20 questions and answers about the movie in json format. Like [{question, answer}]"
print(uncompiled_rag(question).answer)


# gpt3_turbo.inspect_history(n=1)
