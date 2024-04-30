#####################
# Imports
#####################
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


#####################
# Constants
#####################
BOOK_PATH = "books/back-to-the-future-script.txt"
BOOK_NAME = "Back to the Future"
COLLECTION_NAME = BOOK_NAME.lower().replace(" ", "-")
VECTOR_STORE = "./vector-store-langchain"
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")


#####################
# ChromaDB setup
#####################
# if VECTOR_STORE does not exist, create it
if not os.path.exists(VECTOR_STORE):
    f = open(BOOK_PATH)
    text = f.read()

    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=80)

    documents = splitter.create_documents([text])

    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=OpenAIEmbeddings(),
        persist_directory=VECTOR_STORE,
    )
else:
    vector_store = Chroma(
        persist_directory=VECTOR_STORE,
        embedding_function=OpenAIEmbeddings(),
    )

retriever = vector_store.as_retriever()

#####################
# Langchain prompting
#####################
prompt = ChatPromptTemplate.from_template(
    """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: {question}
Context: {context}
Answer: """
)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


llm = ChatOpenAI(model="gpt-4-turbo")

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


response = rag_chain.invoke(
    "What is the name of the main character in Back to the Future?"
)

print(response)


# Alternative way to define the rag_chain using itemgetter
# https://python.langchain.com/docs/expression_language/primitives/parallel/#using-itemgetter-as-shorthand
# rag_chain = (
#     {
#         "context": itemgetter("question") | retriever,
#         "question": itemgetter("question"),
#     }
#     | prompt
#     | llm
#     | StrOutputParser()
# )
# response = rag_chain.invoke(
#     {"question": "What is the name of the main character in Back to the Future?"}
# )
