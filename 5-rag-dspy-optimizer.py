#####################
# Imports
#####################
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
import dspy
from dspy.teleprompt import BootstrapFewShot
from dspy.evaluate.evaluate import Evaluate
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
gpt4_turbo = dspy.OpenAI(model="gpt-4-turbo", max_tokens=300, api_key=OPENAI_API_KEY)

dspy.settings.configure(lm=gpt4_turbo, rm=retriever_model)


#####################
# Signature
#####################
class GenerateAnswer(dspy.Signature):
    """Answer questions with short factoid answers."""

    context = dspy.InputField(desc="the content of a book")
    question = dspy.InputField()
    answer = dspy.OutputField(desc="answer about the content of the book")


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
        return dspy.Prediction(
            context=context, answer=prediction.answer, rationale=prediction.rationale
        )


#####################
# Data for training, development, and testing
#####################

questions = [
    {
        "question": "What does Marty ask Doc to pick up on his way to the mall?",
        "answer": "Video camera",
    },
    {"question": "What is Doc's latest experiment involving?", "answer": "A DeLorean"},
    {
        "question": "What does Doc say about the future's gravitational pull?",
        "answer": "He questions if there is a problem with it",
    },
    {
        "question": "What does Doc suggest to get Marty's parents to meet?",
        "answer": "They need to be alone together",
    },
    {
        "question": "What era do Marty's parents need to interact in?",
        "answer": "The 1950s",
    },
    {
        "question": "What does Doc forget to bring for his journey?",
        "answer": "Extra plutonium",
    },
    {"question": "Who finds Doc according to him?", "answer": "The Libyans"},
    {
        "question": "What is Doc's reaction when he realizes they have been found?",
        "answer": "Tells Marty to run",
    },
    {
        "question": "What vehicle is involved in Doc's experiment?",
        "answer": "A DeLorean",
    },
    {
        "question": "What does Marty refer to the situation as when he sees the DeLorean?",
        "answer": "Heavy",
    },
    {"question": "What does Doc record on tape?", "answer": "His historic journey"},
    {"question": "Who does Doc say is after them?", "answer": "The Libyans"},
    {
        "question": "What does Marty call the Libyans in his exclamation?",
        "answer": "Bastards",
    },
    {
        "question": "What does Doc instruct to do when the Libyans arrive?",
        "answer": "Unroll their fire",
    },
    {
        "question": "What does Marty say when he first sees the DeLorean?",
        "answer": "It's a DeLorean, right?",
    },
    {
        "question": "What does Doc assure Marty when he questions the experiment?",
        "answer": "All your questions will be answered",
    },
    {
        "question": "What does Doc need to make his time travel experiment work?",
        "answer": "Plutonium",
    },
    {
        "question": "What does Marty refer to the weight of the situation?",
        "answer": "Heavy",
    },
    {
        "question": "What does Doc plan to document with the video camera?",
        "answer": "His experiment",
    },
    {
        "question": "What is the urgency in Doc's voice when he asks Marty to pick up the video camera?",
        "answer": "Very important",
    },
]

trainset = questions[:10]  # 10 examples for training
devset = questions[10:15]  # 5 examples for development
testset = questions[15:]  # 5 examples for testing

trainset = [
    dspy.Example(question=i["question"], answer=i["answer"]).with_inputs("question")
    for i in trainset
]
devset = [dspy.Example(question=i["question"]).with_inputs("question") for i in devset]
testset = [
    dspy.Example(question=i["question"]).with_inputs("question") for i in testset
]


#####################
# Bulid metric module
#####################

metricLM = dspy.OpenAI(
    model="gpt-3.5-turbo", max_tokens=300, model_type="chat", api_key=OPENAI_API_KEY
)


class Assess(dspy.Signature):
    """Assess the quality of an answer to a question."""

    context = dspy.InputField(desc="The context for answering the question.")
    assessed_question = dspy.InputField(desc="The evaluation criterion.")
    assessed_answer = dspy.InputField(desc="The answer to the question.")
    assessment_answer = dspy.OutputField(
        desc="A rating between 1 and 5. Only output the rating and nothing else.",
        prefix="Rating[1-5]:",
    )


def llm_metric(gold, pred, trace=None):
    question = gold.question
    predicted_answer = pred.answer
    context = pred.context

    print(f"Test Question: {question}")
    print(f"Predicted Answer: {predicted_answer}")

    detail = "Is the assessed answer detailed?"
    faithful = "Is the assessed text grounded in the context? Say no if it includes significant facts not in the context."
    overall = f"Please rate how well this answer answers the question, `{question}` based on the context.\n `{predicted_answer}`"

    with dspy.context(lm=metricLM):
        # context = dspy.Retrieve(k=5)(question).passages

        detail = dspy.ChainOfThought(Assess)(
            context=context, assessed_question=detail, assessed_answer=predicted_answer
        )
        faithful = dspy.ChainOfThought(Assess)(
            context=context,
            assessed_question=faithful,
            assessed_answer=predicted_answer,
        )
        overall = dspy.ChainOfThought(Assess)(
            context=context, assessed_question=overall, assessed_answer=predicted_answer
        )

    print(f"Faithful: {faithful.assessment_answer}")
    print(f"Detail: {detail.assessment_answer}")
    print(f"Overall: {overall.assessment_answer}")

    score = (
        float(detail.assessment_answer)
        + float(faithful.assessment_answer)
        + float(overall.assessment_answer)
    )

    return score / 3.0


#####################
# Evaluate the UNCOMPILED Model
#####################

# Evaluate our RAG Program before it is compiled
evaluate = Evaluate(
    devset=devset, num_threads=4, display_progress=True, display_table=5
)

uncompiled_evaluation = evaluate(RAG(), metric=llm_metric)

print(f"## Score for uncompiled: {uncompiled_evaluation}")

# gpt4_turbo.inspect_history(n=1)

#####################
# Evaluate the COMPILED Model
#####################

# Set up a basic optimizer, which will compile our RAG program.
optimizer = BootstrapFewShot(metric=llm_metric)

# Compile!
compiled_rag = optimizer.compile(RAG(), trainset=trainset)

compiled_evaluation = evaluate(compiled_rag, metric=llm_metric)

print(f"## Score for compiled: {compiled_evaluation}")

# gpt4_turbo.inspect_history(n=1)


#### Alternative metric

# # Validation logic: check that the predicted answer is correct.
# # Also check that the retrieved context does actually contain that answer.
# def validate_context_and_answer(example, pred, trace=None):
#     answer_EM = dspy.evaluate.answer_exact_match(example, pred)
#     answer_PM = dspy.evaluate.answer_passage_match(example, pred)
#     return answer_EM and answer_PM


# # Set up a basic teleprompter, which will compile our RAG program.
# teleprompter = BootstrapFewShot(metric=validate_context_and_answer)


#####################
# Compare the UNCOMPILED and COMPILED Models
#####################

for test in testset:
    question = test["question"]

    uncompiled_result = RAG()(question)
    compiled_result = compiled_rag(question)

    print(f"Question: {question}")
    print(f"Uncompiled Answer: {uncompiled_result.answer}")
    print(f"Uncompiled Rationale: {uncompiled_result.rationale}")
    print(f"Compiled Answer: {compiled_result.answer}")
    print(f"Compiled Rationale: {compiled_result.rationale}")
    print("\n")


# gpt4_turbo.inspect_history(n=1)
