import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
import dspy
from dspy.teleprompt import BootstrapFewShot
from dspy.evaluate.evaluate import Evaluate
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

#####################
# Build RAG module
#####################


class GenerateAnswer(dspy.Signature):
    """Answer questions with short factoid answers."""

    context = dspy.InputField(desc="the content of a book")
    question = dspy.InputField()
    answer = dspy.OutputField(desc="answer about the content of the book")


class RAG(dspy.Module):
    def __init__(self, num_passages=3):
        super().__init__()

        self.retrieve = dspy.Retrieve(k=num_passages)
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)

    def forward(self, question):
        context = self.retrieve(question).passages
        prediction = self.generate_answer(context=context, question=question)
        return dspy.Prediction(context=context, answer=prediction.answer)


# # Validation logic: check that the predicted answer is correct.
# # Also check that the retrieved context does actually contain that answer.
# def validate_context_and_answer(example, pred, trace=None):
#     answer_EM = dspy.evaluate.answer_exact_match(example, pred)
#     answer_PM = dspy.evaluate.answer_passage_match(example, pred)
#     return answer_EM and answer_PM

#####################
# Building the datasets
#####################

questions = [
    "What is the setting of the movie Stand By Me?",
    "Who is the writer of the story being told in the movie?",
    "What is the main character's name?",
    "How old was the writer when he saw a dead body for the first time?",
    "What is the name of the town where the story takes place?",
    "What is the significance of the radio broadcast in the beginning of the movie?",
    "Why does Vern bring a comb on their journey?",
    "What tragic event happened to the writer's older brother?",
    "How many people lived in Castle Rock?",
    "What is the group of friends' plan for their adventure?",
    "Who is the leader of the group?",
    "What is the reason behind bringing a pistol on the journey?",
    "How far do the boys estimate their journey to be?",
    "What is the writer's relationship with his parents like?",
    "What is the writer's nickname?",
    "What is the writer's older brother's name?",
    "What is the significance of the canteen in the story?",
    "How does the writer feel about his visibility at home?",
    "What is the writer's attitude towards the adventure with his friends?",
    "How does the writer's personal journey parallel the physical journey with his friends?",
]

trainset = questions[:10]  # 10 examples for training
devset = questions[10:15]  # 5 examples for development
testset = questions[15:]  # 5 examples for testing

trainset = [
    dspy.Example(question=question).with_inputs("question") for question in trainset
]
devset = [
    dspy.Example(question=question).with_inputs("question") for question in devset
]
testset = [
    dspy.Example(question=question).with_inputs("question") for question in testset
]


#####################
# Bulid metric module
#####################

metricLM = dspy.OpenAI(
    model="gpt-3.5-turbo", max_tokens=1000, model_type="chat", api_key=OPENAI_API_KEY
)


class Assess(dspy.Signature):
    """Assess the quality of an answer to a question."""

    context = dspy.InputField(desc="The context for answering the question.")
    assessed_question = dspy.InputField(desc="The evaluation criterion.")
    assessed_answer = dspy.InputField(desc="The answer to the question.")
    assessment_answer = dspy.OutputField(
        desc="A rating between 1 and 5. Only output the rating and nothing else."
    )


def llm_metric(gold, pred, trace=None):
    predicted_answer = pred.answer
    question = gold.question

    print(f"Test Question: {question}")
    print(f"Predicted Answer: {predicted_answer}")

    detail = "Is the assessed answer detailed?"
    faithful = "Is the assessed text grounded in the context? Say no if it includes significant facts not in the context."
    overall = f"Please rate how well this answer answers the question, `{question}` based on the context.\n `{predicted_answer}`"

    with dspy.context(lm=metricLM):
        context = dspy.Retrieve(k=5)(question).passages
        detail = dspy.ChainOfThought(Assess)(
            context="N/A", assessed_question=detail, assessed_answer=predicted_answer
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

    total = (
        float(detail.assessment_answer)
        + float(faithful.assessment_answer) * 2
        + float(overall.assessment_answer)
    )

    return total / 5.0


#####################
# Evaluate the UNCOMPILED Model
#####################

# Evaluate our RAG Program before it is compiled
evaluate = Evaluate(
    devset=devset, num_threads=4, display_progress=True, display_table=5
)

uncompiled_evaluation = evaluate(RAG(), metric=llm_metric)

print(f"## Score for uncompiled: {uncompiled_evaluation}")

# Metric analysis
# llm.inspect_history(n=1)

#####################
# Evaluate the COMPILED Model
#####################

# Set up a basic teleprompter, which will compile our RAG program.
teleprompter = BootstrapFewShot(metric=llm_metric)

# Compile!
compiled_rag = teleprompter.compile(RAG(), trainset=trainset)

compiled_evaluation = evaluate(compiled_rag, metric=llm_metric)

print(f"## Score for compiled: {compiled_evaluation}")

#####################
# Make questions
#####################

# Ask any question you like to this simple RAG program.
my_question = "What is the name of all the brothers in the book?"

# Get the prediction. This contains `pred.context` and `pred.answer`.
pred = compiled_rag(my_question)

# Print the contexts and the answer.
print(f"Question: {my_question}")
print(f"Predicted Answer: {pred.answer}")
print(f"Retrieved Contexts (truncated): {[c[:200] + '...' for c in pred.context]}")


#### Alternative metric

# # Validation logic: check that the predicted answer is correct.
# # Also check that the retrieved context does actually contain that answer.
# def validate_context_and_answer(example, pred, trace=None):
#     answer_EM = dspy.evaluate.answer_exact_match(example, pred)
#     answer_PM = dspy.evaluate.answer_passage_match(example, pred)
#     return answer_EM and answer_PM


# # Set up a basic teleprompter, which will compile our RAG program.
# teleprompter = BootstrapFewShot(metric=validate_context_and_answer)
