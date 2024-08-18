from typing import List
from typing_extensions import TypedDict

from LangGraph.generate import generate_answer
from LangGraph.intentClassifier import classify_intent
from LangGraph.queryGrader import grade_query
from LangGraph.retrievalGrader import retrieval_grader
from LangGraph.retriever import get_retriever
from LangGraph.sentimentAnalysis import SentimentAnalysisPipeline


class GraphState(TypedDict):
    question: str
    generation: str
    relevance: str
    emotion: str
    sentiment: str
    intent: str
    documents: List[str]

def check_relevance(state):
    print("---CHECK RELEVANCE START---")
    question = state["question"]
    print("Check query relevance to system")
    relevance_score = grade_query(question)
    relevance = relevance_score.binary_score
    print(relevance)
    return {**state, "relevance" : relevance}

def check_intents(state):
    print("---CHECK INTENTS START---")
    question = state["question"]
    print("Classify intents")
    label = classify_intent(question)
    print(label)
    return {**state, "intent": label}

def sentiment_analysis(state):
    print("---SENTIMENT ANALYSIS START---")
    question = state["question"]
    analyzer = SentimentAnalysisPipeline()
    result = analyzer.analyze(question)

    print(result.emotion)
    print(result.sentiment)
    return {**state, "emotion": result.emotion, "sentiment": result.sentiment}

def retrieve(state):
    print("---RETRIEVE---")
    question = state["question"]

    # Retrieval
    retriever = get_retriever("docs")
    documents = retriever.get_relevant_documents(question)
    print(documents)
    return {**state, "documents": documents}

def grade_documents(state):
    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]

    # Score each doc
    filtered_docs = []
    for d in documents:
        score = retrieval_grader(question, d.page_content)
        grade = score.binary_score
        if grade == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            continue
    return {**state, "documents": filtered_docs}

def generate(state):

    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    emotion = state["emotion"]
    sentiment = state["sentiment"]
    intent = state["intent"]

    generation = generate_answer(documents, question, emotion, sentiment, intent)
    print('question:', documents, emotion, sentiment, intent, generation)
    return {**state, "generation": generation}


def decide_to_proceed(state):
    state["question"]
    relevance = state["relevance"]

    if relevance == "yes":
        print("---DECIDE TO PROCEED---")
        return "check_intents"
    else:
        print("---Sorry---")
        state["generation"] = "Sorry I can't answer this question"
        return "END"


def decide_to_generate(state):
    print("---ASSESS GRADED DOCUMENTS---")
    state["question"]
    filtered_documents = state["documents"]

    if not filtered_documents:
        state['generation'] = "I cant answer that question."
        print("I cant answer that question.")
        return "END"
    else:
        # We have relevant documents, so generate answer
        print("---DECISION: GENERATE---")
        return "generate"
