from langgraph.graph import END, StateGraph, START
import pandas as pd

from src.graph import check_relevance, GraphState, grade_documents, check_intents, sentiment_analysis, retrieve, \
    generate, decide_to_proceed, decide_to_generate

workflow = StateGraph(GraphState)

workflow.add_node("check_relevance", check_relevance)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("check_intents", check_intents)
workflow.add_node("sentiment_analysis", sentiment_analysis)
workflow.add_node("retrieve", retrieve)
workflow.add_node("generate", generate)

workflow.add_edge(START, "check_relevance")
workflow.add_conditional_edges("check_relevance",
                               decide_to_proceed,
                               {
                                   "check_intents": "check_intents",
                                   "END": END
                               })
workflow.add_edge("check_intents", "sentiment_analysis")
workflow.add_edge("sentiment_analysis", "retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges("grade_documents",
                               decide_to_generate,
                               {
                                   "generate": "generate",
                                   "END": END
                               })
workflow.add_edge("generate", END)

app = workflow.compile()


df = pd.read_csv('sample_dataset4.csv')

for col in GraphState.__annotations__.keys():
    if col not in df.columns:
        df[col] = None

for index, row in df.iterrows():
    query = row['query']
    inputs = {"question": query}

    result_data = {}

    for output in app.stream(inputs):
        for key, value in output.items():
            print(f"Node '{key}':")
        print("\n---\n")

    # collect the values
    result_data["question"] = value.get("question", query)
    result_data["relevance"] = value.get("relevance")
    result_data["documents"] = value.get("documents", [])
    result_data["emotion"] = value.get("emotion", "")
    result_data["sentiment"] = value.get("sentiment", "")
    result_data["intent"] = value.get("intent", "")
    result_data["generation"] = value.get("generation", "")

    df.at[index, 'documents'] = result_data["documents"]
    df.at[index, 'relevance'] = result_data["relevance"]
    df.at[index, 'emotion'] = result_data["emotion"]
    df.at[index, 'sentiment'] = result_data["sentiment"]
    df.at[index, 'intent'] = result_data["intent"]
    df.at[index, 'generation'] = result_data["generation"]

df.to_csv('output-test.csv', index=False)

