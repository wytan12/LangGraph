from langgraph.graph import END, StateGraph, START


from LangGraph.graph import check_relevance, GraphState, grade_documents, check_intents, sentiment_analysis, retrieve, \
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

inputs = {"question": "What is XGBoost?"}
for output in app.stream(inputs):
    for key, value in output.items():
        # Node
        print(f"Node '{key}':")
    print("\n---\n")

print(value["generation"])