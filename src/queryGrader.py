from langchain_core.prompts import ChatPromptTemplate
from datetime import datetime
from langchain_core.pydantic_v1 import BaseModel, Field

from LangGraph.config import azure_model


class GradeQuery(BaseModel):
    """Binary score for relevance check on user query."""

    binary_score: str = Field(
        description="Query are relevant to the system, 'yes' or 'no'"
    )

def grade_query(question) :

    llm = azure_model
    structured_llm_grader = llm.with_structured_output(GradeQuery)

    system = """You are a grader responsible for assessing the relevance of user queries, which is designed to assist students with questions about SC1015 Introduction to Data Science and Artificial Intelligence.\n
    Your task is to determine whether a user query is related to the course and its content, specifically if it mentions or relates to course-specific topics (assignments, assessments, projects, quizzes, or tests), course schedule information (e.g., Week 1, Week 8, Week 14), content from the textbook 'Artificial Intelligence: A Modern Approach (3rd edition)' by Russell and Norvig (Chapters 1, 2, 3, 4, and 6), or general course-related inquiries (deadlines, syllabus details, or tips for AI assessments) and other question related to data science and AI.\n
    If the query matches any of these criteria, you should consider it relevant and assign a score of 'yes.' If the query does not relate to these criteria, assign a score of 'no.'\n
    The goal is to ensure that the chatbot only responds to queries directly related to the course SC1015. Queries unrelated to the course or its content should be filtered out."""

    grade_query = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "User question: {question}")
        ]
    )

    query_grader = grade_query | structured_llm_grader
    result = query_grader.invoke({"question": question})
    
    return result
