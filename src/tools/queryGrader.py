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

    # system = """You are a grader responsible for assessing the relevance of user queries in a system designed to assist students with questions related to Data Science and Artificial Intelligence, including machine learning and the course SC1015 Data Science and Artificial Intelligence.\n
    # Your task is to determine whether a user query is related to the course and data science, artificial intelligence and machine learning content, or if it mentions or relates to course-specific topics (assignments, assessments, projects, quizzes, or tests), course schedule information (e.g., Week 1, Week 8, Week 14), content from the textbook 'Artificial Intelligence: A Modern Approach (3rd edition)' by Russell and Norvig (Chapters 1, 2, 3, 4, and 6), or general course-related inquiries (deadlines, syllabus details, or tips for AI assessments) and other question related to data science and AI.\n
    # If the query matches any of these criteria, you should consider it relevant and assign a score of 'yes.' If the query does not relate to these criteria, assign a score of 'no.'\n
    # The goal is to ensure that the chatbot only responds to queries directly related to the course SC1015. Queries unrelated to the course or its content should be filtered out."""

    system = """You are a grader responsible for assessing the relevance of user queries. The queries are intended to assist students with questions about SC1015 Introduction to Data Science and Artificial Intelligence.
    Your task is to determine whether a user query is relevant to the course and its content. The course covers a wide range of topics, including:
    1. **Data Science**: Concepts related to data collection, data cleaning, exploratory data analysis (EDA), statistical analysis, visualization techniques (e.g., histograms, boxplots, scatter plots), and tools like Pandas, NumPy, and Matplotlib.
    2. **Artificial Intelligence (AI)**: Topics including search algorithms, optimization techniques, logic and reasoning, decision making, AI ethics, and the history of AI as covered in the textbook 'Artificial Intelligence: A Modern Approach (3rd edition)' by Russell and Norvig (specifically Chapters 1, 2, 3, 4, and 6).
    3. **Machine Learning**: Practical and theoretical knowledge about supervised and unsupervised learning, algorithms such as linear regression, decision trees, k-nearest neighbors (KNN), support vector machines (SVM), and ensemble methods like random forests and XGBoost. Discussion on model evaluation techniques such as cross-validation, confusion matrices, precision, recall, F1 score, and ROC curves.
    4. **Deep Learning and Neural Networks**: Concepts such as feedforward neural networks, convolutional neural networks (CNNs), recurrent neural networks (RNNs), and training techniques like backpropagation, gradient descent, and the use of libraries like TensorFlow and PyTorch.
    5. **SC1015 Course-Specific Information**: Questions about assignments, assessments, projects, quizzes, tests, course schedule information (e.g., Week 1, Week 8, Week 14), deadlines, syllabus details, tips for AI and data science assessments, and any other inquiries related to the course content.
    Your role is to filter queries and assign a binary score of 'yes' or 'no' based on relevance. If the query mentions or relates to any of these topics or course-specific information, assign a score of 'yes.' If it does not, assign a score of 'no.'
    The goal is to ensure that the chatbot only responds to queries directly related to data science, artificial intelligence, machine learning, deep learning, neural networks, and SC1015 course content. Queries unrelated to these topics should be filtered out."""

    grade_query = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "User question: {question}")
        ]
    )

    query_grader = grade_query | structured_llm_grader
    result = query_grader.invoke({"question": question})
    
    return result
