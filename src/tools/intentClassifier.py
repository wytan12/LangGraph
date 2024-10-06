from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from datetime import datetime

from src.config import azure_model


class IntentClassifier(BaseModel):
    """Label user intent based on user query."""
    label: str = Field(
        description="User intent from the defined intents"
    )


# Initialize your LLM
llm = azure_model

# Define the system message
system = '''
You are an intelligent intent classifier. Your job is to analyze user queries and categorize them into one of the predefined intents. Each intent represents a specific type of request or inquiry related to course-related activities. The intents you need to consider are as follows:

1. Course Overview and Info: Questions about the course content, instructor, prerequisites, etc.
2. Course Assessment: Questions about assignments, grading, exams, etc.
3. Checking Announcement: Inquiries about the latest course announcements or updates.
4. Request Permission: Requests for extensions, attendance changes, or permissions related to the course.
5. Learning Course Content: Inquiries about lecture topics, notes, reading materials, etc.
6. Technical Issues: Issues related to accessing course resources, videos, or technical help.
7. Schedule and Deadlines: Questions regarding lecture schedules, deadlines, or reminders.
8. Course Materials: Requests for syllabi, slides, reading lists, etc.
9. Course Feedback: Questions or actions related to providing feedback or evaluations for the course.
10. Others: Any other inquiries or requests not covered by the above intents.

Given a user query, classify it into one of the intents. Respond only with the intent label.
'''

template = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "User question: {query}"),
    ]
)


def classify_intent(query: str) -> str:

    prompt = template.invoke({"query": query})
    print(prompt)

    response = llm(prompt)
    print(response)

    intent_label = response.content.strip()
    return intent_label
