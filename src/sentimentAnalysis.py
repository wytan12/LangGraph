from transformers import pipeline
from langchain_core.pydantic_v1 import BaseModel, Field

class SentimentAnalyzer(BaseModel):
    sentiment: str
    emotion: str

emotion_classifier = pipeline("sentiment-analysis", model="michellejieli/emotion_text_classifier")
sentiment_pipeline = pipeline("sentiment-analysis", model = "cardiffnlp/twitter-roberta-base-sentiment-latest")


query = "What course is this?"
emo_result = emotion_classifier(query)
emotion = emo_result[0]['label']
sen_result = sentiment_pipeline(query)
sentiment = sen_result[0]['label']
print(f'Sentiment: {sentiment}, Emotion: {emotion}')