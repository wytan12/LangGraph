from transformers import pipeline
from langchain_core.pydantic_v1 import BaseModel, Field

class SentimentAnalyzer(BaseModel):
    sentiment: str = Field(description="The sentiment of the input text.")
    emotion: str = Field(description="The emotion of the input text.")

class SentimentAnalysisPipeline:
    def __init__(self):
        # emotion and sentiment analysis models
        self.emotion_classifier = pipeline("sentiment-analysis", model="michellejieli/emotion_text_classifier")
        self.sentiment_pipeline = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")

    def analyze(self, query: str) -> SentimentAnalyzer:
        
        # emotion
        emo_result = self.emotion_classifier(query)
        emotion = emo_result[0]['label']

        # sentiment
        sen_result = self.sentiment_pipeline(query)
        sentiment = sen_result[0]['label']

        return SentimentAnalyzer(sentiment=sentiment, emotion=emotion)

