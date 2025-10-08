import os
from langchain_google_genai import GoogleGenerativeAI

api_key = os.environ.get('GOOGLE_API_KEY')
print('API key present?', bool(api_key))
llm = GoogleGenerativeAI(model='models/gemini-2.5-pro', google_api_key=api_key)
print('LLM init OK:', type(llm))
