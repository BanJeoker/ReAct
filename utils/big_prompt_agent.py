from vertexai.generative_models import GenerationConfig, GenerativeModel, Content, Part  
import importlib
import chat_functions
importlib.reload(chat_functions)
from chat_functions import execute

class BigPromptAgent:
    
    def __init__(self, pdf_path, model="gemini-1.5-pro-002") -> None:
        
        self.client = GenerativeModel(model)
        self.auditor_notes_doc=None
        with open(pdf_path, 'rb') as fp:
            self.auditor_notes_doc=Part.from_data(data=fp.read(),mime_type='application/pdf')
        
        
    def create_big_prompt(self, query):
        prompt=f"""
        You are a financial expert and an underwriter working for the commerical banking sector of the bank.
        You will be asked a financiall question based on the income statement, balance sheet, and cash flow information of a company.
        You are provided with auditor notes of that company where you can find actual numbers and potential drivers for the input question.
        You need to answer the question based on the auditor notes. Do not make assumptions. Ground you answer truthfully to the auditor notes provided.
        Now the questions is {query}.
        The auditor note is attached as follows:
        """
        return prompt
    
    def run(self, query):
        return execute(self.client, messages=[self.create_big_prompt(query), self.auditor_notes_doc])  
        
