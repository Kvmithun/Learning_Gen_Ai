import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq


load_dotenv()

my_api_key=os.getenv("GROQ_API_KEY")

if not my_api_key:
    raise ValueError("no api key")


client=Groq(api_key=my_api_key)
model="llama-3.3-70b-versatile"


def llms_ans(prompt):
    message={
        'role':'user',
        'content':prompt,
    }
    messages=[message]
    response=client.chat.completions.create(model=model,messages=messages)
    ans=response.choices[0].message.content
    return ans


bad_prompt='''
this is a user complaint 
my laptop is not working 
 classify this'''


good_prompt='''
#role : you are a support assistant at mobile laptop company 
#task : you have to classify the issue in a category
this is a user complaint 
#constraint: you have to classify the issuen in one three categories namely billing 
,techincal ,return
#OUTPUT FORMAT
Your answer should be in one word only. The one word shoud be one of the categories given in constraints
#Example
For instance if a user compalin says he wants a refund then the category is Return
#FALLBACK
If the issue is unrelated to any of the categories mentioned in constraints, then the answer should be OTHER
This is a user complaint:
my girlfiend is not talking 
'''

print(llms_ans(good_prompt))