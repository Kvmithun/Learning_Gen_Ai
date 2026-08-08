import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq


load_dotenv()

my_api_key=os.getenv("GROQ_API_KEY")
if not my_api_key:
    raise ValueError("No api key")
client=Groq(api_key=my_api_key)

model='llama-3.3-70b-versatile'

role='user'

#3 prompts

prompt1 ='hey'
prompt2='time travel in detail in 100 words'
prompt3='write an essay of 1000 words about deep learning '

prompts=[prompt1,prompt2,prompt3]
for prompt in prompts:
    message = {
        'role': role,
        'content': prompt
    }
    messages = [message]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
    max_tokens=2500)
    usage=response.usage
    print('prompt :',prompt, ' ---> your token',usage.prompt_tokens,'  completion_token : ',usage.completion_tokens,'  total_token  : ',usage.total_tokens,
          '  finish_reason :',response.choices[0].finish_reason)
