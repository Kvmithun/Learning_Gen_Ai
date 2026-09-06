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

prompt='Explain how API works '

message={
    'role':role,
    'content':prompt
}

messages=[message]

#Without Streaming 
# response=client.chat.completions.create(
#     model=model,
#     messages=messages
# )


# print(response.choices[0].message.content)


stream=client.chat.completions.create(
    model=model,
    messages=messages,
    stream=True
 )
for chunk in stream :
    content=chunk.choices[0].delta.content # we are extracting content from this chunk 
    if content:
        print(content,end='',flush=True)