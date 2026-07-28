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

prompt='do u know ms dhoni an what is ai and rag and all how a new student can start ai and suggest any indian you tube channels to start my journey and what is ur opinion on campusx'

message={
    'role':role,
    'content':prompt
}

messages=[message]
response=client.chat.completions.create(
    model=model,
    messages=messages
)


print(response.choices[0].message.content)
