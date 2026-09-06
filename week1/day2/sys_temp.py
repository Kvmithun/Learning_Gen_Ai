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

prompt='suggest a single  name for my cloth comapany'

message_system={
    'role':'system',
    'content':'your brand manager who suggests name for my food brand'



}
message={
    'role':role,
    'content':prompt
}

messages=[message_system,message]
response=client.chat.completions.create(
    model=model,
    messages=messages,
    temperature=0.5

)


print(response.choices[0].message.content)
