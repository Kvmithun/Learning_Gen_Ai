import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

from pydantic import BaseModel
load_dotenv()

my_api_key=os.getenv("GROQ_API_KEY")
if not my_api_key:
    raise ValueError("No api key")
client=Groq(api_key=my_api_key)

model='llama-3.3-70b-versatile'
text=''''**Complaint Form**
**Name:** Rahul Sharma
**Address:** 42, Green Park Road, Sector 8, Bengaluru, Karnataka – 560037
**Email:** [rahul.sharma@example.com](mailto:rahul.sharma@example.com)
**Phone Number:** +91 98765 43210
**Subject:** Complaint Regarding Defective Product
**Complaint:**
I purchased a wireless Bluetooth headset from your online store on 5 July 2026. After only three days of normal use, the device stopped charging and no longer powers on.
I have followed all the troubleshooting steps mentioned in the user manual, including trying a different charging cable and adapter, but the issue persists.
I request that the product be inspected and that I receive either a replacement or a full refund under the warranty policy. I have attached a copy of the purchase invoice for reference.
I would appreciate a response within the next seven business days.
Thank you.
**Sincerely,**
Rahul Sharma
'''

class Ticket(BaseModel):
    name:str
    email:str
    ph_no:int
    issue:str

schema=Ticket.model_json_schema()
response_format={
    'type':'json_object'
}

system_prompt= (f""" extract the personal information from this ticket 
                strictly based on this schema{schema} and give me a json output""")
message_system={
    'role':'system',
    'content':system_prompt
}
role='user'

prompt=f"""
this is the cutomer complaint letter.please extract personal information and issue 
 from this {text} """

message={
    'role':role,
    'content':prompt
}

messages=[message_system,message]
response=client.chat.completions.create(
    model=model,
    messages=messages,
    response_format=response_format
)


answer=response.choices[0].message.content
print(answer)


# how to read json


import json
raw_json=answer
data_file=json.loads(raw_json)
ticket=Ticket(**data_file)

print(ticket.name)
print(ticket.email)
print(ticket.ph_no)
print(ticket.issue)
