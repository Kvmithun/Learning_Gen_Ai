import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq
from time import sleep

load_dotenv()

my_api_key=os.getenv('GROQ_API_KEY')

if not my_api_key:
    raise ValueError("API KEY IS NOT FOUND ")

client=Groq(api_key=my_api_key)

model='llama-3.3-70b-versatile'

JD='''
Junior Python Developer
Position: Junior Python Developer
Location: Bangalore
Requirements
Strong knowledge of Python
Basic knowledge of SQL
Familiarity with Git/GitHub
Basic understanding of REST APIs
Knowledge of OOP (Object-Oriented Programming)
Good problem-solving skills
Bachelor's degree in Computer Science or related field
Preferred Skills
Basic knowledge of Docker
Familiarity with AWS
Knowledge of FastAPI or Flask
Responsibilities
Develop and maintain Python applications
Write clean and reusable code
Fix bugs and optimize performance
Work with team members using Git
Participate in code reviews
'''

Resume='''
Mithun KV
+91 6361916073 | kvmithun1234@gmail.com | Bangalore, Karnataka | LinkedIn | GitHub
Summary
Computer Science undergraduate specializing in Data Science with hands-on experience in Python, Machine Learning, NLP
fundamentals, SQL, and Data Structures & Algorithms. Experienced in building AI-powered applications using TensorFlow,
Scikit-learn, Flask, and Streamlit, with practical experience in REST APIs, Git/GitHub, and end-to-end software development.
Education
Presidency University Aug 2023 – Aug 2027
Bachelor of Technology in Computer Science Engineering (Data Science) CGPA: 7.5 / 10
Technical Skills
Languages: Python, SQL
Machine Learning: Scikit-learn, TensorFlow, ANN, CNN, RNN, LSTM
Data Science: NumPy, Pandas, Matplotlib, Seaborn
Natural Language Processing: Tokenization, Text Preprocessing, Stemming, Lemmatization, Stopword Removal, POS
Tagging, Named Entity Recognition, Word2Vec
LLM Fundamentals: LLM API Calls, Prompt Structure, System/User Messages, Tokens, Temperature, Context Window
Backend: Flask, REST APIs
Databases: MySQL, MongoDB
Frontend: HTML, CSS
Developer Tools: Git, GitHub, VS Code, PyCharm, Jupyter Notebook
Core CS: Data Structures & Algorithms, OOP, DBMS, Operating Systems, Computer Networks
Experience
Freelance Full Stack Developer Jan 2026 – Feb 2026
QuizVeda (Private Client) Remote
• Developed RESTful backend services using Node.js, Express.js, and MongoDB for an online quiz registration platform.
• Integrated Razorpay Payment Gateway for secure payment processing and automated registration.
• Designed REST APIs with secure authentication and efficient database integration.
• Developed a responsive React frontend using Vite and Tailwind CSS.
Selected Projects
Multiple Disease Prediction System GitHub ↗
Tech Stack: Python • TensorFlow • Scikit-learn • Streamlit
• Developed multiple disease prediction models using TensorFlow and Scikit-learn.
• Performed data preprocessing, feature engineering, model training, and evaluation using Python.
• Built an interactive Streamlit application for real-time disease prediction.
Jan Suvidha Portal | 3rd Place Hackathon GitHub ↗
Tech Stack: Django • MongoDB • Gemini API • HTML • CSS • JavaScript
• Developed an AI-powered welfare assistance platform using Django, MongoDB, and Google Gemini API.
• Implemented multilingual support, voice interaction, and eligibility-based government scheme recommendations.
• Built an admin dashboard for document verification and welfare analytics.
MicroStream Pay GitHub ↗
Tech Stack: React • Flask • MongoDB • Algorand • JWT • Pera Wallet
• Developed a blockchain-enabled OTT streaming platform using React, Flask, MongoDB, and Algorand.

• Implemented JWT authentication and secure micropayment integration using Pera Wallet.
• Developed Flask REST APIs for streaming sessions, user management, and transaction tracking.
Achievements
• Secured 3rd Place in the Jan Suvidha Hackathon for developing an AI-powered welfare assistance platform.
• Selected for the Dayananda Sagar DevHack 2.0 National Hackathon.
 '''


def ask_llm(system_prompt,user_prompt):
    sys_msg={
    'role':'system',
    'content':system_prompt
}
    user_msg={
        'role': 'user',
        'content': user_prompt
    }

    messages=[sys_msg,user_msg]
    response=client.chat.completions.create(model=model,messages=messages)
    answer=response.choices[0].message.content
    return answer



def step1_resume(Resume):
    #extract skills from resume
    system_prompt='''
    You are proffesional HR assistant extract the skills from the candidates resume provided 
     Only return skills no other information .Do not invent any extra skills by yourself'''

    user_prompt=f"""
    extract skills from this Resume 
    {Resume}
    """
    return ask_llm(system_prompt,user_prompt)

def step2_JD_extract(JD):
    #extract skills from JD
    system_prompt='''
    You are proffesional HR assistant extract the skills from the  provided JD 
     Only return skills no other information .Do not invent any extra skills by yourself
     output format:
     skills should be separated from commas ,just return comma separated skills do not return any other filler information 
     '''

    user_prompt=f"""
    extract skills from this JD
    {JD}
    """
    return ask_llm(system_prompt,user_prompt)


def step3_Match(candidate,jd):
    system_prompt = '''
       You are a professional HR assistant.

Compare the candidate's skills with the skills required in the Job Description (JD).

Your tasks are:
1. Calculate a match score from 1 to 100.
2. List the matching skills (skills present in both the candidate's resume and the JD).
3. List the missing skills (skills required in the JD but not present in the candidate's resume).
4. List the additional skills (skills present in the candidate's resume but not required in the JD).
5. Identify the candidate's strengths.
6. Identify the candidate's weaknesses.
7. Provide a short verdict stating whether the candidate is a good fit for the role.

Return the output in the following format:

Match Score: XX/100

Matching Skills:
- ...

Missing Skills:
- ...

Additional Skills:
- ...

Strengths:
- ...

Weaknesses:
- ...

Verdict:
...'''

    user_prompt = f"""
        compare and match the skills in 
        jd:
        {jd}
        candidate:
        {candidate}
        """
    return ask_llm(system_prompt, user_prompt)

candidate=step1_resume(Resume)
sleep(2)
jd=step2_JD_extract(JD)
sleep(2)
score=step3_Match(candidate,jd)
sleep(2)
print(score)