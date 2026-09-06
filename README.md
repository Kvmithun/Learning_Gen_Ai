


🧠 Learning Generative AI
A hands-on journey through Generative AI, LLM applications, RAG, embeddings, agents, evaluation, and production AI systems.

This repository contains my learning notes, experiments, implementations, and mini-projects while progressing from the fundamentals of LLMs to building practical AI applications.

Goal: Understand the concepts deeply, implement them from scratch where possible, and gradually move toward production-ready Generative AI systems.

🚀 What I'm Learning
The repository is organized as a progressive learning path:

Generative AI
│
├── 01. Foundations
│   ├── Python for AI
│   ├── APIs & HTTP
│   ├── JSON / Pydantic
│   └── Environment & Project Setup
│
├── 02. LLM Fundamentals
│   ├── Tokens
│   ├── Context Windows
│   ├── Temperature
│   ├── System / User / Assistant Messages
│   ├── LLM APIs
│   ├── Streaming
│   └── Structured Outputs
│
├── 03. Prompt Engineering
│   ├── Zero-shot Prompting
│   ├── Few-shot Prompting
│   ├── Role Prompting
│   ├── Chain-of-Thought Concepts
│   ├── Prompt Templates
│   ├── Prompt Chaining
│   └── ReAct-style Reasoning
│
├── 04. Embeddings
│   ├── What are Embeddings?
│   ├── Semantic Similarity
│   ├── Cosine Similarity
│   ├── Vector Representations
│   └── Embedding Models
│
├── 05. RAG
│   ├── Document Loading
│   ├── Text Extraction
│   ├── Cleaning
│   ├── Chunking
│   ├── Overlap
│   ├── Embedding
│   ├── Vector Storage
│   ├── Retrieval
│   ├── Reranking
│   └── Context Construction
│
├── 06. Vector Databases
│   ├── Vector Search
│   ├── Metadata Filtering
│   ├── Similarity Search
│   ├── Chroma
│   ├── FAISS
│   ├── Qdrant
│   └── Hybrid Retrieval
│
├── 07. RAG Evaluation
│   ├── Retrieval Evaluation
│   ├── Answer Evaluation
│   ├── Faithfulness
│   ├── Context Relevance
│   ├── Context Recall
│   ├── Answer Relevance
│   ├── Precision / Recall
│   └── RAGAS-style Evaluation
│
├── 08. Agents
│   ├── Tool Calling
│   ├── Function Calling
│   ├── Tool Selection
│   ├── Agent Loops
│   ├── Iteration Limits
│   ├── Memory
│   ├── Planning
│   └── Multi-tool Agents
│
├── 09. AI Application Engineering
│   ├── Flask / FastAPI
│   ├── Web Interfaces
│   ├── Streaming Responses
│   ├── Authentication
│   ├── Error Handling
│   ├── Logging
│   └── Environment Management
│
└── 10. Production GenAI
    ├── Observability
    ├── Tracing
    ├── Cost Optimization
    ├── Latency Optimization
    ├── Guardrails
    ├── Security
    ├── Deployment
    └── Monitoring
📚 Learning Roadmap
1. Foundations
Building the programming and API foundations required for GenAI development.

Python

Virtual environments

uv

Project structure

Environment variables

.env

HTTP requests

REST APIs

JSON

Pydantic

Error handling

Git & GitHub

2. LLM Fundamentals
Understanding what happens when an application communicates with an LLM.

Core concepts
Tokens

Tokenization

Context window

Input vs output tokens

System prompts

User messages

Assistant messages

Temperature

Top-p

Model selection

Latency

Streaming

API usage

Rate limits

Cost

Structured Generation
JSON outputs

Pydantic models

Schema validation

Structured extraction

Reliable application outputs

✍️ Prompt Engineering
Learning how to communicate with LLMs effectively.

Topics include:

Zero-shot prompting

Few-shot prompting

Role prompting

Instruction hierarchy

Prompt templates

Prompt chaining

Context injection

Output constraints

Structured prompts

ReAct-style workflows

Prompt optimization

The goal is not just to write longer prompts, but to understand how prompt structure affects model behavior and reliability.

🔢 Embeddings
Embeddings convert text into numerical vectors that represent semantic meaning.

"Machine learning"
        │
        ▼
   Embedding Model
        │
        ▼
[0.12, -0.43, 0.87, ...]
        │
        ▼
     Vector Space
Concepts
Dense vectors

Semantic representation

Vector dimensions

Similarity

Cosine similarity

Euclidean distance

Dot product

Embedding models

Query embeddings

Document embeddings

Why embeddings matter
Keyword search asks:

"Does this document contain these words?"

Semantic search asks:

"Does this document mean something similar to my query?"

✂️ Chunking
Large documents cannot always be passed directly to an LLM.

Chunking divides documents into smaller pieces before embedding and retrieval.

Document
   │
   ▼
Cleaning
   │
   ▼
Chunking
   │
   ├── Chunk 1
   ├── Chunk 2
   ├── Chunk 3
   └── Chunk 4
        │
        ▼
    Embeddings
        │
        ▼
   Vector Database
Chunking concepts
Fixed-size chunking

Character chunking

Token-based chunking

Sentence chunking

Paragraph chunking

Recursive chunking

Chunk overlap

Semantic chunking

Metadata preservation

Important trade-off
Small chunks can improve retrieval precision but may lose context.

Large chunks preserve context but can introduce irrelevant information.

A good RAG system therefore treats chunk size and overlap as retrieval design decisions, not arbitrary constants.

📖 RAG — Retrieval-Augmented Generation
RAG combines retrieval with generation.

Instead of asking the LLM to answer only from its internal knowledge:

User Question
      │
      ▼
   Retriever
      │
      ▼
Relevant Documents
      │
      ▼
     LLM
      │
      ▼
    Answer
Complete RAG Pipeline
                 OFFLINE / INDEXING
                 ──────────────────

Documents
   │
   ▼
Load
   │
   ▼
Clean
   │
   ▼
Chunk
   │
   ▼
Embed
   │
   ▼
Vector Database
   │
   │
   │
   ▼
                 ONLINE / QUERY TIME

User Query
   │
   ▼
Query Embedding
   │
   ▼
Retriever
   │
   ▼
Top-K Documents
   │
   ▼
Reranker
   │
   ▼
Relevant Context
   │
   ▼
Prompt Construction
   │
   ▼
LLM
   │
   ▼
Grounded Answer
RAG topics
Naive RAG

Semantic retrieval

Top-K retrieval

Metadata filtering

Hybrid search

Keyword + vector search

Reranking

Query rewriting

Multi-query retrieval

HyDE

Context compression

Parent-child retrieval

Self-query retrieval

Citation generation

Retrieval failure handling

🗄️ Vector Databases
Learning how embeddings are stored and searched efficiently.

Technologies explored / to explore:

FAISS

Chroma

Qdrant

Other vector stores

Concepts
Collections

Vectors

Metadata

Similarity search

Top-K

Filtering

Indexing

Persistence

Approximate nearest-neighbor search

Hybrid retrieval

🧪 RAG Evaluation
A RAG system should not be considered good simply because it produces fluent answers.

Evaluation needs to measure retrieval quality and generation quality separately.

Retrieval Evaluation
Questions include:

Did we retrieve the correct document?

Did we retrieve enough relevant context?

Are irrelevant chunks being retrieved?

Is the correct information ranked highly?

Metrics/concepts:

Precision

Recall

Hit Rate

Recall@K

Precision@K

MRR

NDCG

Generation Evaluation
Questions include:

Is the answer relevant?

Is it supported by retrieved context?

Is the model hallucinating?

Does the answer actually answer the question?

Important dimensions:

Faithfulness

Answer relevance

Context relevance

Context recall

Groundedness

Evaluation mindset
RAG Quality
    │
    ├── Retrieval Quality
    │      ├── Did we find it?
    │      └── Did we rank it correctly?
    │
    └── Generation Quality
           ├── Did we use the context?
           ├── Is the answer relevant?
           └── Did we hallucinate?
🤖 AI Agents
Moving from simple LLM calls to systems that can decide when and how to use tools.

Basic agent loop:

User
 │
 ▼
LLM
 │
 ├── Answer directly
 │
 └── Call Tool
       │
       ▼
    Tool Result
       │
       ▼
      LLM
       │
       ▼
    Final Answer
Agent concepts
Tool calling

Function calling

Tool schemas

Tool descriptions

Automatic tool selection

Tool execution

Tool results

Agent loops

Planning

Memory

Multi-tool agents

Error recovery

Termination conditions

Iteration limits

Bounded Agent Iterations
Agents need a termination condition.

MAX_ITERATIONS = 6
This prevents an agent from repeatedly calling tools when a task is impossible or when the model gets stuck.

Benefits:

Prevents infinite loops

Controls token usage

Reduces latency

Limits API calls

Controls cost

🛠️ Tools & APIs
The learning projects use practical APIs and developer tools.

Examples include:

LLM APIs

Web search

Weather APIs

GitHub REST API

Wikipedia API

Calculator tools

Vector databases

Embedding APIs

The objective is to understand how external tools are connected to an LLM rather than treating the LLM as the entire application.

🌐 AI Application Development
Building interfaces around AI systems.

Areas covered:

Flask

FastAPI

HTML / CSS / JavaScript

Chat interfaces

Streaming UI

Tool-specific UI

Loading states

Error states

API routes

JSON responses

Frontend/backend communication

Example architecture:

Browser
   │
   ▼
Frontend
   │
   ▼
Backend API
   │
   ▼
LLM
   │
   ├── Web Search
   ├── Weather
   ├── GitHub
   ├── Wikipedia
   └── Calculator
🔍 Observability
As AI applications become more complex, debugging requires visibility into every step.

Learning areas:

Logging

Tracing

Latency measurement

Token usage

Tool-call tracking

Retrieval traces

Error tracking

Prompt inspection

Evaluation datasets

💰 Cost & Performance Optimization
Important production considerations:

Token reduction

Context compression

Smaller models where appropriate

Caching

Embedding reuse

Efficient chunking

Limiting retrieved documents

Tool-call limits

Streaming

Rate-limit handling

Retry strategies

The objective is:

Better Quality
     +
Lower Cost
     +
Lower Latency
     =
Better AI Application
🔐 AI Security & Reliability
Topics to explore:

API key protection

.env

Secret management

Prompt injection

Indirect prompt injection

Tool abuse

Input validation

Output validation

Rate limiting

Authentication

Authorization

Data leakage

Safe tool execution

Guardrails

Never commit secrets such as API keys or .env files to GitHub.

🏗️ Project Structure
The repository is organized around learning days/weeks and practical implementations.

learning_ai/
│
├── week1/
│   ├── day1/
│   ├── day2/
│   ├── day3/
│   └── day4/
│
├── week2/
│   ├── day6/
│   ├── day7/
│   ├── day8/
│   └── day9/
│
├── week4/
│   └── day17/
│       ├── agent.py
│       ├── README.md
│       ├── pyproject.toml
│       ├── uv.lock
│       └── .gitignore
│
└── README.md
Each day contains focused experiments, notes, and implementations.

🧩 Current Agent Project
One of the projects in this repository is a multi-tool AI agent.

The agent can reason about a user's request and select an appropriate tool.

Example:

"What is the weather in Bangalore?"
             │
             ▼
          LLM
             │
             ▼
       Weather Tool
             │
             ▼
        Weather UI
Another example:

"Find popular RAG projects on GitHub"
             │
             ▼
          LLM
             │
             ▼
       GitHub Tool
             │
             ▼
      Repository UI
The system is designed around tool descriptions + automatic tool selection, rather than manually routing every query with frontend keyword checks.

📈 Learning Progress
This checklist will evolve as the repository grows.

Foundations
Python fundamentals

API basics

JSON / Pydantic

Environment variables

Git / GitHub

uv project management

LLM Fundamentals
Tokens

Context windows

Temperature

Streaming

Structured outputs

Function calling

Prompt Engineering
Zero-shot

Few-shot

Prompt templates

Prompt chaining

ReAct

Embeddings
Embedding fundamentals

Cosine similarity

Semantic search

Embedding models

RAG
Document loading

Document cleaning

Chunking

Chunk overlap

Embeddings

Vector database

Retrieval

Reranking

Query rewriting

Hybrid search

Advanced RAG

RAG Evaluation
Retrieval evaluation

Precision@K

Recall@K

MRR

NDCG

Faithfulness

Answer relevance

Context relevance

RAG evaluation framework

Agents
Tool-based agent

Automatic tool selection

Multi-tool architecture

Tool-specific UI

Iteration limit

Memory

Planning

Multi-agent systems

Production
Observability

Tracing

Guardrails

Prompt injection defense

Evaluation pipeline

Cost optimization

Deployment

Monitoring

🎯 End Goal
The long-term goal of this repository is to progress from:

LLM API
   ↓
Prompt Engineering
   ↓
Structured Outputs
   ↓
Embeddings
   ↓
RAG
   ↓
RAG Evaluation
   ↓
Tool Calling
   ↓
Agents
   ↓
Production AI Systems
and eventually build complete AI applications that are:

Useful

Grounded

Evaluated

Observable

Reliable

Cost-efficient

Production-ready

🧪 Philosophy
Don't just learn frameworks. Understand what is happening underneath them.

For every abstraction, the goal is to understand:

What problem does it solve?

How does it work?

What happens internally?

What are its limitations?

How can it be evaluated?

When should it be used?

When should it not be used?

📌 Repository
Learning Generative AI — from fundamentals to AI agents and production RAG systems.

Built through continuous experimentation, implementation, debugging, and evaluation
