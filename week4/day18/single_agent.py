import os
import json
import ast
import operator
from dotenv import load_dotenv
from groq import Groq
from tavily import TavilyClient

# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env")

if not TAVILY_API_KEY:
    raise ValueError("TAVILY_API_KEY not found in .env")


# ============================================================
# CLIENTS
# ============================================================

groq_client = Groq(api_key=GROQ_API_KEY)
tavily_client = TavilyClient(api_key=TAVILY_API_KEY)

MODEL = "openai/gpt-oss-120b"


# ============================================================
# TOKEN TRACKER
# ============================================================

total_prompt_tokens = 0
total_completion_tokens = 0
total_tokens = 0


def track_usage(response, label):
    global total_prompt_tokens
    global total_completion_tokens
    global total_tokens

    usage = response.usage

    prompt = usage.prompt_tokens
    completion = usage.completion_tokens
    total = usage.total_tokens

    total_prompt_tokens += prompt
    total_completion_tokens += completion
    total_tokens += total

    print("\n" + "=" * 60)
    print(f"TOKEN USAGE - {label}")
    print("=" * 60)
    print(f"Prompt tokens     : {prompt}")
    print(f"Completion tokens : {completion}")
    print(f"Total tokens      : {total}")
    print("=" * 60)


# ============================================================
# TOOL 1: TAVILY
# ============================================================

def tavily_search(query: str):
    print(f"\n[TAVILY] Searching: {query}")

    result = tavily_client.search(
        query=query,
        max_results=5
    )

    return result


# ============================================================
# TOOL 2: CALCULATOR
# ============================================================

allowed_operators = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
}


def safe_calculate(expression: str):

    def evaluate(node):

        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError("Invalid number")

        if isinstance(node, ast.BinOp):
            left = evaluate(node.left)
            right = evaluate(node.right)

            operation = allowed_operators[type(node.op)]

            return operation(left, right)

        if isinstance(node, ast.UnaryOp):
            operation = allowed_operators[type(node.op)]
            return operation(evaluate(node.operand))

        raise ValueError("Unsupported expression")

    tree = ast.parse(expression, mode="eval")

    return evaluate(tree.body)


# ============================================================
# TOOL DEFINITIONS FOR GROQ
# ============================================================

tools = [
    {
        "type": "function",
        "function": {
            "name": "tavily_search",
            "description": (
                "Search the web for current or external information. "
                "Use this when the user asks for information that "
                "requires web search."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": (
                "Perform mathematical calculations."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression"
                    }
                },
                "required": ["expression"]
            }
        }
    }
]


# ============================================================
# SINGLE AGENT
# ============================================================

SYSTEM_PROMPT = """
You are a single intelligent agent.

You have access to two tools:

1. tavily_search
   - Used for web/current information.

2. calculator
   - Used for mathematical calculations.

Decide yourself whether a tool is necessary.

You can:
- use no tool
- use one tool
- use multiple tools

After receiving tool results, answer the user clearly.

Do not explain internal tool-calling details unless asked.
"""


def run_single_agent(user_query):

    global total_prompt_tokens
    global total_completion_tokens
    global total_tokens

    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": user_query
        }
    ]

    print("\n[SINGLE AGENT]")
    print(f"User: {user_query}")

    while True:

        response = groq_client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )

        track_usage(response, "Single Agent LLM Call")

        message = response.choices[0].message

        # ----------------------------------------------------
        # No tool call -> final answer
        # ----------------------------------------------------

        if not message.tool_calls:

            print("\nANSWER")
            print("-" * 60)
            print(message.content)

            break

        # Add assistant tool-call message
        messages.append(message)

        # ----------------------------------------------------
        # Execute tools
        # ----------------------------------------------------

        for tool_call in message.tool_calls:

            function_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)

            print(f"\n[SINGLE AGENT DECISION] -> {function_name}")

            if function_name == "tavily_search":

                result = tavily_search(
                    arguments["query"]
                )

            elif function_name == "calculator":

                result = safe_calculate(
                    arguments["expression"]
                )

            else:

                result = "Unknown tool"

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result)
                }
            )

    # --------------------------------------------------------
    # FINAL TOKEN REPORT
    # --------------------------------------------------------

    print("\n")
    print("=" * 60)
    print("SINGLE AGENT FINAL TOKEN REPORT")
    print("=" * 60)

    print(f"Total prompt tokens     : {total_prompt_tokens}")
    print(f"Total completion tokens : {total_completion_tokens}")
    print(f"TOTAL TOKENS            : {total_tokens}")

    print("=" * 60)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    query = input("\nAsk something: ")

    run_single_agent(query)