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


def track_usage(response, agent_name):

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
    print(f"TOKEN USAGE - {agent_name}")
    print("=" * 60)

    print(f"Prompt tokens     : {prompt}")
    print(f"Completion tokens : {completion}")
    print(f"Total tokens      : {total}")

    print("=" * 60)


# ============================================================
# TOOL 1
# ============================================================

def tavily_agent(query):

    print("\n[TAVILY AGENT]")
    print(f"Searching: {query}")

    result = tavily_client.search(
        query=query,
        max_results=5
    )

    return result


# ============================================================
# TOOL 2
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


def calculator_agent(expression):

    print("\n[CALCULATOR AGENT]")
    print(f"Calculating: {expression}")

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

            return operation(
                evaluate(node.operand)
            )

        raise ValueError("Unsupported expression")

    tree = ast.parse(
        expression,
        mode="eval"
    )

    return evaluate(tree.body)


# ============================================================
# MANAGER TOOLS
# ============================================================

manager_tools = [
    {
        "type": "function",
        "function": {
            "name": "delegate_to_tavily",
            "description": (
                "Delegate the task to the Tavily research agent "
                "when web/current information is required."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "delegate_to_calculator",
            "description": (
                "Delegate mathematical calculations to the "
                "calculator agent."
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
# MANAGER
# ============================================================

MANAGER_PROMPT = """
You are the Manager / Orchestrator of a multi-agent system.

You control two specialized agents:

1. Tavily Agent
   - Searches the web.
   - Use for current/external information.

2. Calculator Agent
   - Performs mathematical calculations.
   - Use for arithmetic.

Your responsibilities:

- Understand the user's request.
- Decide which agent is required.
- Decide whether no agent is required.
- You may call one agent.
- You may call both agents.
- Do not call an agent unnecessarily.
- After receiving the agent result, produce the final answer.

You are NOT the tool itself.
You are the manager who delegates work.
"""


# ============================================================
# MULTI-AGENT SYSTEM
# ============================================================

def run_multi_agent(user_query):

    global total_prompt_tokens
    global total_completion_tokens
    global total_tokens

    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0

    print("\n")
    print("=" * 60)
    print("MULTI-AGENT SYSTEM")
    print("=" * 60)

    print(f"User: {user_query}")

    messages = [
        {
            "role": "system",
            "content": MANAGER_PROMPT
        },
        {
            "role": "user",
            "content": user_query
        }
    ]

    while True:

        # ====================================================
        # MANAGER LLM
        # ====================================================

        response = groq_client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=manager_tools,
            tool_choice="auto"
        )

        track_usage(
            response,
            "MANAGER / ORCHESTRATOR"
        )

        manager_message = response.choices[0].message

        # ----------------------------------------------------
        # MANAGER HAS FINAL ANSWER
        # ----------------------------------------------------

        if not manager_message.tool_calls:

            print("\nFINAL ANSWER")
            print("-" * 60)
            print(manager_message.content)

            break

        messages.append(manager_message)

        # ====================================================
        # MANAGER DELEGATES
        # ====================================================

        for tool_call in manager_message.tool_calls:

            function_name = tool_call.function.name

            arguments = json.loads(
                tool_call.function.arguments
            )

            # ------------------------------------------------
            # TAVILY AGENT
            # ------------------------------------------------

            if function_name == "delegate_to_tavily":

                print("\nMANAGER DECISION")
                print("-> Delegate to Tavily Agent")

                result = tavily_agent(
                    arguments["query"]
                )

            # ------------------------------------------------
            # CALCULATOR AGENT
            # ------------------------------------------------

            elif function_name == "delegate_to_calculator":

                print("\nMANAGER DECISION")
                print("-> Delegate to Calculator Agent")

                result = calculator_agent(
                    arguments["expression"]
                )

            else:

                result = "Unknown delegation"

            # ------------------------------------------------
            # SEND RESULT BACK TO MANAGER
            # ------------------------------------------------

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result)
                }
            )

    # ========================================================
    # FINAL TOKEN REPORT
    # ========================================================

    print("\n")
    print("=" * 60)
    print("MULTI-AGENT FINAL TOKEN REPORT")
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

    run_multi_agent(query)