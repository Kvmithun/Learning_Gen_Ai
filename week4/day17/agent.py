import ast
import json
import operator
import os
from urllib.parse import quote, urlparse

import requests
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template_string, request
from groq import Groq
from tavily import TavilyClient

# ============================================================
# CONFIG
# ============================================================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
GROQ_MODEL = "openai/gpt-oss-120b"

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY is missing from .env")
if not TAVILY_API_KEY:
    raise RuntimeError("TAVILY_API_KEY is missing from .env")

client = Groq(api_key=GROQ_API_KEY)
tavily = TavilyClient(api_key=TAVILY_API_KEY)
app = Flask(__name__)

REQUEST_HEADERS = {
    "User-Agent": "AI-Search-Agent/1.0 (educational project)"
}

# ============================================================
# TOOL 1: TAVILY WEB SEARCH
# ============================================================

def web_search(query: str):
    try:
        response = tavily.search(
            query=query,
            search_depth="advanced",
            max_results=6,
            include_answer=True,
        )

        results = []
        for result in response.get("results", []):
            results.append({
                "title": result.get("title", ""),
                "url": result.get("url", ""),
                "content": result.get("content", ""),
                "score": result.get("score", 0),
            })

        return {
            "success": True,
            "tool": "web_search",
            "ui_type": "web",
            "query": query,
            "answer": response.get("answer", ""),
            "results": results,
        }
    except Exception as exc:
        return {
            "success": False,
            "tool": "web_search",
            "ui_type": "web",
            "error": str(exc),
            "results": [],
        }

# ============================================================
# TOOL 2: CALCULATOR
# Safe AST-based calculator. No eval().
# ============================================================

BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _eval_math(node):
    if isinstance(node, ast.Expression):
        return _eval_math(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in BIN_OPS:
        left = _eval_math(node.left)
        right = _eval_math(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ValueError("Exponent is too large.")
        value = BIN_OPS[type(node.op)](left, right)
        if abs(value) > 1e100:
            raise ValueError("Result is too large.")
        return value
    if isinstance(node, ast.UnaryOp) and type(node.op) in UNARY_OPS:
        return UNARY_OPS[type(node.op)](_eval_math(node.operand))
    raise ValueError("Only basic arithmetic is supported.")


def calculator(expression: str):
    try:
        cleaned = expression.replace("^", "**").strip()
        if len(cleaned) > 200:
            raise ValueError("Expression is too long.")
        tree = ast.parse(cleaned, mode="eval")
        value = _eval_math(tree)
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        return {
            "success": True,
            "tool": "calculator",
            "ui_type": "calculator",
            "expression": expression,
            "result": value,
        }
    except Exception as exc:
        return {
            "success": False,
            "tool": "calculator",
            "ui_type": "calculator",
            "expression": expression,
            "error": str(exc),
        }

# ============================================================
# TOOL 3: WEATHER - Open-Meteo
# No API key required.
# ============================================================


def weather(location: str):
    try:
        geo = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": location, "count": 1, "language": "en", "format": "json"},
            headers=REQUEST_HEADERS,
            timeout=12,
        )
        geo.raise_for_status()
        geo_data = geo.json()
        places = geo_data.get("results", [])
        if not places:
            return {
                "success": False,
                "tool": "weather",
                "ui_type": "weather",
                "location": location,
                "error": f"Could not find location: {location}",
            }

        place = places[0]
        latitude = place["latitude"]
        longitude = place["longitude"]

        forecast = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": latitude,
                "longitude": longitude,
                "current": "temperature_2m,relative_humidity_2m,apparent_temperature,is_day,precipitation,weather_code,wind_speed_10m",
                "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max",
                "forecast_days": 5,
                "timezone": "auto",
            },
            headers=REQUEST_HEADERS,
            timeout=12,
        )
        forecast.raise_for_status()
        data = forecast.json()

        return {
            "success": True,
            "tool": "weather",
            "ui_type": "weather",
            "location": place.get("name", location),
            "country": place.get("country", ""),
            "latitude": latitude,
            "longitude": longitude,
            "timezone": data.get("timezone", ""),
            "current": data.get("current", {}),
            "daily": data.get("daily", {}),
        }
    except Exception as exc:
        return {
            "success": False,
            "tool": "weather",
            "ui_type": "weather",
            "location": location,
            "error": str(exc),
        }

# ============================================================
# TOOL 4: GITHUB SEARCH
# Public GitHub repository search. No API key required.
# ============================================================


def github_search(query: str):
    try:
        response = requests.get(
            "https://api.github.com/search/repositories",
            params={
                "q": query,
                "sort": "stars",
                "order": "desc",
                "per_page": 10,
            },
            headers={
                **REQUEST_HEADERS,
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2026-03-10",
            },
            timeout=12,
        )
        response.raise_for_status()
        data = response.json()

        repositories = []
        for item in data.get("items", []):
            repositories.append({
                "name": item.get("full_name", ""),
                "description": item.get("description") or "",
                "url": item.get("html_url", ""),
                "stars": item.get("stargazers_count", 0),
                "forks": item.get("forks_count", 0),
                "language": item.get("language") or "",
                "open_issues": item.get("open_issues_count", 0),
                "updated_at": item.get("updated_at", ""),
                "topics": item.get("topics", []),
                "owner": (item.get("owner") or {}).get("login", ""),
            })

        return {
            "success": True,
            "tool": "github_search",
            "ui_type": "github",
            "query": query,
            "total_count": data.get("total_count", 0),
            "repositories": repositories,
        }
    except requests.HTTPError as exc:
        detail = "GitHub API request failed."
        try:
            body = response.json()
            detail = body.get("message", detail)
        except Exception:
            pass
        return {
            "success": False,
            "tool": "github_search",
            "ui_type": "github",
            "query": query,
            "error": detail,
            "repositories": [],
        }
    except Exception as exc:
        return {
            "success": False,
            "tool": "github_search",
            "ui_type": "github",
            "query": query,
            "error": str(exc),
            "repositories": [],
        }

# ============================================================
# TOOL 5: WIKIPEDIA - MediaWiki REST API
# No API key required.
# ============================================================


def wikipedia_search(query: str):
    try:
        response = requests.get(
            "https://en.wikipedia.org/w/rest.php/v1/search/page",
            params={"q": query, "limit": 8},
            headers={**REQUEST_HEADERS, "Api-User-Agent": REQUEST_HEADERS["User-Agent"]},
            timeout=12,
        )
        response.raise_for_status()
        data = response.json()

        pages = []
        for page in data.get("pages", []):
            title = page.get("title", "")
            key = page.get("key", "")
            pages.append({
                "title": title,
                "description": page.get("description") or "",
                "excerpt": page.get("excerpt") or "",
                "thumbnail": (page.get("thumbnail") or {}).get("url", ""),
                "url": f"https://en.wikipedia.org/wiki/{quote(key or title.replace(' ', '_'))}",
            })

        return {
            "success": True,
            "tool": "wikipedia_search",
            "ui_type": "wikipedia",
            "query": query,
            "pages": pages,
        }
    except Exception as exc:
        return {
            "success": False,
            "tool": "wikipedia_search",
            "ui_type": "wikipedia",
            "query": query,
            "error": str(exc),
            "pages": [],
        }

# ============================================================
# GROQ TOOL DEFINITIONS
# ============================================================


tools = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the live internet using Tavily. Use this for current, latest, "
                "recent, news, research, technology, companies, products, people, "
                "websites, or general web information. Prefer this when the answer "
                "depends on live internet content."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "A clear web search query."}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": (
                "Perform mathematical calculations such as addition, subtraction, "
                "multiplication, division, percentages, powers, and arithmetic expressions. "
                "Use this instead of doing arithmetic mentally."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "A basic arithmetic expression."}
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "weather",
            "description": (
                "Get current weather and a short forecast for a location. Use this "
                "when the user asks about weather, temperature, rain, precipitation, "
                "humidity, wind, or forecast."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City or location name."}
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "github_search",
            "description": (
                "Search public GitHub repositories. Use this when the user asks "
                "to find repositories, open-source projects, libraries, frameworks, "
                "developer tools, GitHub projects, or code projects. Prefer this "
                "over general web search when the target is specifically GitHub."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to search for on GitHub."}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "wikipedia_search",
            "description": (
                "Search Wikipedia for encyclopedia articles. Use this when the user "
                "asks for an explanation, background, biography, history, definition, "
                "or factual encyclopedia information that is well suited to Wikipedia."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Topic to search on Wikipedia."}
                },
                "required": ["query"],
            },
        },
    },
]

# ============================================================
# TOOL EXECUTION
# ============================================================


def execute_tool(name, arguments):
    if name == "web_search":
        return web_search(arguments.get("query", ""))
    if name == "calculator":
        return calculator(arguments.get("expression", ""))
    if name == "weather":
        return weather(arguments.get("location", ""))
    if name == "github_search":
        return github_search(arguments.get("query", ""))
    if name == "wikipedia_search":
        return wikipedia_search(arguments.get("query", ""))
    return {"success": False, "error": f"Unknown tool: {name}"}

# ============================================================
# AGENT
# ============================================================

SYSTEM_PROMPT = """
You are the central router and assistant for a multi-tool AI interface.

Available tools:
- web_search: live internet search through Tavily.
- calculator: exact arithmetic calculations.
- weather: current weather and forecast for a location.
- github_search: search public GitHub repositories and open-source projects.
- wikipedia_search: search Wikipedia encyclopedia articles.

IMPORTANT TOOL-ROUTING RULES:
1. Decide automatically whether a tool is needed.
2. Use calculator for arithmetic instead of calculating mentally.
3. Use weather for weather/forecast questions.
4. Use github_search when the user specifically wants GitHub repositories, open-source projects, libraries, frameworks, or developer projects.
5. Use wikipedia_search when an encyclopedia-style Wikipedia lookup is appropriate.
6. Use web_search for current/latest/news/research/general live-web information.
7. Do not fabricate tool data.
8. After tool results arrive, answer using those results.
9. Keep the final answer useful and concise.
10. Do not mention internal tool calls or implementation details unless the user asks.

TOOL AVAILABILITY RESTRICTION:
11. Only handle requests that clearly match one of the available tools.
12. If the request does not match web search, calculator, weather, GitHub,
    or Wikipedia, DO NOT call any tool and reply exactly:
    "Tool unavailable: I don't have a tool for that request."
13. Do not pretend another tool exists.
14. If a request mixes supported and unsupported tasks, handle the supported
    part and clearly say which part is unavailable.
15. Do not repeatedly call the same tool with the same arguments unless the
    new result is needed to make progress.
16. Stop when you have enough information to answer.
"""


def run_agent(user_message):
    """
    Iterative agent loop.

    The model chooses tools automatically. After each tool result, the model
    gets another iteration to decide whether another action is required.

    MAX_ITERATIONS prevents an impossible/unstable task from looping forever,
    wasting tokens, time, API calls, and money.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    MAX_ITERATIONS = 6
    all_results = []
    primary_tool = "none"
    primary_ui = "chat"

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"[Agent] Iteration {iteration}/{MAX_ITERATIONS}")

        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=0.2,
        )

        assistant_message = response.choices[0].message

        # No tool call means the agent has finished.
        if not assistant_message.tool_calls:
            answer = assistant_message.content or ""

            # The system prompt tells the model to use this exact response
            # when the request does not match any available tool.
            return {
                "answer": answer,
                "tool": primary_tool,
                "ui_type": primary_ui,
                "results": all_results,
                "iterations": iteration,
                "max_iterations": MAX_ITERATIONS,
            }

        messages.append(assistant_message)

        for call in assistant_message.tool_calls:
            tool_name = call.function.name

            try:
                arguments = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                arguments = {}

            print(
                f"[Agent] Iteration {iteration}: "
                f"calling {tool_name} with {arguments}"
            )

            result = execute_tool(tool_name, arguments)
            all_results.append(result)

            if primary_tool == "none":
                primary_tool = tool_name
                primary_ui = result.get("ui_type", "chat")

            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": json.dumps(result),
            })

    # Hard stop: the agent used all allowed iterations.
    return {
        "answer": (
            "I couldn't complete this request within the maximum "
            f"of {MAX_ITERATIONS} agent iterations. "
            "The task may require more steps or may not be solvable "
            "with the available tools."
        ),
        "tool": primary_tool,
        "ui_type": primary_ui,
        "results": all_results,
        "iterations": MAX_ITERATIONS,
        "max_iterations": MAX_ITERATIONS,
        "stopped_reason": "max_iterations",
    }


# ============================================================
# UI
# ============================================================

HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AI Search</title>
<style>
*{box-sizing:border-box}
html,body{margin:0;min-height:100%;font-family:Inter,Arial,Helvetica,sans-serif;color:#202124;background:#fff}
button,input{font:inherit}
button{cursor:pointer}

.home-page{min-height:100vh;display:flex;flex-direction:column;align-items:center}
.nav{width:100%;height:64px;display:flex;justify-content:flex-end;align-items:center;gap:22px;padding:0 28px;font-size:14px}
.nav a{color:#202124;text-decoration:none}
.nav a:hover{text-decoration:underline}

.logo{margin-top:145px;font-size:72px;font-weight:500;letter-spacing:-5px;user-select:none}
.blue{color:#4285f4}.red{color:#ea4335}.yellow{color:#fbbc05}.green{color:#34a853}

.search-area{width:min(680px,92%);margin-top:32px;display:flex;flex-direction:column;align-items:center}
.search-box,.results-search{display:flex;align-items:center;border:1px solid #dfe1e5;border-radius:28px;transition:.2s;background:#fff}
.search-box{width:100%;height:54px;padding:0 18px}
.search-box:hover,.search-box:focus-within,.results-search:focus-within{box-shadow:0 1px 7px rgba(32,33,36,.18);border-color:transparent}
.search-icon{width:20px;height:20px;margin-right:12px;color:#9aa0a6;display:flex;align-items:center}
.search-icon svg{width:19px;height:19px}
.search-input{flex:1;border:0;outline:0;font-size:16px;color:#202124;background:transparent}
.clear-button{display:none;border:0;background:none;color:#5f6368;font-size:20px}
.search-input:not(:placeholder-shown)~.clear-button{display:block}

.buttons{margin-top:30px;display:flex;gap:12px}
.search-button{border:1px solid #f8f9fa;background:#f8f9fa;color:#202124;padding:10px 18px;border-radius:6px;font-size:14px}
.search-button:hover{border-color:#dadce0;box-shadow:0 1px 1px rgba(0,0,0,.1)}

.examples{margin-top:35px;text-align:center;color:#5f6368;font-size:13px}
.example{display:inline-block;margin:6px;padding:7px 13px;border:1px solid #dadce0;border-radius:18px;cursor:pointer;transition:.15s}
.example:hover{background:#f8f9fa;border-color:#c5c7ca}

.results-page{display:none;min-height:100vh;background:#fff}
.results-header{height:86px;border-bottom:1px solid #dadce0;display:flex;align-items:center;padding:0 28px;gap:28px;position:sticky;top:0;background:rgba(255,255,255,.96);backdrop-filter:blur(10px);z-index:5}
.small-logo{font-size:28px;font-weight:500;letter-spacing:-2px;cursor:pointer;white-space:nowrap}
.results-search{width:min(720px,72vw);height:46px;padding:0 16px;box-shadow:0 1px 4px rgba(32,33,36,.15)}
.results-search input{flex:1;border:0;outline:0;font-size:16px;background:transparent}
.results-search button{border:0;background:none;color:#4285f4}

.back-button{margin-left:auto;border:1px solid #dadce0;background:#fff;border-radius:22px;padding:8px 15px;color:#3c4043}
.back-button:hover{background:#f8f9fa}

.results-container{width:min(950px,92%);margin:0 auto;padding:28px 0 80px}
.result-count{color:#70757a;font-size:13px;margin-bottom:25px}
.loading{display:flex;align-items:center;gap:10px;color:#5f6368;padding:30px 0}
.spinner{width:18px;height:18px;border:2px solid #ddd;border-top-color:#4285f4;border-radius:50%;animation:spin .8s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}

.ai-answer{border:1px solid #dadce0;border-radius:18px;padding:22px 24px;margin-bottom:24px;background:#fff;box-shadow:0 2px 8px rgba(60,64,67,.08)}
.ai-label{font-size:14px;font-weight:600;margin-bottom:13px;display:flex;align-items:center;gap:8px}
.ai-icon{width:25px;height:25px;border-radius:50%;background:linear-gradient(135deg,#4285f4,#9b72cb);color:#fff;display:flex;align-items:center;justify-content:center;font-size:10px}
.ai-text{font-size:16px;line-height:1.65;white-space:pre-wrap}

.section-title{font-size:14px;font-weight:600;margin:25px 0 18px}
.result{margin-bottom:32px;animation:appear .3s ease}
@keyframes appear{from{opacity:0;transform:translateY(5px)}to{opacity:1;transform:translateY(0)}}
.result-site{display:flex;align-items:center;gap:8px;color:#202124;font-size:13px;margin-bottom:4px;overflow:hidden;white-space:nowrap}
.favicon{width:20px;height:20px;border-radius:50%;background:#f1f3f4;display:flex;align-items:center;justify-content:center;font-size:10px;overflow:hidden}
.favicon img{width:16px;height:16px}
.result-url{color:#5f6368;overflow:hidden;text-overflow:ellipsis}
.result-title{font-size:20px;color:#1a0dab;text-decoration:none;line-height:1.3;display:block;margin-bottom:5px}
.result-title:hover{text-decoration:underline}
.result-snippet{font-size:14px;line-height:1.55;color:#4d5156;max-width:800px}

.tool-card{border:1px solid #dadce0;border-radius:22px;background:#fff;box-shadow:0 3px 14px rgba(60,64,67,.10);overflow:hidden;margin-bottom:28px}
.tool-head{padding:19px 22px;border-bottom:1px solid #eee;display:flex;align-items:center;gap:10px}
.tool-head .emoji{font-size:25px}
.tool-head strong{font-size:17px}
.tool-body{padding:24px}
.muted{color:#6b7280;font-size:13px}

.app-toolbar{display:flex;align-items:center;justify-content:space-between;margin-bottom:18px}
.app-title{font-size:14px;color:#6b7280}
.app-back{border:1px solid #dadce0;background:#fff;border-radius:20px;padding:7px 13px;color:#3c4043}
.app-back:hover{background:#f8f9fa}

/* Calculator app */
.calc-app{max-width:460px;margin:0 auto}
.calc-display{border:1px solid #e1e4e8;border-radius:18px;padding:22px 20px;text-align:right;background:#fafbfc;min-height:120px}
.calc-expression{font-size:17px;color:#6b7280;min-height:25px;word-break:break-all}
.calc-result{font-size:48px;font-weight:650;margin-top:10px;word-break:break-all;line-height:1.05}
.calc-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-top:14px}
.calc-key{height:58px;border:1px solid #e1e4e8;background:#fff;border-radius:14px;font-size:18px}
.calc-key:hover{background:#f5f7f9}
.calc-key.equal{background:#4285f4;color:#fff;border-color:#4285f4}
.calc-key.equal:hover{background:#3367d6}
.calc-key.wide{grid-column:span 2}

/* Weather app */
.weather-app{max-width:820px;margin:0 auto}
.weather-hero{border:1px solid #e5e7eb;border-radius:22px;padding:25px;background:linear-gradient(135deg,#f7faff,#fff);display:flex;justify-content:space-between;gap:25px;align-items:center}
.weather-location{font-size:14px;color:#6b7280}
.weather-big{font-size:62px;font-weight:650;letter-spacing:-3px;margin:8px 0}
.weather-condition{font-size:17px;font-weight:600}
.weather-details{color:#5f6368;line-height:1.8;font-size:14px}
.weather-symbol{font-size:70px}
.forecast-grid{display:grid;grid-template-columns:repeat(5,1fr);gap:10px;margin-top:18px}
.forecast-day{border:1px solid #eee;border-radius:16px;padding:15px;text-align:center;font-size:13px;background:#fff}
.forecast-day b{display:block;margin:9px 0}
.forecast-icon{font-size:25px}

/* GitHub */
.github-list{display:flex;flex-direction:column;gap:14px}
.github-card{border:1px solid #e1e4e8;border-radius:15px;padding:18px;background:#fff}
.github-name{font-size:18px;font-weight:600;color:#0969da;text-decoration:none}
.github-name:hover{text-decoration:underline}
.github-main p{margin:8px 0 12px;color:#57606a;line-height:1.5;font-size:14px}
.github-meta{display:flex;flex-wrap:wrap;gap:14px;color:#57606a;font-size:13px}
.topics{display:flex;flex-wrap:wrap;gap:7px;margin-top:12px}
.topics span{background:#ddf4ff;color:#0969da;border-radius:999px;padding:4px 9px;font-size:12px}

/* Wikipedia */
.wiki-grid{display:grid;grid-template-columns:repeat(2,1fr);gap:16px}
.wiki-card{border:1px solid #eee;border-radius:15px;padding:16px;display:flex;gap:14px}
.wiki-card img{width:76px;height:76px;object-fit:cover;border-radius:10px;background:#f1f3f4}
.wiki-card h3{margin:0 0 7px;font-size:17px}
.wiki-card p{margin:0;color:#5f6368;font-size:13px;line-height:1.5}
.wiki-card a{text-decoration:none;color:inherit}

.error{color:#b42318;background:#fff1f0;border:1px solid #f5c2c0;padding:12px;border-radius:10px}
.empty{padding:40px 0;color:#70757a}
.iteration-info{font-size:12px;color:#8a8f98;margin-top:12px;text-align:right}

@media(max-width:800px){
.logo{margin-top:120px;font-size:58px}
.results-header{padding:0 16px;gap:15px}
.small-logo{display:none}
.results-search{width:100%}
.results-container{width:92%}
.forecast-grid,.wiki-grid{grid-template-columns:1fr 1fr}
.weather-hero{flex-direction:column;align-items:flex-start}
.back-button{display:none}
}
@media(max-width:500px){
.nav{padding-right:16px}
.logo{font-size:50px;letter-spacing:-4px}
.buttons{flex-wrap:wrap;justify-content:center}
.forecast-grid,.wiki-grid{grid-template-columns:1fr}
.calc-result{font-size:38px}
}
</style>
</head>
<body>

<div id="homePage" class="home-page">
    <div class="nav"><a href="#">About</a><a href="#">How it works</a></div>

    <div class="logo">
        <span class="blue">A</span><span class="red">I</span><span class="yellow">S</span><span class="blue">e</span><span class="green">a</span><span class="red">r</span><span class="blue">c</span><span class="yellow">h</span>
    </div>

    <div class="search-area">
        <div class="search-box">
            <div class="search-icon">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <circle cx="11" cy="11" r="7"/><path d="m20 20-4-4"/>
                </svg>
            </div>
            <input id="homeInput" class="search-input" placeholder="Ask anything..." autocomplete="off">
            <button class="clear-button" onclick="clearHomeInput()">×</button>
        </div>

        <div class="buttons">
            <button class="search-button" onclick="performSearch()">AI Search</button>
            <button class="search-button" onclick="performSearch()">I'm Feeling Lucky</button>
        </div>

        <div class="examples">
            Try searching:
            <div class="example" onclick="exampleSearch(this)">What's the weather in Bangalore?</div>
            <div class="example" onclick="exampleSearch(this)">Calculate 128 * 47</div>
            <div class="example" onclick="exampleSearch(this)">Find RAG projects on GitHub</div>
            <div class="example" onclick="exampleSearch(this)">Who is Alan Turing?</div>
            <div class="example" onclick="exampleSearch(this)">Latest AI news</div>
        </div>
    </div>
</div>

<div id="resultsPage" class="results-page">
    <div class="results-header">
        <div class="small-logo" onclick="goHome()">
            <span class="blue">A</span><span class="red">I</span><span class="yellow">S</span><span class="blue">e</span><span class="green">a</span><span class="red">r</span><span class="blue">c</span><span class="yellow">h</span>
        </div>

        <div class="results-search">
            <input id="resultsInput" autocomplete="off">
            <button onclick="performSearch()">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <circle cx="11" cy="11" r="7"/><path d="m20 20-4-4"/>
                </svg>
            </button>
        </div>

        <button class="back-button" onclick="goHome()">← Back</button>
    </div>

    <div class="results-container">
        <div id="resultCount" class="result-count"></div>

        <div id="loading" class="loading" style="display:none">
            <div class="spinner"></div><span id="loadingText">Thinking and choosing a tool...</span>
        </div>

        <div id="answerBox" class="ai-answer" style="display:none">
            <div class="ai-label"><div class="ai-icon">AI</div>AI Answer</div>
            <div id="answerText" class="ai-text"></div>
            <div id="iterationInfo" class="iteration-info"></div>
        </div>

        <div id="toolUI"></div>

        <div id="webSection" style="display:none">
            <div class="section-title">Web Results</div>
            <div id="results"></div>
        </div>

        <div id="empty" class="empty" style="display:none">No results found.</div>
    </div>
</div>

<script>
const homePage=document.getElementById('homePage');
const resultsPage=document.getElementById('resultsPage');
const homeInput=document.getElementById('homeInput');
const resultsInput=document.getElementById('resultsInput');
const resultsContainer=document.getElementById('results');
const answerBox=document.getElementById('answerBox');
const answerText=document.getElementById('answerText');
const loading=document.getElementById('loading');
const loadingText=document.getElementById('loadingText');
const webSection=document.getElementById('webSection');
const toolUI=document.getElementById('toolUI');
const resultCount=document.getElementById('resultCount');
const empty=document.getElementById('empty');
const iterationInfo=document.getElementById('iterationInfo');

homeInput.addEventListener('keydown',e=>{if(e.key==='Enter')performSearch()});
resultsInput.addEventListener('keydown',e=>{if(e.key==='Enter')performSearch()});

function clearHomeInput(){homeInput.value='';homeInput.focus()}
function exampleSearch(el){homeInput.value=el.textContent.trim();performSearch()}
function goHome(){
    resultsPage.style.display='none';
    homePage.style.display='flex';
    homeInput.focus();
    window.scrollTo({top:0,behavior:'smooth'});
}
function escapeHtml(v){
    return String(v??'').replaceAll('&','&amp;').replaceAll('<','&lt;')
        .replaceAll('>','&gt;').replaceAll('"','&quot;').replaceAll("'",'&#039;');
}
function weatherText(code){
    const m={0:'Clear sky',1:'Mainly clear',2:'Partly cloudy',3:'Overcast',45:'Fog',48:'Rime fog',
    51:'Light drizzle',53:'Drizzle',55:'Heavy drizzle',61:'Light rain',63:'Rain',65:'Heavy rain',
    71:'Light snow',73:'Snow',75:'Heavy snow',80:'Rain showers',81:'Rain showers',
    82:'Heavy rain showers',95:'Thunderstorm',96:'Thunderstorm with hail',99:'Thunderstorm with hail'};
    return m[code]||'Weather';
}
function weatherIcon(code){
    if(code===0)return '☀️';
    if([1,2].includes(code))return '⛅';
    if([3,45,48].includes(code))return '☁️';
    if([51,53,55,61,63,65,80,81,82].includes(code))return '🌧️';
    if([71,73,75].includes(code))return '❄️';
    if([95,96,99].includes(code))return '⛈️';
    return '🌤️';
}
function formatDate(s){
    try{return new Date(s+'T00:00:00').toLocaleDateString(undefined,{weekday:'short',month:'short',day:'numeric'})}
    catch{return s}
}

async function performSearch(){
    const query=(resultsPage.style.display==='block'?resultsInput.value:homeInput.value).trim();
    if(!query){homeInput.focus();return}

    homePage.style.display='none';
    resultsPage.style.display='block';
    resultsInput.value=query;

    resultsContainer.innerHTML='';
    toolUI.innerHTML='';
    answerText.textContent='';
    answerBox.style.display='none';
    webSection.style.display='none';
    empty.style.display='none';
    resultCount.textContent='';
    iterationInfo.textContent='';
    loadingText.textContent='AI is choosing the right tool...';
    loading.style.display='flex';
    window.scrollTo({top:0,behavior:'smooth'});

    try{
        const response=await fetch('/api/chat',{
            method:'POST',
            headers:{'Content-Type':'application/json'},
            body:JSON.stringify({message:query})
        });
        const data=await response.json();
        if(!response.ok)throw new Error(data.error||'Request failed.');

        loading.style.display='none';

        if(data.answer){
            answerText.textContent=data.answer;
            answerBox.style.display='block';
        }

        if(data.iterations){
            iterationInfo.textContent=`Agent iterations: ${data.iterations}/${data.max_iterations||6}`;
        }

        renderToolUI(data);
    }catch(err){
        loading.style.display='none';
        answerBox.style.display='block';
        answerText.innerHTML='<div class="error">'+escapeHtml(err.message)+'</div>';
    }
}

function renderToolUI(data){
    const result=(data.results||[]).find(x=>x&&x.success!==false)||(data.results||[])[0];

    if(!result){
        if(!data.answer)empty.style.display='block';
        return;
    }

    if(data.ui_type==='web'||result.ui_type==='web')renderWeb(result);
    else if(data.ui_type==='weather'||result.ui_type==='weather')renderWeather(result);
    else if(data.ui_type==='calculator'||result.ui_type==='calculator')renderCalculator(result);
    else if(data.ui_type==='github'||result.ui_type==='github')renderGitHub(result);
    else if(data.ui_type==='wikipedia'||result.ui_type==='wikipedia')renderWikipedia(result);
}

function appToolbar(title){
    return `<div class="app-toolbar">
        <div class="app-title">${escapeHtml(title)}</div>
        <button class="app-back" onclick="goHome()">← Back to Search</button>
    </div>`;
}

function renderWeb(result){
    const list=result.results||[];
    if(!list.length)return;
    webSection.style.display='block';
    resultCount.textContent=`About ${list.length} results`;
    list.forEach((r,i)=>addResult(r,i));
}

function addResult(r,index){
    const div=document.createElement('div');
    div.className='result';
    const url=r.url||'';
    let hostname='';
    try{hostname=new URL(url).hostname}catch{hostname=url}
    const favicon=hostname?`https://www.google.com/s2/favicons?domain=${encodeURIComponent(hostname)}&sz=32`:'';
    div.innerHTML=`<div class="result-site"><div class="favicon">${favicon?`<img src="${escapeHtml(favicon)}" onerror="this.style.display='none'">`:'🌐'}</div><div class="result-url">${escapeHtml(hostname)}</div></div><a class="result-title" href="${escapeHtml(url)}" target="_blank" rel="noopener noreferrer">${escapeHtml(r.title||'Untitled result')}</a><div class="result-snippet">${escapeHtml(r.content||'')}</div>`;
    resultsContainer.appendChild(div);
}

function renderCalculator(r){
    const resultText=r.success?String(r.result):r.error||'Calculation failed';
    toolUI.innerHTML=`<div class="tool-card"><div class="tool-body"><div class="calc-app">
        ${appToolbar('Calculator')}
        <div class="calc-display">
            <div class="calc-expression">${escapeHtml(r.expression||'')}</div>
            <div class="calc-result">${escapeHtml(resultText)}</div>
        </div>
        <div class="calc-grid">
            <button class="calc-key" onclick="calcKey('C')">C</button>
            <button class="calc-key" onclick="calcKey('(')">(</button>
            <button class="calc-key" onclick="calcKey(')')">)</button>
            <button class="calc-key" onclick="calcKey('/')">÷</button>
            <button class="calc-key" onclick="calcKey('7')">7</button>
            <button class="calc-key" onclick="calcKey('8')">8</button>
            <button class="calc-key" onclick="calcKey('9')">9</button>
            <button class="calc-key" onclick="calcKey('*')">×</button>
            <button class="calc-key" onclick="calcKey('4')">4</button>
            <button class="calc-key" onclick="calcKey('5')">5</button>
            <button class="calc-key" onclick="calcKey('6')">6</button>
            <button class="calc-key" onclick="calcKey('-')">−</button>
            <button class="calc-key" onclick="calcKey('1')">1</button>
            <button class="calc-key" onclick="calcKey('2')">2</button>
            <button class="calc-key" onclick="calcKey('3')">3</button>
            <button class="calc-key" onclick="calcKey('+')">+</button>
            <button class="calc-key wide" onclick="calcKey('0')">0</button>
            <button class="calc-key" onclick="calcKey('.')">.</button>
            <button class="calc-key equal" onclick="calculateFromUI()">=</button>
        </div>
    </div></div></div>`;
    window.calcExpression=r.expression||'';
}

let calcExpression='';
function calcKey(key){
    if(key==='C')calcExpression='';
    else calcExpression+=key;
    const el=document.querySelector('.calc-expression');
    const res=document.querySelector('.calc-result');
    if(el)el.textContent=calcExpression||'0';
    if(res)res.textContent='';
}
async function calculateFromUI(){
    if(!calcExpression.trim())return;
    const res=await fetch('/api/chat',{
        method:'POST',headers:{'Content-Type':'application/json'},
        body:JSON.stringify({message:'Calculate '+calcExpression})
    });
    const data=await res.json();
    const result=(data.results||[]).find(x=>x.ui_type==='calculator');
    const out=document.querySelector('.calc-result');
    if(out)out.textContent=result&&result.success?String(result.result):(result?.error||'Error');
}

function renderWeather(r){
    if(!r.success){
        toolUI.innerHTML=`<div class="tool-card"><div class="tool-body">${appToolbar('Weather')}<div class="error">${escapeHtml(r.error||'Weather lookup failed')}</div></div></div>`;
        return;
    }
    const c=r.current||{},d=r.daily||{};
    const times=d.time||[],max=d.temperature_2m_max||[],min=d.temperature_2m_min||[],rain=d.precipitation_probability_max||[],codes=d.weather_code||[];
    const cards=times.map((day,i)=>`<div class="forecast-day">
        <span>${escapeHtml(formatDate(day))}</span>
        <div class="forecast-icon">${weatherIcon(codes[i])}</div>
        <b>${escapeHtml(weatherText(codes[i]))}</b>
        <span>${Math.round(max[i]??0)}° / ${Math.round(min[i]??0)}°</span><br>
        <span class="muted">Rain ${rain[i]??0}%</span>
    </div>`).join('');

    toolUI.innerHTML=`<div class="tool-card"><div class="tool-body"><div class="weather-app">
        ${appToolbar('Weather')}
        <div class="weather-hero">
            <div>
                <div class="weather-location">${escapeHtml(r.location||'')} · ${escapeHtml(r.country||'')}</div>
                <div class="weather-big">${Math.round(c.temperature_2m??0)}°C</div>
                <div class="weather-condition">${weatherIcon(c.weather_code)} ${escapeHtml(weatherText(c.weather_code))}</div>
                <div class="weather-details">
                    Feels like ${Math.round(c.apparent_temperature??0)}°C ·
                    Humidity ${c.relative_humidity_2m??0}% ·
                    Wind ${Math.round(c.wind_speed_10m??0)} km/h ·
                    Precipitation ${c.precipitation??0} mm
                </div>
            </div>
            <div class="weather-symbol">${weatherIcon(c.weather_code)}</div>
        </div>
        <div class="forecast-grid">${cards}</div>
    </div></div></div>`;
}

function renderGitHub(r){
    const repos=r.repositories||[];
    if(!repos.length){
        toolUI.innerHTML=`<div class="tool-card"><div class="tool-body">${appToolbar('GitHub')}No repositories found.</div></div>`;
        return;
    }
    toolUI.innerHTML=`<div class="tool-card"><div class="tool-body">
        ${appToolbar('GitHub repositories')}
        <div class="github-list">${repos.map(repo=>`<div class="github-card">
            <a class="github-name" href="${escapeHtml(repo.url)}" target="_blank" rel="noopener noreferrer">${escapeHtml(repo.name)}</a>
            <p>${escapeHtml(repo.description||'No description available.')}</p>
            <div class="github-meta">
                <span>⭐ ${Number(repo.stars||0).toLocaleString()}</span>
                <span>🍴 ${Number(repo.forks||0).toLocaleString()}</span>
                ${repo.language?`<span>● ${escapeHtml(repo.language)}</span>`:''}
                <span>Issues ${Number(repo.open_issues||0).toLocaleString()}</span>
            </div>
            ${repo.topics?.length?`<div class="topics">${repo.topics.slice(0,6).map(t=>`<span>${escapeHtml(t)}</span>`).join('')}</div>`:''}
        </div>`).join('')}</div>
    </div></div>`;
}

function renderWikipedia(r){
    const pages=r.pages||[];
    if(!pages.length){
        toolUI.innerHTML=`<div class="tool-card"><div class="tool-body">${appToolbar('Wikipedia')}No articles found.</div></div>`;
        return;
    }
    toolUI.innerHTML=`<div class="tool-card"><div class="tool-body">
        ${appToolbar('Wikipedia')}
        <div class="wiki-grid">${pages.map(p=>`<div class="wiki-card">
            <a href="${escapeHtml(p.url)}" target="_blank" rel="noopener noreferrer">${p.thumbnail?`<img src="${escapeHtml(p.thumbnail)}" onerror="this.style.display='none'">`:''}</a>
            <div><a href="${escapeHtml(p.url)}" target="_blank" rel="noopener noreferrer"><h3>${escapeHtml(p.title)}</h3></a>
            <p>${escapeHtml(p.description||p.excerpt||'')}</p></div>
        </div>`).join('')}</div>
    </div></div>`;
}
</script>
</body>
</html>
"""

# ============================================================
# ROUTES
# ============================================================


@app.route("/")
def home():
    return render_template_string(HTML)


@app.route("/api/chat", methods=["POST"])
def chat_api():
    try:
        data = request.get_json(silent=True) or {}
        message = str(data.get("message", "")).strip()
        if not message:
            return jsonify({"error": "Message is required."}), 400
        return jsonify(run_agent(message))
    except Exception as exc:
        print("AGENT ERROR:", repr(exc))
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    print("=" * 60)
    print("AI SEARCH - MULTI TOOL AGENT")
    print("=" * 60)
    print("Model :", GROQ_MODEL)
    print("Tools : Tavily + Weather + Calculator + GitHub + Wikipedia")
    print("URL   : http://127.0.0.1:5002")
    print("=" * 60)
    app.run(host="0.0.0.0", port=5002, debug=True)
