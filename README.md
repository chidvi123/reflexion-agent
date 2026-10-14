# Reflexion Agent: Research, Search, and Revision with LangGraph

This project demonstrates a research workflow built with **LangChain, LangGraph, Groq, Pydantic, and Tavily**.

The graph is intended to:
1. Generate an initial research answer.
2. Ask the model for search queries through a structured tool call.
3. Execute those searches with Tavily.
4. Ask the model to revise its answer using the search results.
5. Repeat the search-and-revision cycle until the iteration limit is reached.

> Important: the current code has a few workflow issues noted below. This README explains what the code currently does and what should be corrected.

## 1. Project structure

```text
ReflexionAgent/
├── .env
├── main.py
├── chains.py
├── schemas.py
├── tool_executor.py
├── pyproject.toml
├── poetry.lock
└── graph.png
```

- `main.py`: defines and runs the LangGraph workflow.
- `chains.py`: defines prompts, the Groq model, and chains for the initial answer and revision.
- `schemas.py`: defines Pydantic schemas used to describe tool-call arguments.
- `tool_executor.py`: defines the Tavily search tool and LangGraph `ToolNode`.
- `.env`: stores API keys. Do not commit this file to GitHub.

## 2. Dependencies and environment

Install the dependencies with Poetry:

```powershell
poetry add python-dotenv langchain-core langchain-groq langgraph langchain-tavily pydantic
```

If they are already declared in `pyproject.toml`, install the locked dependencies instead:

```powershell
poetry install
```

Put your keys in `.env`, using the variable names expected by the integrations:

```dotenv
GROQ_API_KEY=your_groq_api_key
TAVILY_API_KEY=your_tavily_api_key
```

Never paste real API keys into source code or commit `.env`.

Run the project with:

```powershell
poetry run python main.py
```

## 3. `schemas.py`: defining structured output

```python
from typing import List
from pydantic import BaseModel, Field


class Reflection(BaseModel):
    missing: str = Field(
        description="Critique of what important information is missing."
    )
    superfluous: str = Field(
        description="Critique of what is unnecessary or repetitive."
    )


class AnswerQuestion(BaseModel):
    """The initial answer, critique, and suggested searches."""

    answer: str = Field(
        description="A detailed answer of approximately 250 words."
    )
    reflection: Reflection = Field(
        description="A critique of the initial answer."
    )
    search_queries: List[str] = Field(
        description="One to three queries to research improvements."
    )


class ReviseAnswer(AnswerQuestion):
    """A revised answer that also supplies references."""

    references: List[str] = Field(
        description="Sources supporting the revised answer."
    )
```

`Reflection` is a nested model. `AnswerQuestion` contains an answer, critique, and suggested searches. `ReviseAnswer` inherits those fields and adds references. These classes describe data; they do not themselves search the web or execute Python functions.

## 4. `chains.py`: prompting the model

The model and parsers are:

```python
llm = ChatGroq(model="openai/gpt-oss-120b")
parser = JsonOutputToolsParser(return_id=True)
parser_pydantic = PydanticToolsParser(tools=[AnswerQuestion])
```

- `ChatGroq` calls the selected model through Groq.
- `JsonOutputToolsParser` extracts tool-call information into JSON-like data. In the shown code it is defined but not used.
- `PydanticToolsParser` parses tool-call arguments into the configured Pydantic model. The current parser is configured only for `AnswerQuestion`, not `ReviseAnswer`.

The prompt template includes the current time, a `first_instruction` placeholder, instructions to critique the answer and recommend searches, and `MessagesPlaceholder("messages")` to insert the conversation history.

The first responder is configured to produce an approximately 250-word answer:

```python
first_responder = first_responder_prompt_template | llm.bind_tools(
    tools=[AnswerQuestion],
    tool_choice="AnswerQuestion",
)
```

This makes `AnswerQuestion` an available tool schema and forces the model to produce that tool call. It does **not** execute a Python function named `AnswerQuestion`.

The revisor is configured similarly:

```python
revisor = actor_prompt_template.partial(
    first_instruction=revise_instructions
) | llm.bind_tools(
    tools=[ReviseAnswer],
    tool_choice="ReviseAnswer",
)
```

The keyword is `tools`, not `tool`.

## 5. `tool_executor.py`: searching with Tavily

```python
from langchain_tavily import TavilySearch
from langchain_core.tools import StructuredTool
from langgraph.prebuilt import ToolNode

tavily_tool = TavilySearch(max_results=5)


def run_queries(search_queries: list[str], **kwargs):
    """Run each generated search query."""
    return tavily_tool.batch(
        [{"query": query} for query in search_queries]
    )


execute_tools = ToolNode(
    [
        StructuredTool.from_function(
            run_queries, name=AnswerQuestion.__name__
        ),
        StructuredTool.from_function(
            run_queries, name=ReviseAnswer.__name__
        ),
    ]
)
```

### How `run_queries` works

Given:

```python
[
    "AI SOC startup funding",
    "autonomous SOC architecture",
]
```

the list comprehension creates:

```python
[
    {"query": "AI SOC startup funding"},
    {"query": "autonomous SOC architecture"},
]
```

Tavily expects each request as a dictionary with a `query` key. `.batch(...)` runs the tool for multiple inputs and returns the results.

### Why wrap the same function twice?

`StructuredTool.from_function(...)` creates a LangChain tool from a Python function. The same `run_queries` implementation is registered under two names: `AnswerQuestion` and `ReviseAnswer`. These names match the tool-call names expected from the two model stages.

The function's input schema comes from `run_queries`, **not** from the Pydantic classes. This is a potential mismatch: `AnswerQuestion` and `ReviseAnswer` have multiple fields, while `run_queries` only declares `search_queries` plus `**kwargs`. Depending on schema validation and extra-field handling, the call may work by ignoring extra fields or fail validation. A clearer design is to define dedicated search tools with an explicit `search_queries` input schema, separately from the answer schemas.

### What `ToolNode` does

`ToolNode` reads tool calls from the latest AI message, finds the registered tool by name, executes it, and returns a `ToolMessage` containing the result. `bind_tools` alone only lets the model request a tool call; `ToolNode` executes the registered Python tool.

## 6. `main.py`: graph nodes

The graph uses LangGraph's `MessagesState`. Its `messages` list stores the user request, AI messages containing tool calls, tool messages containing search results, and later AI messages from the revisor. LangGraph's message reducer accumulates messages, so nodes normally return only new messages.

### `draft_node`

```python
def draft_node(state: MessagesState):
    response = first_responder.invoke(
        {"messages": state["messages"]}
    )
    return {"messages": [response]}
```

1. Receives the current graph state.
2. Passes the conversation history to the first-responder chain.
3. The model generates an `AnswerQuestion` tool call containing a draft, critique, and search queries.
4. Returns that AI tool-call message to the state.

It does not perform the search itself.

### `execute_tools`

The `ToolNode` inspects the latest AI message's tool call, executes `run_queries` under the matching registered tool name, and adds search results as tool messages.

### `revise_node`

```python
def revise_node(state: MessagesState):
    response = revisor.invoke(
        {"messages": state["messages"]}
    )
    return {"messages": [response]}
```

1. Receives the accumulated history, including the user request and search results.
2. Calls the revisor chain.
3. The model generates a `ReviseAnswer` tool call containing a revised answer, critique, search queries, and references.
4. Returns that AI tool-call message.

The revised answer is still encoded in tool-call arguments; it is not automatically converted into a plain-text final answer.

## 7. Graph connections

```python
builder.add_edge(START, "draft")
builder.add_edge("draft", "execute_tools")
builder.add_edge("execute_tools", "revise")
builder.add_conditional_edges(
    "revise",
    event_loop,
    ["execute_tools", END],
)
```

The intended sequence is:

```text
START
  ↓
draft
  ↓
execute_tools
  ↓
revise
  ↓
event_loop ── continue ──→ execute_tools
      └────── stop ──────→ END
```

The graph always moves from `draft` to `execute_tools`, then to `revise`. After revision, `event_loop` decides whether to search again or finish.

## 8. How the loop currently works

The current code counts `ToolMessage` objects:

```python
count_tool_visits = sum(
    isinstance(item, ToolMessage)
    for item in state["messages"]
)
```

This counts tool-result messages, not visits to the `execute_tools` node or complete revision rounds.

With:

```python
MAX_ITERATIONS = 2

if count_tool_visits > MAX_ITERATIONS:
    return END
return "execute_tools"
```

the graph ends only when the number of `ToolMessage` objects is **greater than 2**. If each tool-node execution creates exactly one `ToolMessage`, it continues after counts 0, 1, and 2, and stops after count 3. If one execution produces multiple tool messages, the count can increase by more than one.

If the intended maximum is two tool-result messages, use `>= MAX_ITERATIONS`. If the intended maximum is two complete research/revision rounds, use an explicit iteration counter in graph state instead; that is more reliable than counting tool messages.

## 9. Execution and final output caveats

### Graph invocation currently happens at module level

In the provided `main.py`, `graph.invoke(...)` appears before:

```python
if __name__ == "__main__":
    print("Hello from reflexion..")
```

Therefore, importing `main.py` also runs the graph. Usually, put graph execution inside the `if __name__ == "__main__":` block to avoid side effects on import.

### The final message may not contain plain answer text

The current code does:

```python
last_message = res["messages"][-1]
```

but doesn't extract or print the answer. Since the responder and revisor are forced to use tool calls, the final AI message may contain `tool_calls` whose arguments contain the answer, rather than a plain `content` string. Extract the final answer from the `ReviseAnswer` tool-call arguments or parse it into a `ReviseAnswer` Pydantic model.

### The parser is not used by the graph

`parser_pydantic` is used in the standalone chain example in `chains.py`, but it is not part of the graph nodes shown in `main.py`. The graph currently uses tool calls and `ToolNode` rather than parsing the answer and revision into Pydantic objects inside those nodes.

## 10. End-to-end flow in plain English

1. The user asks about AI-powered SOCs and funded startups.
2. `draft_node` sends the request to the initial responder.
3. The model returns an `AnswerQuestion` tool call with a draft, critique, and search queries.
4. `ToolNode` finds `AnswerQuestion` and runs `run_queries`.
5. `run_queries` submits the generated queries to Tavily.
6. `revise_node` passes the full history, including search results, to the revisor.
7. The model returns a `ReviseAnswer` tool call with an updated answer and references.
8. `event_loop` counts tool-result messages and either loops to `execute_tools` or ends.
9. When the graph ends, the caller should extract the answer from the final `ReviseAnswer` tool call and display it.

Key distinctions:
- **Pydantic schema:** describes data structure.
- **`bind_tools`:** tells the model which tool-call schemas it can emit; `tool_choice` can force a choice.
- **`ToolNode`:** executes registered Python tools.
- **Tavily:** performs web searches.
- **LangGraph:** controls sequence and looping.
- **Parser:** converts model output to a Python representation when explicitly included in a chain.

## 11. Suggested improvements

1. Track explicit iteration rounds instead of counting `ToolMessage` objects.
2. Parse and print the final revised answer.
3. Put graph execution inside `if __name__ == "__main__":`.
4. Make the search tool input schema explicit and consistent with its tool-call arguments.
5. Add error handling for malformed tool calls and search failures.
6. Use real source URLs in references rather than placeholders.
7. Keep `.env` out of version control.
