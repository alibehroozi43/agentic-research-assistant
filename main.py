"""Agentic research assistant with tool use, reflection, and HTML reporting."""

from __future__ import annotations

import inspect
import json
import os
from pathlib import Path
from typing import Any, get_origin

from dotenv import load_dotenv
from openai import OpenAI

from research_tools import TOOL_MAPPING

load_dotenv()

SYSTEM_PROMPT = """
You are a research assistant that can search the web and arXiv to write
accurate, well-sourced research reports.

Requirements:
- Use tools when fresh or external evidence is needed.
- Cite sources whenever relevant and include full URLs when possible.
- Prefer primary and authoritative sources.
- Use an academic, concise tone and clearly labeled sections.
- Never fabricate sources, citations, or evidence.
- Do not use placeholder citations.
""".strip()


def build_client() -> OpenAI:
    """Create an OpenAI-compatible client from environment variables."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured.")

    kwargs: dict[str, Any] = {"api_key": api_key}
    base_url = os.getenv("OPENAI_BASE_URL")
    if base_url:
        kwargs["base_url"] = base_url

    return OpenAI(**kwargs)


def python_type_to_json_type(annotation: Any) -> str:
    """Convert common Python annotations to JSON Schema primitive types."""
    mapping = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
    }

    if annotation in mapping:
        return mapping[annotation]

    origin = get_origin(annotation)
    if origin is list:
        return "array"
    if origin is dict:
        return "object"

    return "string"


def function_to_tool_schema(func) -> dict[str, Any]:
    """Generate an OpenAI function-tool schema from a Python callable."""
    signature = inspect.signature(func)
    properties: dict[str, Any] = {}
    required: list[str] = []

    for name, parameter in signature.parameters.items():
        annotation = parameter.annotation
        if annotation is inspect.Parameter.empty:
            annotation = str

        property_schema: dict[str, Any] = {
            "type": python_type_to_json_type(annotation)
        }
        if parameter.default is not inspect.Parameter.empty:
            property_schema["default"] = parameter.default
        else:
            required.append(name)

        properties[name] = property_schema

    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": inspect.getdoc(func) or f"Execute {func.__name__}",
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
    }


def generate_research_report_with_tools(
    prompt: str,
    *,
    model: str | None = None,
    tool_names: list[str] | None = None,
    max_turns: int = 5,
    client: OpenAI | None = None,
) -> str:
    """Run the research agent loop and return the synthesized report."""
    if max_turns < 1:
        raise ValueError("max_turns must be at least 1.")

    client = client or build_client()
    model = model or os.getenv("RESEARCH_MODEL", "gpt-4o-mini")
    tool_names = tool_names or ["tavily_search_tool", "arxiv_search_tool"]

    unknown_tools = [name for name in tool_names if name not in TOOL_MAPPING]
    if unknown_tools:
        raise ValueError(f"Unknown tools: {', '.join(unknown_tools)}")

    selected_tools = {name: TOOL_MAPPING[name] for name in tool_names}
    tool_definitions = [
        function_to_tool_schema(func) for func in selected_tools.values()
    ]

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    for turn in range(max_turns):
        print(f"\nTurn {turn + 1}/{max_turns}")
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tool_definitions,
            tool_choice="auto",
        )
        message = response.choices[0].message
        messages.append(message.model_dump(exclude_none=True))

        if not message.tool_calls:
            return message.content or ""

        for tool_call in message.tool_calls:
            tool_name = tool_call.function.name
            try:
                arguments = json.loads(tool_call.function.arguments or "{}")
            except json.JSONDecodeError:
                arguments = {}

            print(f"Calling tool: {tool_name} | arguments={arguments}")

            if tool_name not in selected_tools:
                result: Any = {"error": f"Tool '{tool_name}' is unavailable."}
            else:
                try:
                    result = selected_tools[tool_name](**arguments)
                except Exception as exc:
                    result = {"error": str(exc)}

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(
                        result,
                        ensure_ascii=False,
                        default=str,
                    ),
                }
            )

    messages.append(
        {
            "role": "user",
            "content": (
                "Using all information gathered so far, write the final research "
                "report now. Do not call additional tools."
            ),
        }
    )
    response = client.chat.completions.create(model=model, messages=messages)
    return response.choices[0].message.content or ""


def reflective_research_tool(
    prompt: str,
    report: str,
    *,
    model: str | None = None,
    temperature: float = 0.3,
    client: OpenAI | None = None,
) -> dict[str, str]:
    """Critique a report and return both reflection and revised report."""
    if not report.strip():
        raise ValueError("The research report is empty.")

    client = client or build_client()
    model = model or os.getenv("REFLECTION_MODEL", "gpt-5.1")

    review_prompt = f"""
Original research question:
{prompt}

Research report:
{report}

Review this research report carefully.

Return EXACTLY these two sections:

=== REFLECTION ===
Analyze strengths, limitations, missing evidence, weak or unsupported claims,
citation quality, and opportunities for improvement.

=== REVISED_REPORT ===
Rewrite the full report incorporating the improvements.

Requirements:
- Preserve valid citations and URLs.
- Do not fabricate sources.
- Improve academic clarity and organization.
- Remove unsupported claims where appropriate.
- Keep the report focused on the original question.
""".strip()

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert academic reviewer and research editor. "
                    "Critically improve reports without fabricating evidence."
                ),
            },
            {"role": "user", "content": review_prompt},
        ],
        temperature=temperature,
    )

    full_output = (response.choices[0].message.content or "").strip()
    marker = "=== REVISED_REPORT ==="

    if marker not in full_output:
        return {"reflection": full_output, "revised_report": full_output}

    reflection_part, revised_report = full_output.split(marker, 1)
    reflection = reflection_part.replace("=== REFLECTION ===", "").strip()

    return {
        "reflection": reflection,
        "revised_report": revised_report.strip(),
    }


def convert_report_to_html(
    report: str,
    *,
    model: str | None = None,
    client: OpenAI | None = None,
) -> str:
    """Convert a research report into a clean standalone HTML document."""
    if not report.strip():
        raise ValueError("The research report is empty.")

    client = client or build_client()
    model = model or os.getenv("REPORT_MODEL", "gpt-5.1")

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You convert research reports into clean standalone HTML.",
            },
            {
                "role": "user",
                "content": (
                    "Convert the following report into a complete, readable HTML "
                    "document. Preserve citations and URLs and make links clickable. "
                    "Respond only with valid HTML.\n\nReport:\n" + report
                ),
            },
        ],
        temperature=0.3,
    )
    return (response.choices[0].message.content or "").strip()


def run_demo() -> None:
    """Run the end-to-end research -> reflection -> HTML workflow."""
    research_prompt = """
What are the latest developments in agentic AI?

Focus on:
1. Tool-using agents
2. Multi-agent systems
3. Agent memory
4. Recent academic research
5. Major limitations and open research problems

Include relevant academic papers and web sources.
""".strip()

    client = build_client()
    report = generate_research_report_with_tools(
        research_prompt,
        client=client,
    )
    reflection = reflective_research_tool(
        research_prompt,
        report,
        client=client,
    )
    html = convert_report_to_html(
        reflection["revised_report"],
        client=client,
    )

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "research_report.html"
    output_path.write_text(html, encoding="utf-8")

    print("\nReflection:\n")
    print(reflection["reflection"])
    print(f"\nSaved revised HTML report to: {output_path}")


if __name__ == "__main__":
    run_demo()
