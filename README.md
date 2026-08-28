# Agentic Research Assistant

A portfolio-oriented **Agentic AI research workflow** built with Python and an OpenAI-compatible LLM API. The assistant can decide when to call external research tools, gather evidence from the web and arXiv, synthesize a sourced report, critique its own output through a reflection step, revise the report, and export the final result as HTML.

## What this project demonstrates

- **LLM tool / function calling** with dynamically generated tool schemas
- **Multi-step agent execution** with an iterative tool-use loop
- **External API integration** through Tavily and the arXiv API
- **Reflection and self-critique** for report quality improvement
- **Structured research workflow** from question -> retrieval -> synthesis -> review -> revised report
- **Environment-based configuration** for API keys, base URLs, and model selection

## Architecture

```text
Research question
      |
      v
Research Agent (LLM)
      |
      +------> Tavily web search
      |
      +------> arXiv paper search
      |
      v
Draft research report
      |
      v
Reflection / Review Agent
      |
      v
Revised report
      |
      v
HTML report generator
```

## Repository structure

```text
.
├── main.py              # Agent loop, reflection, and report-generation workflow
├── research_tools.py    # External research/retrieval tools
├── .env.example         # Environment-variable template
├── requirements.txt     # Python dependencies
├── .gitignore
├── LICENSE
└── outputs/             # Generated reports (ignored by Git)
```

## Setup

```bash
python -m venv .venv
```

Activate the environment, then install dependencies:

```bash
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and provide your keys:

```env
OPENAI_API_KEY=...
OPENAI_BASE_URL=https://api.gapgpt.app/v1
TAVILY_API_KEY=...
```

Then run:

```bash
python main.py
```

The example workflow researches recent developments in Agentic AI, performs a reflection/revision pass, and writes the final HTML report to `outputs/research_report.html`.

## Implementation notes

The main agent does not hard-code OpenAI tool schemas. Instead, it inspects the Python signatures of enabled research functions and converts them into function-calling schemas at runtime. The LLM can then choose a tool, provide arguments, receive the tool output, and continue the research loop until it produces a final response.

The second stage acts as a reviewer: it explicitly checks strengths, limitations, missing evidence, unsupported claims, and citation quality before producing a revised report.

## Code provenance

The **initial retrieval helper file** used during the course exercise was based on starter/reference code. In this repository it has been **refactored and reduced to the arXiv and Tavily integrations actually used by the project**, and this provenance is documented in `research_tools.py`.

The agent orchestration, dynamic tool-schema generation, iterative tool-calling workflow, reflection/revision stage, configuration structure, and report-generation pipeline are presented as the core portfolio implementation of this project.

This distinction is intentional: the repository is meant to demonstrate the integration and Agentic AI workflow without claiming original authorship for course-provided utility code.

## Skills demonstrated

`Python` · `LLM APIs` · `Agentic AI` · `Tool Calling` · `Reflection` · `Tavily` · `arXiv API` · `REST APIs` · `Structured Workflows`

## License

See [LICENSE](LICENSE).
