# ADK Reflection – Week 10

In Week 9, every step of the agent loop required explicit Python code: a `while True` block
to drive iteration, manual JSON parsing to extract tool-call arguments from the raw API
response, and hand-written logic to detect when the model had reached a final answer.
The ADK replaces all of this boilerplate with a declarative runtime that reads the agent's
tool signatures and docstrings directly and executes the Perceive → Think → Act → Check
cycle autonomously.

The most striking difference is state management: in Week 9, conversation history had to be
carried as a manually appended list of dicts passed into every API call, whereas the ADK
maintains session state transparently across turns in the Web UI without a single line of
state-management code from the developer.

Tool calling in Week 9 required parsing a structured JSON object from the model output,
dispatching to the correct function by name, and serialising the result back into the
message array – with the ADK, registering a Python function with full type hints and a
docstring is sufficient for the framework to generate the schema, route calls, and inject
results, reducing the integration surface from ~50 lines of glue code to a single entry
in the `tools=` list.
