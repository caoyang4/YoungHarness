"""The agent loop: stream a model turn, run requested tools, feed results back."""
import json
from concurrent.futures import ThreadPoolExecutor

from YoungHarness.agent.tools import MAX_OUTPUT_CHARS, REGISTRY, tool_specs
from YoungHarness.llm.client import stream_message

MAX_TURNS = 10
COMPACT_THRESHOLD_CHARS = 150_000  # rough proxy for context size
KEEP_RECENT_TOOL_RESULTS = 4
COMPACTED = "[旧的工具输出已压缩]"

# Callables run after each tool round: hook(messages) -> None.
POST_SAMPLING_HOOKS = []


def normalize_messages(messages):
    """Build the API payload: drop empty messages, merge same-role neighbours."""
    out = []
    for m in messages:
        content = m["content"]
        if isinstance(content, str):
            content = [{"type": "text", "text": content}] if content else []
        if not content:
            continue
        if out and out[-1]["role"] == m["role"]:
            out[-1] = {"role": m["role"], "content": out[-1]["content"] + content}
        else:
            out.append({"role": m["role"], "content": list(content)})
    return out


def extract_tool_use_blocks(message):
    return [b for b in message["content"] if b["type"] == "tool_use"]


def partition_tool_calls(blocks, tools):
    """Group consecutive concurrency-safe calls into one batch; others run alone."""
    batches = []
    for b in blocks:
        tool = tools.get(b["name"])
        safe = bool(tool and tool.concurrency_safe)
        if safe and batches and batches[-1][0]:
            batches[-1][1].append(b)
        else:
            batches.append((safe, [b]))
    return [calls for _, calls in batches], [safe for safe, _ in batches]


def _run_one(block, tools):
    tool = tools.get(block["name"])
    if tool is None:
        return _result(block, f"unknown tool: {block['name']}", True)
    try:
        return _result(block, tool.func(block["input"]), False)
    except Exception as e:
        return _result(block, f"{type(e).__name__}: {e}", True)


def _result(block, content, is_error):
    content = str(content)
    if len(content) > MAX_OUTPUT_CHARS:
        content = content[:MAX_OUTPUT_CHARS] + "\n...[已截断]"
    return {
        "type": "tool_result",
        "tool_use_id": block["id"],
        "content": content or "(no output)",
        "is_error": is_error,
    }


def run_tools(blocks, tools):
    """Yield tool_result blocks in call order; safe batches run in parallel."""
    batches, safe_flags = partition_tool_calls(blocks, tools)
    for calls, safe in zip(batches, safe_flags):
        if safe and len(calls) > 1:
            with ThreadPoolExecutor(max_workers=len(calls)) as pool:
                results = list(pool.map(lambda b: _run_one(b, tools), calls))
        else:
            results = [_run_one(b, tools) for b in calls]
        yield from results


def should_compact(messages):
    return len(json.dumps(messages, ensure_ascii=False)) > COMPACT_THRESHOLD_CHARS


def compact(messages):
    """Blank out all but the most recent tool results (keeps tool_use pairing valid)."""
    refs = [
        b for m in messages if m["role"] == "user" and isinstance(m["content"], list)
        for b in m["content"] if b["type"] == "tool_result"
    ]
    for b in refs[:-KEEP_RECENT_TOOL_RESULTS]:
        b["content"] = COMPACTED


def run_agent(messages, system=None, tools=None, max_turns=MAX_TURNS):
    """Drive the loop; yield events for the UI.

    Events: text, tool_use, tool_result, compact, done.
    """
    tools = REGISTRY if tools is None else tools
    specs = tool_specs(tools)
    messages = list(messages)

    for _ in range(max_turns):
        # 1. assemble context (memory injection would hook in here)
        api_messages = normalize_messages(messages)

        # 2. stream one model turn
        assistant = None
        for event in stream_message(api_messages, system, specs):
            if event["type"] == "message":
                assistant = {"role": "assistant", "content": event["content"]}
            else:
                yield event
        messages.append(assistant)

        # 3. tool calls requested?
        blocks = extract_tool_use_blocks(assistant)
        if not blocks:
            yield {"type": "done", "reason": "end_turn"}
            return

        # 4. execute tools, streaming results out as they finish
        for b in blocks:
            yield {"type": "tool_use", "id": b["id"], "name": b["name"], "input": b["input"]}
        results = []
        names = {b["id"]: b["name"] for b in blocks}
        for r in run_tools(blocks, tools):
            results.append(r)
            yield {
                "type": "tool_result",
                "id": r["tool_use_id"],
                "name": names[r["tool_use_id"]],
                "content": r["content"],
                "is_error": r["is_error"],
            }

        # 5. feed results back for the next turn
        messages.append({"role": "user", "content": results})

        # 6. hooks, then compact if the context is getting large
        for hook in POST_SAMPLING_HOOKS:
            hook(messages)
        if should_compact(messages):
            compact(messages)
            yield {"type": "compact"}

    yield {"type": "done", "reason": "max_turns"}
