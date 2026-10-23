import json

from django.http import JsonResponse, StreamingHttpResponse
from django.shortcuts import render
from django.views.decorators.http import require_POST

from YoungHarness.agent.loop import run_agent

SYSTEM_PROMPT = (
    "你是 YoungHarness 里的助手,回答简洁准确。"
    "需要时可以调用工具查看项目文件或当前时间。"
)


def index(request):
    return render(request, "chat.html")


def _sse(data, event=None):
    head = f"event: {event}\n" if event else ""
    return f"{head}data: {json.dumps(data, ensure_ascii=False)}\n\n"


def _parse_messages(request):
    try:
        messages = json.loads(request.body)["messages"]
    except (ValueError, KeyError, TypeError):
        return None
    valid = (
        isinstance(messages, list)
        and messages
        and all(
            isinstance(m, dict)
            and m.get("role") in ("user", "assistant")
            and isinstance(m.get("content"), str)
            and m["content"]
            for m in messages
        )
        and messages[-1]["role"] == "user"
    )
    return messages if valid else None


@require_POST
def chat_stream(request):
    """POST {"messages": [{"role", "content"}, ...]} -> text/event-stream."""
    messages = _parse_messages(request)
    if messages is None:
        return JsonResponse({"error": "invalid messages"}, status=400)

    def events():
        try:
            for ev in run_agent(messages, system=SYSTEM_PROMPT):
                kind = ev.pop("type")
                # plain text deltas use the default SSE event name
                yield _sse(ev, event=None if kind == "text" else kind)
        except Exception as e:
            yield _sse({"error": str(e)}, event="error")

    resp = StreamingHttpResponse(events(), content_type="text/event-stream")
    resp["Cache-Control"] = "no-cache"
    resp["X-Accel-Buffering"] = "no"
    return resp
