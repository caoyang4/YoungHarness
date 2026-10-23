import anthropic
from django.conf import settings


def get_client():
    return anthropic.Anthropic(
        auth_token=settings.ANTHROPIC_AUTH_TOKEN,
        base_url=settings.ANTHROPIC_BASE_URL,
    )


def chat(messages, system=None, max_tokens=1024):
    kwargs = {"system": system} if system else {}
    return get_client().messages.create(
        model=settings.ANTHROPIC_MODEL,
        max_tokens=max_tokens,
        messages=messages,
        **kwargs,
    )


def stream_message(messages, system=None, tools=None, max_tokens=4096):
    """Stream one model turn.

    Yields {"type": "text", "text"} deltas, then a final
    {"type": "message", "content": [blocks], "stop_reason"}.
    """
    kwargs = {}
    if system:
        kwargs["system"] = system
    if tools:
        kwargs["tools"] = tools
    with get_client().messages.stream(
        model=settings.ANTHROPIC_MODEL,
        max_tokens=max_tokens,
        messages=messages,
        **kwargs,
    ) as stream:
        for text in stream.text_stream:
            yield {"type": "text", "text": text}
        final = stream.get_final_message()

    content = []
    for b in final.content:
        if b.type == "text" and b.text:
            content.append({"type": "text", "text": b.text})
        elif b.type == "tool_use":
            content.append({"type": "tool_use", "id": b.id, "name": b.name, "input": b.input})
    yield {"type": "message", "content": content, "stop_reason": final.stop_reason}


if __name__ == "__main__":
    import os

    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "YoungHarness.settings")
    resp = chat([{"role": "user", "content": "用一句话介绍你自己"}])
    print(resp.content[0].text)
    print(resp.usage)
