from dataclasses import dataclass
from datetime import datetime
from typing import Callable

from django.conf import settings

MAX_OUTPUT_CHARS = 20000
MAX_READ_BYTES = 200_000
# Secrets and tooling dirs the model must never read.
DENIED_PARTS = {".git", ".venv", ".idea", "__pycache__"}


@dataclass
class Tool:
    name: str
    description: str
    input_schema: dict
    func: Callable[[dict], str]
    # Read-only tools can run in parallel; others run one at a time.
    concurrency_safe: bool = False

    def spec(self):
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


REGISTRY: dict[str, Tool] = {}


def register(tool: Tool) -> Tool:
    REGISTRY[tool.name] = tool
    return tool


def tool_specs(tools=None):
    return [t.spec() for t in (tools or REGISTRY).values()]


def _safe_path(rel):
    path = (settings.BASE_DIR / rel).resolve()
    if settings.BASE_DIR not in path.parents and path != settings.BASE_DIR:
        raise ValueError("path is outside the project")
    parts = path.relative_to(settings.BASE_DIR).parts
    if any(p in DENIED_PARTS or p.startswith(".env") for p in parts):
        raise ValueError("access to this path is denied")
    return path


def _get_time(_):
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")


def _list_dir(args):
    path = _safe_path(args.get("path", "."))
    if not path.is_dir():
        raise ValueError("not a directory")
    entries = sorted(
        p for p in path.iterdir()
        if p.name not in DENIED_PARTS and not p.name.startswith(".env")
    )
    return "\n".join(p.name + ("/" if p.is_dir() else "") for p in entries) or "(empty)"


def _read_file(args):
    path = _safe_path(args["path"])
    if not path.is_file():
        raise ValueError("not a file")
    data = path.read_bytes()[:MAX_READ_BYTES]
    return data.decode("utf-8", errors="replace")


register(Tool(
    name="get_time",
    description="获取当前的本地日期和时间。",
    input_schema={"type": "object", "properties": {}},
    func=_get_time,
    concurrency_safe=True,
))
register(Tool(
    name="list_dir",
    description="列出项目目录下的文件和子目录,path 相对项目根目录,默认为根目录。",
    input_schema={
        "type": "object",
        "properties": {"path": {"type": "string", "description": "相对路径,如 YoungHarness/llm"}},
    },
    func=_list_dir,
    concurrency_safe=True,
))
register(Tool(
    name="read_file",
    description="读取项目内的文本文件内容,path 相对项目根目录。",
    input_schema={
        "type": "object",
        "properties": {"path": {"type": "string", "description": "相对路径,如 manage.py"}},
        "required": ["path"],
    },
    func=_read_file,
    concurrency_safe=True,
))
