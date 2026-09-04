"""Backend-neutral, read-only host tools for agent harnesses.

The model gets only cwd-confined Read/Glob/Grep plus an optional in-memory
task checklist.  Tool handlers keep the MCP-shaped content contract used by
``ToolSpec``; individual backends translate those blocks to their SDK's
native tool-result representation.
"""

from __future__ import annotations

import asyncio
import base64
import glob as globlib
import os
import shutil
import subprocess
from pathlib import Path

from .harness import ToolSpec

ALLOWED_CAPABILITIES = frozenset(("read", "glob", "grep", "tasks"))
READ_MAX_BYTES = 512 * 1024
IMAGE_MAX_BYTES = 20 * 1024 * 1024
SEARCH_MAX_BYTES = 256 * 1024
SEARCH_MAX_RESULTS = 500


def readonly_tools(cwd: str, allowed_builtin: tuple[str, ...]) -> list[ToolSpec]:
    """Build the exact cwd-confined exploration surface requested by a call."""
    unknown = sorted(set(allowed_builtin) - ALLOWED_CAPABILITIES)
    if unknown:
        raise ValueError(f"unsupported allowed_builtin capabilities: {unknown}")

    root = Path(cwd).resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"agent cwd is not a directory: {root}")

    def _text(value: str):
        return {"content": [{"type": "text", "text": value}]}

    def _err(value: str):
        return {"content": [{"type": "text", "text": value}], "is_error": True}

    def _within_root(path: Path) -> bool:
        try:
            return os.path.commonpath((str(root), str(path))) == str(root)
        except ValueError:
            return False

    def _resolve(raw: object) -> Path:
        value = str(raw or "").strip()
        if not value:
            raise ValueError("path must be non-empty")
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = root / candidate
        resolved = candidate.resolve(strict=True)
        if not _within_root(resolved):
            raise ValueError(f"path is outside the working directory {root}: {value}")
        return resolved

    def _display(path: Path) -> str:
        return "." if path == root else path.relative_to(root).as_posix()

    def _image_type(data: bytes) -> str | None:
        if data.startswith(b"\x89PNG\r\n\x1a\n"):
            return "image/png"
        if data.startswith(b"\xff\xd8\xff"):
            return "image/jpeg"
        if data.startswith((b"GIF87a", b"GIF89a")):
            return "image/gif"
        if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
            return "image/webp"
        return None

    async def read(args):
        try:
            path = _resolve(args.get("file_path"))
            if not path.is_file():
                return _err(f"not a regular file: {_display(path)}")
            size = path.stat().st_size
            with path.open("rb") as fh:
                prefix = fh.read(16)
                media_type = _image_type(prefix)
                if media_type is not None:
                    if size > IMAGE_MAX_BYTES:
                        return _err(f"image is {size} bytes; maximum is {IMAGE_MAX_BYTES}: {_display(path)}")
                    fh.seek(0)
                    data = fh.read()
                    return {"content": [
                        {"type": "text", "text": f"Image file: {_display(path)} ({media_type}, {size} bytes)"},
                        {"type": "image", "data": base64.b64encode(data).decode("ascii"),
                         "mimeType": media_type},
                    ]}
                fh.seek(0)
                data = fh.read(READ_MAX_BYTES + 1)
            truncated = len(data) > READ_MAX_BYTES
            data = data[:READ_MAX_BYTES]
            if b"\x00" in data:
                return _err(f"binary file is not a supported raster image: {_display(path)}")
            body = data.decode("utf-8", errors="replace")
            suffix = (f"\n\n[truncated after {READ_MAX_BYTES} bytes; narrow the source file before reading]"
                      if truncated else "")
            return _text(f"<path>{_display(path)}</path>\n<content>\n{body}{suffix}\n</content>")
        except (OSError, ValueError) as exc:
            return _err(str(exc))

    async def glob(args):
        pattern = str(args.get("pattern") or "").strip()
        if not pattern:
            return _err("pattern must be non-empty")
        parts = Path(pattern).parts
        if Path(pattern).is_absolute() or ".." in parts:
            return _err("glob pattern must be relative to the working directory and cannot contain '..'")
        matches: list[str] = []
        try:
            for raw in globlib.iglob(str(root / pattern), recursive=True):
                path = Path(raw).resolve(strict=True)
                if not _within_root(path):
                    continue
                matches.append(_display(path) + ("/" if path.is_dir() else ""))
                if len(matches) >= SEARCH_MAX_RESULTS:
                    break
        except (OSError, ValueError) as exc:
            return _err(str(exc))
        matches = sorted(set(matches))
        suffix = f"\n[limited to {SEARCH_MAX_RESULTS} results]" if len(matches) >= SEARCH_MAX_RESULTS else ""
        return _text("\n".join(matches) + suffix if matches else "no matches")

    async def grep(args):
        pattern = str(args.get("pattern") or "")
        if not pattern:
            return _err("pattern must be non-empty")
        try:
            target = _resolve(args.get("path") or ".")
        except (OSError, ValueError) as exc:
            return _err(str(exc))
        rg = shutil.which("rg")
        if rg is None:
            return _err("rg is unavailable on the host")
        command = [rg, "--line-number", "--no-heading", "--color", "never",
                   "--max-count", str(SEARCH_MAX_RESULTS), "--", pattern, str(target)]
        try:
            proc = await asyncio.to_thread(
                subprocess.run, command, cwd=root, capture_output=True, timeout=30, check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return _err(f"grep failed: {exc}")
        if proc.returncode not in (0, 1):
            return _err(proc.stderr.decode("utf-8", errors="replace")[:4000] or
                        f"rg exited with status {proc.returncode}")
        data = proc.stdout
        truncated = len(data) > SEARCH_MAX_BYTES
        body = data[:SEARCH_MAX_BYTES].decode("utf-8", errors="replace")
        if not body:
            return _text("no matches")
        if target.is_dir():
            prefix = str(root) + os.sep
            body = body.replace(prefix, "")
        if truncated:
            body += f"\n[truncated after {SEARCH_MAX_BYTES} bytes]"
        return _text(body.rstrip())

    tools: list[ToolSpec] = []
    if "read" in allowed_builtin:
        tools.append(ToolSpec(
            "Read",
            f"Read a UTF-8 text file or inspect a PNG/JPEG/WebP/GIF image. Relative paths resolve from "
            f"the working directory {root}; paths outside it are rejected.",
            {"file_path": str}, read,
        ))
    if "glob" in allowed_builtin:
        tools.append(ToolSpec(
            "Glob", f"List files matching a recursive glob relative to the working directory {root}.",
            {"pattern": str}, glob,
        ))
    if "grep" in allowed_builtin:
        tools.append(ToolSpec(
            "Grep",
            f"Search text with ripgrep inside the working directory {root}. path may be a relative file or "
            "directory (use '.' for the whole working directory).",
            {"pattern": str, "path": str}, grep,
        ))
    return tools


def task_tools() -> list[ToolSpec]:
    """In-memory progress checklist for one ``run_agent`` call."""
    tasks: dict[str, dict] = {}
    statuses = ("pending", "in_progress", "completed")

    def _text(value: str):
        return {"content": [{"type": "text", "text": value}]}

    def _err(value: str):
        return {"content": [{"type": "text", "text": value}], "is_error": True}

    def _render(task):
        return (f"#{task['id']} [{task['status']}] {task['subject']}"
                + (f" — {task['description']}" if task["description"] else ""))

    async def task_create(args):
        task_id = str(len(tasks) + 1)
        tasks[task_id] = {
            "id": task_id,
            "subject": str(args.get("subject") or "").strip(),
            "description": str(args.get("description") or "").strip(),
            "status": "pending",
        }
        if not tasks[task_id]["subject"]:
            del tasks[task_id]
            return _err("subject is required")
        return _text(f"created task #{task_id}: {tasks[task_id]['subject']}")

    async def task_update(args):
        task_id = str(args.get("taskId") or "").lstrip("#")
        if task_id not in tasks:
            return _err(f"no task #{task_id}; existing: {sorted(tasks, key=int)}")
        status = str(args.get("status") or "").strip()
        if status not in statuses:
            return _err(f"status must be one of {statuses}")
        tasks[task_id]["status"] = status
        return _text(f"updated {_render(tasks[task_id])}")

    async def task_list(_args):
        if not tasks:
            return _text("no tasks yet")
        done = sum(task["status"] == "completed" for task in tasks.values())
        return _text("\n".join(_render(tasks[key]) for key in sorted(tasks, key=int))
                     + f"\n({done}/{len(tasks)} completed)")

    async def task_get(args):
        task_id = str(args.get("taskId") or "").lstrip("#")
        if task_id not in tasks:
            return _err(f"no task #{task_id}; existing: {sorted(tasks, key=int)}")
        return _text(_render(tasks[task_id]))

    return [
        ToolSpec("TaskCreate", "Create a task on your session task list (a progress checklist). Returns its id.",
                 {"subject": str, "description": str}, task_create),
        ToolSpec("TaskUpdate", "Set a task's status: pending | in_progress | completed.",
                 {"taskId": str, "status": str}, task_update),
        ToolSpec("TaskList", "List every task on your session task list with its status.", {}, task_list),
        ToolSpec("TaskGet", "Show one task by id.", {"taskId": str}, task_get),
    ]
