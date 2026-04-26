from __future__ import annotations

import json
import os
import select
import subprocess
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any


class CodexAppServerError(RuntimeError):
    pass


class CodexAppServerClient:
    def __init__(self, *, cwd: Path | None = None, model: str = "gpt-5.4") -> None:
        self.cwd = cwd or Path.cwd()
        self.model = model
        self._proc: subprocess.Popen[bytes] | None = None
        self._next_id = 0
        self.stderr_text = ""

    def __enter__(self) -> "CodexAppServerClient":
        self._proc = subprocess.Popen(
            ["codex", "app-server", "--listen", "stdio://"],
            cwd=str(self.cwd),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self._initialize()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if not self._proc:
            return
        if self._proc.poll() is None:
            if self._proc.stdin:
                self._proc.stdin.close()
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait(timeout=5)
        if self._proc.stderr is not None:
            try:
                self.stderr_text = self._proc.stderr.read().decode("utf-8", errors="replace")
            except Exception:
                self.stderr_text = ""
        if self._proc.stdout:
            self._proc.stdout.close()
        self._proc = None

    def _new_id(self) -> int:
        self._next_id += 1
        return self._next_id

    def _send(self, payload: dict[str, Any]) -> None:
        assert self._proc and self._proc.stdin
        self._proc.stdin.write((json.dumps(payload) + "\n").encode("utf-8"))
        self._proc.stdin.flush()

    def _read_messages(self, timeout_sec: float = 120.0) -> Iterator[dict[str, Any]]:
        assert self._proc and self._proc.stdout
        deadline = time.time() + timeout_sec
        buffer = b""
        fd = self._proc.stdout.fileno()
        while time.time() < deadline:
            remaining = max(0.0, deadline - time.time())
            ready, _, _ = select.select([fd], [], [], remaining)
            if not ready:
                break
            chunk = os.read(fd, 65536)
            if not chunk:
                if self._proc.poll() is not None:
                    break
                continue
            buffer += chunk
            while b"\n" in buffer:
                raw_line, buffer = buffer.split(b"\n", 1)
                text = raw_line.decode("utf-8", errors="replace").strip()
                if not text:
                    continue
                try:
                    yield json.loads(text)
                except json.JSONDecodeError:
                    continue
        raise CodexAppServerError("Timed out waiting for app-server message")

    def _wait_for_response(self, request_id: int, timeout_sec: float = 120.0) -> dict[str, Any]:
        for message in self._read_messages(timeout_sec=timeout_sec):
            if message.get("id") == request_id:
                if "error" in message:
                    raise CodexAppServerError(f"App-server error for request {request_id}: {message['error']}")
                return message["result"]
        raise CodexAppServerError(f"No response for request {request_id}")

    def _initialize(self) -> None:
        initialize_id = self._new_id()
        self._send(
            {
                "id": initialize_id,
                "method": "initialize",
                "params": {
                    "clientInfo": {
                        "name": "interp-baseline-client",
                        "title": "Interp Baseline Client",
                        "version": "0.1.0",
                    },
                    "capabilities": {
                        "experimentalApi": True,
                        "optOutNotificationMethods": [
                            "command/exec/outputDelta",
                            "item/agentMessage/delta",
                            "item/plan/delta",
                            "item/fileChange/outputDelta",
                            "item/reasoning/summaryTextDelta",
                            "item/reasoning/textDelta",
                        ],
                    },
                },
            }
        )
        self._wait_for_response(initialize_id)
        self._send({"method": "initialized"})

    def run_prompt(
        self,
        prompt_text: str,
        *,
        output_schema: dict[str, Any] | None = None,
        turn_timeout_sec: float = 300.0,
    ) -> str:
        thread_id = self._new_id()
        self._send(
            {
                "id": thread_id,
                "method": "thread/start",
                "params": {
                    "model": self.model,
                    "cwd": str(self.cwd),
                    "approvalPolicy": "never",
                },
            }
        )
        thread_result = self._wait_for_response(thread_id)
        actual_thread_id = thread_result["thread"]["id"]

        turn_request_id = self._new_id()
        self._send(
            {
                "id": turn_request_id,
                "method": "turn/start",
                "params": {
                    "threadId": actual_thread_id,
                    "input": [{"type": "text", "text": prompt_text, "textElements": []}],
                    "outputSchema": output_schema,
                    "model": self.model,
                    "approvalPolicy": "never",
                },
            }
        )
        self._wait_for_response(turn_request_id)

        final_text: str | None = None
        for message in self._read_messages(timeout_sec=turn_timeout_sec):
            if message.get("method") == "item/completed":
                item = message.get("params", {}).get("item", {})
                if item.get("type") == "agentMessage" and item.get("phase") == "final_answer":
                    final_text = item.get("text", "")
            if message.get("method") == "turn/completed":
                break
        if final_text is None:
            raise CodexAppServerError("App-server turn completed without a final agent message")
        return final_text
