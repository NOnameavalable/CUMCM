"""问题 3 和问题 4 共用的模拟器 HTTP 客户端。"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from CUMCM_B_FINAL_SUBMITION.utils import Point


class SimulatorError(RuntimeError):
    """模拟器连接、HTTP 或业务响应错误。"""


class SimulatorClient:
    """严格串行、支持幂等重试的四接口客户端。"""

    def __init__(
        self,
        robot_id: str,
        base_url: str = "http://127.0.0.1:2026",
        timeout_s: float = 5.0,
        retries: int = 3,
        log_path: Path | None = None,
    ) -> None:
        if not robot_id:
            raise ValueError("robot_id 不能为空。")
        self.robot_id = robot_id
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.retries = retries
        self.log_path = log_path
        self.session_prefix = uuid.uuid4().hex[:10]
        self.sequence = 0
        self.virtual_time_s = 0.0
        self.position = Point(0.0, 0.0)
        self.current_channel = 1
        self.records: list[dict[str, Any]] = []

    def _request_id(self, action: str) -> str:
        self.sequence += 1
        return f"{self.session_prefix}-{action}-{self.sequence}"

    def _base(self, request_id: str) -> dict[str, Any]:
        return {
            "arena_id": "default",
            "robot_id": self.robot_id,
            "request_id": request_id,
        }

    def _action_payload(
        self, action: str, point: Point, channel: int
    ) -> dict[str, Any]:
        payload = self._base(self._request_id(action))
        payload["position"] = {"x": float(point.x), "y": float(point.y)}
        payload["channel"] = int(channel)
        return payload

    def _record(
        self,
        path: str,
        payload: dict[str, Any],
        response: dict[str, Any],
    ) -> None:
        record = {
            "path": path,
            "request": payload,
            "response": response,
            "wall_time_s": time.time(),
        }
        self.records.append(record)
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    def record_planning_event(self, event: dict[str, Any]) -> None:
        """把不调用 HTTP 的本地规划事件写入同一 JSONL 日志。"""
        record = {
            "path": "/local/shared-planner",
            "event": event,
            "wall_time_s": time.time(),
        }
        self.records.append(record)
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(
            payload, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
        last_error: Exception | None = None
        for attempt in range(self.retries):
            request = Request(
                self.base_url + path,
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urlopen(request, timeout=self.timeout_s) as http_response:
                    response = json.loads(http_response.read().decode("utf-8"))
                if response.get("accepted") is not True:
                    raise SimulatorError(f"{path} 未被接受：{response}")
                self.virtual_time_s = float(response["virtual_time_s"])
                self._record(path, payload, response)
                return response
            except HTTPError as exc:
                try:
                    detail = exc.read().decode("utf-8", errors="replace")
                except Exception:
                    detail = ""
                raise SimulatorError(
                    f"{path} HTTP {exc.code}: {detail}"
                ) from exc
            except (URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
                last_error = exc
                if attempt + 1 < self.retries:
                    time.sleep(0.2 * (attempt + 1))
        raise SimulatorError(
            f"{path} 在 {self.retries} 次幂等重试后失败。"
        ) from last_error

    def enter(self) -> dict[str, Any]:
        request_id = self._request_id("enter")
        return self._post("/enter", self._base(request_id))

    def measure(self, point: Point, channel: int) -> dict[str, Any]:
        response = self._post(
            "/measure", self._action_payload("measure", point, channel)
        )
        self.position = point
        self.current_channel = channel
        return response

    def clear(self, point: Point, channel: int) -> dict[str, Any]:
        response = self._post(
            "/clear", self._action_payload("clear", point, channel)
        )
        self.position = point
        return response

    def exit(self) -> dict[str, Any]:
        request_id = self._request_id("exit")
        return self._post("/exit", self._base(request_id))
