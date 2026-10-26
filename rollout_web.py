"""Local, read-only-view HTTP interface for a rollout controller.

The only writable operations are labeling the current round and stopping the
session. Controller state and JPEGs are supplied by the owner of the rollout;
this server does not serve arbitrary filesystem paths.
"""

from __future__ import annotations

import ipaddress
import json
import socket
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, urlsplit

_MAX_BODY_BYTES = 4096
_HTML_PATH = Path(__file__).with_name("rollout.html")


def make_server(host: str, port: int, controller) -> ThreadingHTTPServer:
    """Bind the rollout UI to a loopback address and attach *controller*.

    Controller methods: snapshot() -> dict, frame(round_id=None) -> JPEG bytes
    or None, label(round_id, result) -> dict, and stop() -> dict. frame() must
    return None when the requested round is no longer current.
    """
    if host != "localhost":
        try:
            loopback = ipaddress.ip_address(host).is_loopback
        except ValueError:
            loopback = False
        if not loopback:
            raise ValueError("The rollout UI must bind to a loopback address")

    page = _HTML_PATH.read_bytes()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _send(self, status: int, body: bytes = b"", *, content_type: str | None = None,
                  round_id: str | None = None) -> None:
            self.send_response(status)
            if content_type is not None:
                self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            if round_id is not None:
                self.send_header("X-Round-Id", quote(round_id, safe=""))
            self.end_headers()
            if body:
                self.wfile.write(body)

        def _json(self, status: int, value: object) -> None:
            payload = json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
            self._send(status, payload, content_type="application/json; charset=utf-8")

        def _error(self, status: int, message: str) -> None:
            self._json(status, {"error": message})

        def _valid_host(self) -> bool:
            # Prevent DNS-rebinding requests whose Host names point at loopback.
            authority = self.headers.get("Host", "")
            try:
                parsed = urlsplit("http://" + authority)
                address = parsed.hostname
                try:
                    local = ipaddress.ip_address(address).is_loopback
                except (ValueError, TypeError):
                    local = address == "localhost"
                request_port = parsed.port or 80
                return (local and request_port == self.server.server_port
                        and not parsed.username and not parsed.password
                        and not parsed.path and not parsed.query and not parsed.fragment)
            except ValueError:
                return False

        def _same_origin_write(self) -> bool:
            if self.headers.get("Sec-Fetch-Site") not in (None, "same-origin", "none"):
                return False
            origin = self.headers.get("Origin")
            if origin is None:
                # A cross-origin browser cannot send this header without a
                # preflight, and this server never grants CORS preflights.
                return self.headers.get("X-Requested-With") == "XMLHttpRequest"
            try:
                parsed = urlsplit(origin)
                host = urlsplit("http://" + self.headers["Host"])
                return (parsed.scheme == "http" and parsed.hostname == host.hostname
                        and (parsed.port or 80) == (host.port or 80)
                        and not parsed.username and not parsed.password
                        and not parsed.path and not parsed.query and not parsed.fragment)
            except (ValueError, KeyError):
                return False

        def _body(self, *, allow_empty: bool = False) -> bytes | None:
            length_header = self.headers.get("Content-Length")
            if length_header is None:
                if allow_empty and not self.headers.get("Transfer-Encoding"):
                    return b""
                self._error(411, "缺少 Content-Length")
                return None
            try:
                length = int(length_header)
            except ValueError:
                self._error(400, "无效的 Content-Length")
                return None
            if length < 0:
                self._error(400, "无效的 Content-Length")
                return None
            if length > _MAX_BODY_BYTES:
                self.close_connection = True
                self._error(413, "请求体过大")
                return None
            if self.headers.get("Transfer-Encoding"):
                self.close_connection = True
                self._error(400, "不支持分块请求体")
                return None
            body = self.rfile.read(length)
            if len(body) != length:
                self.close_connection = True
                self._error(400, "请求体不完整")
                return None
            return body

        def _controller_call(self, operation, *args) -> object | None:
            try:
                return operation(*args)
            except ValueError as exc:
                self._error(409, str(exc))
            except Exception:
                self.log_error("Controller operation failed")
                self._error(500, "服务器内部错误")
            return None

        def do_GET(self) -> None:
            if not self._valid_host():
                self._error(421, "仅允许本机访问")
                return
            route = urlsplit(self.path)
            if route.path == "/":
                self._send(200, page, content_type="text/html; charset=utf-8")
            elif route.path == "/api/status":
                snapshot = self._controller_call(controller.snapshot)
                if snapshot is not None:
                    try:
                        self._json(200, snapshot)
                    except Exception:
                        self.log_error("Snapshot serialization failed")
                        self._error(500, "服务器内部错误")
            elif route.path == "/api/frame":
                try:
                    query = parse_qs(route.query, keep_blank_values=True, strict_parsing=True)
                except ValueError:
                    self._error(400, "无效的查询参数")
                    return
                ids = query.get("round_id")
                if ids is not None and (len(ids) != 1 or not ids[0] or len(ids[0]) > 256):
                    self._error(400, "无效的 round_id")
                    return
                round_id = ids[0] if ids else None
                try:
                    jpeg = controller.frame(round_id)
                    if jpeg is None:
                        self._send(204)
                    else:
                        self._send(200, jpeg, content_type="image/jpeg", round_id=round_id)
                except ValueError as exc:
                    self._error(409, str(exc))
                except Exception:
                    self.log_error("Frame request failed")
                    self._error(500, "服务器内部错误")
            elif route.path == "/api/artifact":
                try:
                    query = parse_qs(route.query, keep_blank_values=True, strict_parsing=True)
                except ValueError:
                    self._error(400, "无效的查询参数")
                    return
                ids = query.get("round_id")
                kinds = query.get("kind")
                if (set(query) != {"round_id", "kind"} or ids is None or len(ids) != 1
                        or not ids[0] or len(ids[0]) > 256 or kinds not in (["first"], ["last"])):
                    self._error(400, "需要 round_id 和 first/last 类型")
                    return
                try:
                    png = controller.artifact(ids[0], kinds[0])
                    if png is None:
                        self._error(404, "已保存图像不存在")
                    else:
                        self._send(200, png, content_type="image/png")
                except ValueError as exc:
                    self._error(409, str(exc))
                except Exception:
                    self.log_error("Artifact request failed")
                    self._error(500, "服务器内部错误")
            else:
                self._error(404, "接口不存在")

        def do_POST(self) -> None:
            # Error responses must not leave an unread POST body on a keep-alive connection.
            self.close_connection = True
            if not self._valid_host():
                self._error(421, "仅允许本机访问")
                return
            if not self._same_origin_write():
                self._error(403, "禁止跨域写入")
                return
            route = urlsplit(self.path)
            if route.path not in ("/api/label", "/api/stop"):
                self._error(404, "接口不存在")
                return
            if route.path == "/api/label" and self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower() != "application/json":
                self._error(415, "仅接受 JSON 请求体")
                return
            body = self._body(allow_empty=route.path == "/api/stop")
            if body is None:
                return
            if route.path == "/api/stop":
                if body:
                    self._error(400, "停止请求不需要请求体")
                    return
                result = self._controller_call(controller.stop)
            else:
                try:
                    payload = json.loads(body)
                except (UnicodeDecodeError, json.JSONDecodeError):
                    self._error(400, "无效的 JSON")
                    return
                if (not isinstance(payload, dict)
                        or not isinstance(payload.get("round_id"), str)
                        or not payload["round_id"] or len(payload["round_id"]) > 256
                        or payload.get("result") not in ("success", "failure")):
                    self._error(400, "需要 round_id 和 success/failure 结果")
                    return
                result = self._controller_call(controller.label, payload["round_id"], payload["result"])
            if result is not None:
                try:
                    self._json(200, result)
                except Exception:
                    self.log_error("Response serialization failed")
                    self._error(500, "服务器内部错误")

    if host == "::1":
        class IPv6ThreadingHTTPServer(ThreadingHTTPServer):
            address_family = socket.AF_INET6

        return IPv6ThreadingHTTPServer((host, port), Handler)
    return ThreadingHTTPServer((host, port), Handler)
