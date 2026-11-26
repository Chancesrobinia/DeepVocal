from contextlib import contextmanager
from dataclasses import dataclass, field
import json
import os
import ssl

import httpx


PROVIDERS = {
    "openai": "https://api.openai.com/v1",
    "deepseek": "https://api.deepseek.com",
    "dashscope": "https://dashscope.aliyuncs.com/compatible-mode/v1",
}
DASHSCOPE_RERANK_BASE = "https://dashscope.aliyuncs.com/api/v1"


class ModelError(Exception):
    pass


def chat_text(message):
    answer = message.get("content")
    if not isinstance(answer, str) or not answer.strip() or message.get("tool_calls"):
        raise ModelError("模型服务未返回文本内容。")
    return answer.strip()


def status_error(status):
    messages = {
        400: "请检查模型是否支持本次请求。",
        401: "鉴权失败，请确认 API Key 有效，且与地域和 Base URL 匹配。",
        403: "访问被拒绝，请检查模型权限、服务开通状态和账户余额。",
        404: "找不到模型或接口，请检查模型 ID 和 Base URL。",
        429: "触发限流或额度限制，请检查调用配额。",
    }
    hint = messages.get(status, "模型服务暂时无法处理请求，请稍后重试。" if status >= 500
                        else "请检查模型服务地址和请求配置。")
    return ModelError(f"模型服务返回 HTTP {status}。{hint}")


def certificate_error(exc):
    cause = exc
    while cause is not None:
        if isinstance(cause, ssl.SSLCertVerificationError):
            return True
        cause = cause.__cause__
    return False


@contextmanager
def model_transport():
    try:
        yield
    except httpx.TimeoutException:
        raise ModelError("模型服务请求超时，请检查网络后重试。") from None
    except httpx.ProxyError:
        raise ModelError("无法通过代理连接模型服务，请检查代理配置。") from None
    except httpx.ConnectError as exc:
        if certificate_error(exc):
            raise ModelError("模型服务 TLS 证书校验失败，请检查证书和代理配置。") from None
        raise ModelError("无法建立到模型服务的连接，请检查网络、代理和 Base URL。") from None
    except (httpx.InvalidURL, httpx.UnsupportedProtocol):
        raise ModelError("模型服务 Base URL 无效，请检查地址和协议。") from None
    except (httpx.HTTPError, ValueError):
        raise ModelError("模型请求失败，请检查连接和 API 设置。") from None


def chat_payload(messages, tools, tool_choice):
    payload = {"messages": messages}
    if tools is not None:
        payload["tools"] = tools
    if tool_choice is not None:
        payload["tool_choice"] = tool_choice
    return payload


@dataclass(frozen=True)
class Model:
    provider: str
    name: str
    key: str = field(repr=False)

    def endpoint(self):
        base = PROVIDERS.get(self.provider)
        if self.provider == "custom":
            base = os.getenv("DEEPVOCAL_CUSTOM_API_BASE")
        elif self.provider == "dashscope":
            base = os.getenv("DEEPVOCAL_DASHSCOPE_API_BASE") or base
        if not base:
            raise ModelError("模型服务地址未配置或不能识别该服务商。")
        return base.rstrip("/")

    def request_options(self, payload, timeout):
        # DashScope is normally reachable directly on this host. An explicit
        # opt-in keeps corporate proxy routing available for other installations.
        use_environment = self.provider != "dashscope" or os.getenv("DEEPVOCAL_DASHSCOPE_USE_PROXY") == "1"
        return {"json": {"model": self.name, **payload},
                "headers": {"Authorization": f"Bearer {self.key}"} if self.key else {},
                "timeout": httpx.Timeout(timeout, connect=min(timeout, 8)), "trust_env": use_environment}

    def check_response(self, response):
        if not response.is_success:
            if self.provider == "dashscope" and response.status_code == 400:
                try:
                    if response.json()["error"]["code"] == "Arrearage":
                        raise ModelError("百炼返回账户欠费状态（Arrearage）。")
                except (KeyError, TypeError, ValueError):
                    pass
            raise status_error(response.status_code)

    def call(self, path, payload, *, base=None, timeout=35):
        with model_transport():
            response = httpx.post((base or self.endpoint()).rstrip("/") + path, **self.request_options(payload, timeout))

        self.check_response(response)
        try:
            return response.json()
        except ValueError:
            raise ModelError("模型服务返回了非 JSON 内容，请确认使用的是模型 API 地址。") from None

    def chat_message(self, messages, *, tools=None, tool_choice=None):
        data = self.call("/chat/completions", chat_payload(messages, tools, tool_choice))
        try:
            choice = data["choices"][0]
            message = choice["message"]
            if not isinstance(message, dict) or choice.get("finish_reason") in ("length", "content_filter"):
                raise ValueError
        except (IndexError, KeyError, TypeError, ValueError):
            raise ModelError("模型服务未返回完整的问答结果。") from None
        return {**message, "role": "assistant"}

    def chat(self, messages, *, tools=None, tool_choice=None):
        return chat_text(self.chat_message(messages, tools=tools, tool_choice=tool_choice))

    def stream_message(self, messages, *, on_text, cancelled, tools=None, tool_choice=None):
        payload = {**chat_payload(messages, tools, tool_choice), "stream": True}
        content, reasoning, calls = [], [], {}
        finish = None
        with model_transport(), httpx.stream("POST", self.endpoint() + "/chat/completions", **self.request_options(payload, 35)) as response:
            if not response.is_success:
                response.read()
                self.check_response(response)
            for line in response.iter_lines():
                if cancelled():
                    raise ModelError("问答连接已关闭。")
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    event = json.loads(data)
                    if event.get("error"):
                        raise ModelError("模型服务中断了流式回答。")
                    if event["choices"] == []:
                        continue
                    choice = event["choices"][0]
                    finish = choice.get("finish_reason") or finish
                    delta = choice["delta"]
                    for fragment in delta.get("tool_calls") or []:
                        index = fragment["index"]
                        call = calls.setdefault(index, {"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
                        call["id"] = fragment.get("id") or call["id"]
                        call["type"] = fragment.get("type") or call["type"]
                        fields = fragment.get("function") or {}
                        call["function"]["name"] += fields.get("name") or ""
                        call["function"]["arguments"] += fields.get("arguments") or ""
                    if delta.get("reasoning_content"):
                        reasoning.append(delta["reasoning_content"])
                    if delta.get("content"):
                        content.append(delta["content"])
                        if not calls:
                            on_text(delta["content"])
                except (AttributeError, IndexError, KeyError, TypeError, ValueError):
                    raise ModelError("模型服务返回了无效的流式数据。") from None
        if finish not in ("stop", "tool_calls") or (finish == "tool_calls" and not calls):
            raise ModelError("模型服务未返回完整的流式回答。")
        message = {"role": "assistant", "content": "".join(content) or None}
        if calls:
            message["tool_calls"] = [calls[index] for index in sorted(calls)]
        if reasoning:
            message["reasoning_content"] = "".join(reasoning)
        return message

    def embed(self, texts, dimensions=None):
        payload = {"input": texts}
        if dimensions is not None:
            payload["dimensions"] = dimensions
        body = self.call("/embeddings", payload)
        try:
            rows = sorted(body["data"], key=lambda row: row["index"])
            if [item["index"] for item in rows] != list(range(len(texts))):
                raise ValueError
            return [item["embedding"] for item in rows]
        except (KeyError, TypeError, ValueError):
            raise ModelError("Embedding 服务返回的数据格式无效。") from None

    def rerank(self, query, documents, top_n):
        data = self.call(
            "/services/rerank/text-rerank/text-rerank",
            {"input": {"query": query, "documents": documents}, "parameters": {"top_n": top_n}},
            base=os.getenv("DEEPVOCAL_RERANK_API_BASE") or DASHSCOPE_RERANK_BASE,
            timeout=5,
        )
        try:
            scores = {}
            for item in data["output"]["results"]:
                index, score = item["index"], item["relevance_score"]
                if (type(index) is not int or not 0 <= index < len(documents) or index in scores
                        or type(score) not in (int, float) or not 0 <= score <= 1):
                    raise ValueError
                scores[index] = score
            if len(scores) != min(len(documents), top_n):
                raise ValueError
            return sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        except (KeyError, TypeError, ValueError):
            raise ModelError("重排模型返回的段落位置或分数无效。") from None
