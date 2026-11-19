from dataclasses import dataclass, field
import os
import ssl

import httpx


PROVIDERS = {
    "openai": "https://api.openai.com/v1",
    "deepseek": "https://api.deepseek.com",
    "dashscope": "https://dashscope.aliyuncs.com/compatible-mode/v1",
}


class ModelError(Exception):
    pass


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

    def call(self, path, payload):
        # DashScope is normally reachable directly on this host. An explicit
        # opt-in keeps corporate proxy routing available for other installations.
        use_environment = self.provider != "dashscope" or os.getenv("DEEPVOCAL_DASHSCOPE_USE_PROXY") == "1"
        try:
            response = httpx.post(
                self.endpoint() + path, json={"model": self.name, **payload},
                headers={"Authorization": f"Bearer {self.key}"} if self.key else {},
                timeout=httpx.Timeout(35, connect=8), trust_env=use_environment,
            )
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

        if not response.is_success:
            if self.provider == "dashscope" and response.status_code == 400:
                try:
                    if response.json()["error"]["code"] == "Arrearage":
                        raise ModelError("百炼返回账户欠费状态（Arrearage）。")
                except (KeyError, TypeError, ValueError):
                    pass
            raise status_error(response.status_code)
        try:
            return response.json()
        except ValueError:
            raise ModelError("模型服务返回了非 JSON 内容，请确认使用的是模型 API 地址。") from None

    def chat(self, messages):
        data = self.call("/chat/completions", {"messages": messages})
        try:
            answer = data["choices"][0]["message"]["content"]
            if not isinstance(answer, str) or not answer.strip():
                raise ValueError
            return answer.strip()
        except (IndexError, KeyError, TypeError, ValueError):
            raise ModelError("模型服务未返回文本内容。") from None

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
