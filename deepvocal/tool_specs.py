"""Function registrations with one worked example per tool."""

import json


def inspect_call(reply):
    calls = reply.get("tool_calls") if isinstance(reply, dict) else None
    if not isinstance(calls, list) or len(calls) != 1:
        return None
    try:
        call = calls[0]
        function = call["function"]
        args = json.loads(function["arguments"])
        if (call["type"] == "function" and isinstance(call["id"], str) and call["id"].strip()
                and isinstance(function["name"], str) and isinstance(args, dict)):
            return {"id": call["id"], "name": function["name"], "args": args}
    except (KeyError, TypeError, ValueError):
        pass
    return None


def function_tool(name, purpose, *, when, avoid, example_context, example_args, properties=None, optional=()):
    properties = properties or {}
    arguments = json.dumps(example_args, ensure_ascii=False, separators=(",", ":"))
    description = (
        f"{purpose}。应调用的情况是{when}。"
        f"不应调用的情况是{avoid}。"
        f"示例中，{example_context}。发起 {name}({arguments}) 的结构化工具调用。"
    )
    return {"type": "function", "function": {"name": name, "description": description, "parameters": {
        "type": "object", "properties": properties, "required": [key for key in properties if key not in optional],
        "additionalProperties": False,
    }}}
