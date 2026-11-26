"""Model-directed tool use for the recording assessment worker."""

import json
import logging
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from .assessment import AssessmentState, report_metrics, song_markers
from .audio import UNUSABLE_ISSUES
from .audio_alignment import align_audio, assess_aligned_audio, selected_reference
from .dtw_alignment import align_dtw
from .model import Model, ModelError
from .storage import normalize_song_title
from .tool_specs import function_tool, inspect_call


log = logging.getLogger(__name__)
MAX_TURNS = 5
SEARCH = "search_song_reference"
ALIGN = "align_recording"
ALIGN_DTW = "align_recording_dtw"
ANALYZE = "analyze_recording"
REUSE = "load_assessment_report"
FINISH = "finish_assessment"
FINAL_STATUSES = ("completed", "needs_input", "failed")


TOOLS = [
    function_tool(
        SEARCH, "按演唱的歌曲名从曲库检索完整参考歌曲",
        when="已上传演唱录音，reference_kind 为 unspecified，本轮要借助曲库查找参考且尚未检索。"
             "歌名缺失或不明确时传空对象请求补充",
        avoid="已有参考附件、指定了内置旋律或历史报告，或者已经取得曲库参考。不可猜测歌名",
        example_context="只有清唱录音，用户明确说「我唱的是《山谷回声》」",
        example_args={"song_title": "山谷回声"},
        properties={"song_title": {"type": "string", "maxLength": 180}},
        optional=("song_title",),
    ),
    function_tool(
        ALIGN, "用音频模型核验参考歌曲与演唱是否匹配，并定位对应片段",
        when="alignment_mode 为 llm，已有参考附件或已从曲库取得参考，仍需自动定位片段",
        avoid="尚无参考、使用内置旋律、已有用户选定的片段或工具返回 aligned，或本轮使用 signal 或 dtw 模式",
        example_context="用户同时上传了演唱和参考歌曲，alignment_mode 为 llm 且没有选定片段",
        example_args={},
    ),
    function_tool(
        ALIGN_DTW, "用备用 DTW 对齐列出可供用户试听和确认的候选片段",
        when="alignment_mode 为 dtw，已有参考附件或曲库参考，仍需寻找候选片段",
        avoid="当前模式是 llm 或 signal、还没有参考、片段已选定，或工具已经返回候选和终态。模型对齐失败时也不自行切换模式",
        example_context="参考附件已上传，用户请求尝试备用对齐，本轮 alignment_mode 为 dtw",
        example_args={},
    ),
    function_tool(
        ANALYZE, "根据有效参考测评演唱录音",
        when="本轮有演唱录音，且已经指定内置旋律、取得 aligned 片段、由用户明确选定片段，"
             "或在 signal 模式下已提供参考附件或取得曲库参考",
        avoid="任务只是复用旧报告、没有可用参考、对齐仍待确认、参考不匹配，或工具已经返回终态",
        example_context="has_recording 为 true，reference_kind 为 builtin，用户要求用内置热身旋律测评录音",
        example_args={},
    ),
    function_tool(
        REUSE, "读取本次请求明确指定的已有测评报告",
        when="has_requested_report 为 true，尚未读取指定报告",
        avoid="没有指定报告，或用户要求测评新录音。不要从聊天文字猜测报告 ID",
        example_context="用户要求查看刚才的结果，客户端已附 assessment_job_id，因此 has_requested_report 为 true",
        example_args={},
    ),
    function_tool(
        FINISH, "按工具的真实终态提交测评任务结果",
        when="已拿到 completed、needs_input 或 failed，原样传入其 status",
        avoid="还没有实际工具结果，或最新结果为 reference_found、aligned 或 rejected。不得猜测状态或自行改为成功",
        example_context="search_song_reference 返回 needs_input 和 song_title_required，任务需等用户补充歌名",
        example_args={"status": "needs_input"},
        properties={"status": {"type": "string", "enum": list(FINAL_STATUSES)}},
    ),
]
TOOLS_WITHOUT_SEARCH = [entry for entry in TOOLS if entry["function"]["name"] != SEARCH]


class AgentState(TypedDict, total=False):
    spec: dict
    model: Model
    messages: list[dict]
    turns: int
    actions: list[dict]
    call: dict | None
    alignment: AssessmentState
    references: list[dict]
    selected: dict
    alignment_observation: dict
    search_observation: dict
    observation: dict
    report: dict
    error: str
    status: str


def model_step(model, messages, *, tools):
    return model.chat_message(messages, tools=tools, tool_choice="auto")


def reference_kind(spec):
    if spec.get("reference_audio_id"):
        return "uploaded_song"
    return "builtin" if spec.get("reference_mode", "builtin") == "builtin" else "unspecified"


def library_search_allowed(spec):
    return reference_kind(spec) == "unspecified"


def valid_args(name, args):
    if not args:
        return True
    title = args.get("song_title")
    return (name == SEARCH and set(args) == {"song_title"} and isinstance(title, str)
            and 1 <= len(title.strip()) <= 180 and bool(normalize_song_title(title)))


def build_react_agent(store, assessment_graph):
    def decide(state):
        spec = state["spec"]
        allow_search = library_search_allowed(spec)
        messages = state.get("messages") or [
            {"role": "system", "content": (
                "你是 DeepVocal 的录音测评 Agent。你负责选择工具并根据实际观察验收任务。"
                "每次请求都通过工具调用通道发起恰好一个调用。只输出测评建议或描述下一步无法执行任务。"
                "依据各工具的调用条件和示例选择下一步，只使用与当前任务和工具观察相符的参数。"
                "指定历史报告时调用 load_assessment_report。reference_kind 为 uploaded_song 时，核验并使用已上传的参考音频。"
                "没有上传参考且未指定内置旋律时，先调用 search_song_reference。"
                "仅在调用 search_song_reference 时检查歌名。优先使用明确给出的 song_title，否则从用户原话提取这段录音的歌名。"
                "只传实际演唱的歌曲名，保留原文，不要根据歌词、歌手或唱法猜歌名，也不要把声乐术语或其他问题中的歌名当作录音曲名。"
                "用户只回复一个歌名时也可以提取。若需搜索曲库而歌名缺失或有歧义，调用 search_song_reference 并传空对象。"
                "alignment_mode 为 llm 且没有已选时间范围时，用已上传的参考或 reference_found 的结果调用 align_recording，"
                "拿到 aligned 后再调用 analyze_recording；为 dtw 时，使用 align_recording_dtw 返回候选。"
                "已指定内置旋律、参考时间范围，或 signal 模式下已完成检索，可以直接 analyze_recording。"
                "不传路径、ID 或猜测的数值。每轮只调用一个工具。"
                "拿到工具结果后检查 status：completed 代表报告可用，needs_input 代表需要用户补充输入，"
                "reference_mismatch 或 reference_segment_mismatch 表示参考与演唱不匹配，以 needs_input 结束。"
                "song_title_required 表示需要用户补充歌名；reference_not_found 表示需要补充对比歌声。两者都以 needs_input 结束。"
                "failed 代表工具失败。调用 finish_assessment 提交实际终态。"
                "无法测评时不能声称成功，不要自行编造音频指标。只通过工具执行和提交结果。"
            )},
            {"role": "user", "content": json.dumps({
                "task": spec.get("assessment_query", "测评演唱录音"),
                "user_query": spec.get("user_query", spec.get("assessment_query", "")),
                "has_recording": bool(spec.get("audio_id")),
                "has_requested_report": bool(spec.get("assessment_job_id")),
                "song_title": spec.get("song_title"),
                "reference_kind": reference_kind(spec),
                "library_search_allowed": allow_search,
                "alignment_mode": spec.get("alignment_mode", "signal"),
                "has_selected_range": spec.get("reference_start_seconds") is not None,
            }, ensure_ascii=False)},
        ]
        turns = state.get("turns", 0)
        if turns >= MAX_TURNS:
            return {"status": "failed", "error": "测评模型达到工具调用上限，未提交有效的测评结果。"}
        try:
            reply = model_step(state["model"], messages, tools=TOOLS if allow_search else TOOLS_WITHOUT_SEARCH)
        except ModelError as exc:
            return {"status": "failed", "error": str(exc), "turns": turns + 1}
        call = inspect_call(reply)
        if call is None:
            correction = {"role": "user", "content":
                          "上一轮没有产生有效的单次工具调用。请用结构化工具调用通道选择并执行一个工具。"}
            return {"messages": [*messages, correction], "call": None, "turns": turns + 1}
        assistant = {"role": "assistant", "content": reply.get("content"), "tool_calls": reply["tool_calls"]}
        if reply.get("reasoning_content"):
            assistant["reasoning_content"] = reply["reasoning_content"]
        return {"messages": [*messages, assistant], "call": call, "turns": turns + 1}

    def references_for(state):
        if "references" in state:
            return [dict(reference) for reference in state["references"]]
        if key := state["spec"].get("reference_audio_id"):
            reference = store.audio(key)
            if reference and reference["kind"] == "reference":
                return [{"audio_id": key, "title": reference["filename"]}]
        return []

    def use_audio_tool(name, state, args):
        spec = state["spec"]
        mode = spec.get("alignment_mode", "signal")
        if name == SEARCH:
            title = spec.get("song_title")
            if not title and (extracted := args.get("song_title")):
                query = spec.get("user_query", spec.get("assessment_query", ""))
                if normalize_song_title(extracted) in normalize_song_title(query):
                    title = extracted
            if not title or not normalize_song_title(title):
                issue = "song_title_required"
                result = report_metrics(issue) | {"recording_audio_id": spec["audio_id"],
                                                  "offset_seconds": spec["offset_seconds"]}
                observed = {"status": "needs_input", "issue": issue}
                return {"report": result, "observation": observed}, observed
            search_spec = {**spec, "song_title": title.strip()}
            alignment = assessment_graph.invoke({"spec": search_spec, "search_only": True,
                                                 "candidates_only": mode != "signal"}, output_keys=[
                "spec", "reference", "references", "duration", "reference_track", "performance", "match", "result",
            ])
            if references := alignment.get("references"):
                observed = {"status": "reference_found", "references": references}
                return {"references": references, "search_observation": observed}, observed
            if "result" not in alignment:
                reference = song_markers(alignment["spec"], alignment["reference"], alignment["match"], alignment["duration"])
                observed = {"status": "reference_found", "reference": {
                    key: reference[key] for key in ("audio_id", "start_seconds", "end_seconds", "detected_transpose", "fingerprint_hits")
                }}
                return {"alignment": alignment, "search_observation": observed}, observed
            result = alignment["result"]
        elif name in (ALIGN, ALIGN_DTW):
            references = references_for(state)
            if name == ALIGN:
                selected, result = align_audio(store, spec, references, state["model"])
                if selected:
                    observed = {"status": "aligned", "reference": selected}
                    return {"selected": selected, "alignment_observation": observed}, observed
            else:
                result = align_dtw(store, spec, references)
        elif name == ANALYZE:
            if mode != "signal" and reference_kind(spec) != "builtin":
                selected = state.get("selected")
                if selected is None:
                    selected, result = selected_reference(store, spec, references_for(state)[0], state["model"])
                if selected:
                    score_spec = {**spec, "reference_start_seconds": None, "reference_end_seconds": None, "transpose": 0} if "references" in state else spec
                    result = assess_aligned_audio(store, score_spec, selected, state["model"])
            else:
                result = assessment_graph.invoke(state.get("alignment") or {"spec": spec}, output_keys="result")
        else:
            job = store.job(spec["assessment_job_id"])
            if not job or job["kind"] != "assessment" or job["state"] != "succeeded":
                raise ValueError("指定测评任务暂无可用报告。")
            result = {**job["result"], "source_assessment": job["result"].get("source_assessment", job["id"])}
            result.pop("plan", None)
            result.pop("plan_error", None)
        status = "needs_input" if result["issue"] in UNUSABLE_ISSUES else "completed"
        observed = {"status": status, "issue": result["issue"], "matched_notes": result["matched_notes"],
                    "total_notes": result["total_notes"], "median_pitch_error_cents": result.get("median_pitch_error_cents"),
                    "reference_candidates": len(result.get("candidates", []))}
        if reason := result.get("reference_search_reason"):
            observed["reference_search_reason"] = reason
        return {"report": result, "observation": observed}, observed

    def act(state):
        call = state["call"]
        name, args = call["name"], call["args"]
        spec = state["spec"]
        updates = {}
        if name == FINISH:
            evidence = state.get("observation")
            valid = (evidence and set(args) == {"status"} and args["status"] == evidence["status"])
            if valid:
                observed = {"status": evidence["status"]}
                updates["status"] = evidence["status"]
            else:
                observed = {"status": "rejected", "message": "验收状态必须与实际工具结果一致。"}
                if evidence:
                    observed["expected_status"] = evidence["status"]
                elif "alignment" in state or "selected" in state:
                    observed["message"] = "已找到参考片段，请先调用 analyze_recording 取得测评报告。"
        elif name in (SEARCH, ALIGN, ALIGN_DTW, ANALYZE, REUSE) and valid_args(name, args):
            kind = reference_kind(spec)
            mode = spec.get("alignment_mode", "signal")
            has_range = (kind == "uploaded_song" and spec.get("reference_start_seconds") is not None
                         and "references" not in state)
            if (name == REUSE) != bool(spec.get("assessment_job_id")) or (name != REUSE and not spec.get("audio_id")):
                observed = {"status": "rejected", "message": "此工具所需的输入未提供。"}
            elif evidence := state.get("observation"):
                observed = {"status": "rejected", "message": "工具已返回终态，请调用 finish_assessment。",
                            "expected_status": evidence["status"]}
            elif name == SEARCH and kind == "builtin":
                observed = {"status": "rejected", "message": "本次已指定内置练习旋律，请直接调用 analyze_recording。"}
            elif name == SEARCH and not library_search_allowed(spec):
                observed = {"status": "rejected", "message": "本轮已有参考附件，请使用附件完成对齐或测评。"}
            elif name == SEARCH and "selected" in state:
                observed = {"status": "rejected", "message": "参考片段已对齐，请调用 analyze_recording。"}
            elif name in (ALIGN, ALIGN_DTW) and (kind == "builtin" or not references_for(state)
                    or mode != ("llm" if name == ALIGN else "dtw")):
                observed = {"status": "rejected", "message": "请先取得参考歌曲，并调用与本轮对齐方式对应的工具。"}
            elif name == ALIGN and "alignment_observation" in state:
                observed = state["alignment_observation"]
            elif name == ANALYZE and kind == "unspecified" and "search_observation" not in state:
                observed = {"status": "rejected", "message": "本轮没有指定参考，请先调用 search_song_reference。"}
            elif name == ANALYZE and mode != "signal" and kind != "builtin" and not has_range and "selected" not in state:
                observed = {"status": "rejected", "message": "需要先对齐参考歌曲并确认片段，才能测评。"}
            elif name == SEARCH and "search_observation" in state:
                observed = state["search_observation"]
            else:
                try:
                    updates, observed = use_audio_tool(name, state, args)
                except (ValueError, OSError, ModelError) as exc:
                    message = "无法读取已上传的音频。" if isinstance(exc, OSError) else str(exc)
                    observed = {"status": "failed", "message": message}
                    updates.update(observation=observed, error=message)
                except Exception:
                    log.exception("Assessment tool failed")
                    observed = {"status": "failed", "message": "音频分析失败，请检查录音后重试。"}
                    updates.update(observation=observed, error=observed["message"])
        else:
            observed = {"status": "rejected", "message": "工具名称或参数无效。"}
        action = name if name in (SEARCH, ALIGN, ALIGN_DTW, ANALYZE, REUSE, FINISH) else "unknown_tool"
        return {**updates, "actions": [*state.get("actions", []), {"tool": action, "status": observed["status"]}],
                "messages": [*state["messages"], {"role": "tool", "tool_call_id": call["id"],
                                                 "content": json.dumps(observed, ensure_ascii=False)}]}

    graph = StateGraph(AgentState)
    graph.add_node("decide", decide)
    graph.add_node("act_and_observe", act)
    graph.add_edge(START, "decide")
    graph.add_conditional_edges("decide", lambda state: END if state.get("status") else
                                "act_and_observe" if state.get("call") else "decide",
                                [END, "act_and_observe", "decide"])
    graph.add_conditional_edges("act_and_observe", lambda state: END if state.get("status") else "decide",
                                [END, "decide"])
    return graph.compile()
