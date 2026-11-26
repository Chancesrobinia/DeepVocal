"""Tool-driven practice planning with locally verified assessment evidence."""

import json
import re
from sqlite3 import Error as StorageError
from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import Field, ValidationError

from .audio import UNUSABLE_ISSUES
from .knowledge import EXERCISES
from .model import Model, ModelError
from .practice_plan import (DRILLS, InvalidPlan, ModelPlan, PlanningOutput, PracticeBasis,
                            compile_plan, goal_focus, plan_validation, prepare_evidence_practice)
from .storage import normalize_song_title
from .tool_specs import function_tool, inspect_call


SEARCH = "search_assessments"
BUILD = "build_practice_plan"
FINISH = "finish_practice_plan"
MAX_TURNS = 5
ASSESSMENT_REF = "assessment_1"
TERMINAL = ("completed", "needs_input", "failed")
LINKED_CUE = re.compile(r"(?:刚刚|刚才|本次|这次|这份)[^，。！？；]{0,18}?(?:测评|评估|报告|结果)")
PAST_REPORT_CUE = re.compile(r"(?:上一次|上次|以前|之前|历史|上一份|上个)[^，。！？；]{0,18}?(?:测评|评估|报告)")
REPORT_REQUEST = re.compile(r"(?:根据|结合|依据|基于|参考|沿用|使用|按)[^，。！？；]{0,25}?(?:测评|评估|报告)")
WITHOUT_REPORT = re.compile(r"不(?:要|用|必|参考|结合|依据)\s*(?:使用|参考|结合)?(?:录音)?(?:测评|评估|报告)")
BASELINE_REQUEST = re.compile(r"入门|通用练习|建立(?:复测)?基准")
SONG_TITLE = re.compile(r"《([^》]{1,180})》")
PREVIOUS_SONG = re.compile(r"(?:上一次|上次|之前)\s*([^，。！？；《》]{1,60}?)的(?:测评|评估|报告)")
MINUTE_CUE = re.compile(
    r"(?:每天|每日|一天|每次)\s*(?:的?(?:练习)?时间)?\s*(?P<limit>最多|不超过|不能超过|至多|上限(?:是|为)?|只有|只能)?"
    r"\s*(?:练习|用|练)?\s*(?P<minutes>[0-9]{1,3}|[零〇一二两三四五六七八九十]{1,3})\s*分钟"
    r"|(?P<cap>最多|不超过|不能超过|至多|只有|只能)\s*(?:练习|用|练)?\s*"
    r"(?P<cap_minutes>[0-9]{1,3}|[零〇一二两三四五六七八九十]{1,3})\s*分钟"
)


class SearchArgs(PlanningOutput):
    scope: Literal["linked", "history"]
    song_title: str | None = Field(default=None, min_length=1, max_length=180)
    segment_id: str | None = Field(default=None, pattern=r"^[a-zA-Z0-9_-]{1,80}$")


class BuildArgs(PlanningOutput):
    focus: str
    assessment_ref: Literal["", "assessment_1"] = ""
    draft: ModelPlan


def draft_schema():
    schema = ModelPlan.model_json_schema()
    types = schema.pop("$defs")
    day = types["PlannedDay"]
    day["properties"]["exercises"]["items"] = types["PlannedExercise"]
    schema["properties"]["sessions"]["items"] = day
    return schema


TOOLS = [
    function_tool(
        SEARCH, "查找练习规划可用的测评证据",
        when="用户要求依据当前或以前的测评，或 assessment_required 为 true。绑定和本会话测评用 linked，过往测评用 history",
        avoid="只要求通用练习或已经取得对应报告。歌名和片段只使用当前请求明确提供的信息",
        example_context="用户说「按上次《山谷回声》的测评做计划」且没有绑定报告",
        example_args={"scope": "history", "song_title": "山谷回声"},
        properties={"scope": {"type": "string", "enum": ["linked", "history"]},
                    "song_title": {"type": "string", "minLength": 1, "maxLength": 180},
                    "segment_id": {"type": "string", "maxLength": 80}},
        optional=("song_title", "segment_id"),
    ),
    function_tool(
        BUILD, "用批准的 Skill 和给定证据校验并组装练习计划",
        when="通用目标可直接提交，要求引用测评时先拿到 assessment_ref。当前的 primary_evidence_id 就是 focus",
        avoid="必需报告尚未找到，草案引用了未返回的依据，或工具已经验证了计划",
        example_context="本轮只要求一次入门训练，当前的 primary_evidence_id 为 general，没有使用测评",
        example_args={"focus": "general", "draft": {"rationale": "先建立短句复测基准。", "minutes_per_day": 8,
            "sessions": [{"day": 1, "objective": "录下舒适乐句", "exercises": [
                {"evidence_id": "general", "step": 1, "minutes": 6, "instructions": "挑一句容易轻声唱的旋律。"}]}]}},
        properties={"focus": {"type": "string", "enum": list(EXERCISES)},
                    "assessment_ref": {"type": "string", "enum": ["", ASSESSMENT_REF]},
                    "draft": draft_schema()}, optional=("assessment_ref",),
    ),
    function_tool(
        FINISH, "提交工具已经确认的规划结果",
        when="build_practice_plan 返回 validated 时提交 completed，或工具明确返回需要用户补充的信息时提交 needs_input",
        avoid="没有验收结果、检索刚找到报告或草案尚待修正。不得自行编造成功或失败",
        example_context="build_practice_plan 已返回 status 为 validated",
        example_args={"status": "completed"},
        properties={"status": {"type": "string", "enum": list(TERMINAL)}},
    ),
]
TOOL_NAMES = {tool["function"]["name"] for tool in TOOLS}
SYSTEM_PROMPT = (
    "你是 DeepVocal 的声乐练习规划师。每轮发起一个结构化工具调用，观察返回后决定下一步。"
    "根据工具注册中的条件与示例行动，示例数据不能用作本轮证据。"
    "若 assessment_required 为 true，先按 suggested_scope 检索，拿到报告后用最新 planning 字段覆盖旧规划上下文。"
    "通用目标可以跳过测评检索，直接编制计划。工具没有找到必需的测评时，请提交 needs_input。"
    "制定时用已批准的 evidence_id，主项每天出现。主项一天用 step 1，两天用 [1,2] 或 [1,3]，三天用 [1,2,3]。"
    "超过三天从 step 1 单向推进到 step 3。每次练习给出不同的执行细节，再做完整原句复测。"
    "遵守 schedule 的天数和每日总时长范围。专项 minutes 是建议比例，程序会预留热身和复测并统一分配。"
    "说明练习顺序与证据的关系，不编造歌词、时间点、数值或生理原因。"
    "收到 invalid_plan 时按校验提示修正，收到 validated 后提交 completed。"
)
VALIDATION_HINTS = {
    "schedule": "核对天数、每日总时长和从 1 开始的日期编号。",
    "exercise_selection": "主项每天都要练。专项只用 approved_skills 内的证据 ID，同一天不能重复。",
    "exercise_progression": "按递增阶段提供每天不同的练习动作。",
    "primary_progression": "第一天从分解开始，三天及以上最后一天回到原段。",
    "unsupported_cause": "练习细节请使用可观察的表现，避免自行推断原因。",
    "schema": "根据字段位置和错误代码修正参数结构。",
    "assessment_ref": "使用本轮检索返回的 assessment_ref，通用计划留空。",
    "focus": "focus 必须与当前 planning 中的 primary_evidence_id 一致。",
}


class PracticeState(TypedDict, total=False):
    request: dict
    model: Model
    session_id: str | None
    upstream: dict | None
    required: bool
    scope: str
    basis: PracticeBasis
    report_ref: str
    messages: list[dict]
    actions: list[dict]
    turns: int
    call: dict | None
    status: str
    failure: str
    needs_input: str
    tool_error: str
    plan: dict
    validation: dict
    invalid_drafts: int


def apply_time_constraint(req):
    match = (MINUTE_CUE.search(req.get("query", "")) or MINUTE_CUE.search(req.get("goal") or "")
             or MINUTE_CUE.search(req.get("user_query", "")))
    if match is None:
        return None
    token = match["minutes"] or match["cap_minutes"]
    if token.isdigit():
        minutes = int(token)
    else:
        digits = {char: i for i, char in enumerate("零一二三四五六七八九")} | {"两": 2, "〇": 0}
        tens, unit, ones = token.partition("十")
        minutes = (digits.get(tens, 1) * 10 if unit else 0) + digits.get(ones if unit else tens, 0)
    capped = bool(match["cap"] or match["limit"])
    if minutes < 5 or (minutes > 90 and not capped):
        return "完整计划包含热身、专项和复测。请给出 5–90 分钟内可用的练习时间。"
    if capped:
        req["minutes_cap"] = min(minutes, 90)
        if req.get("minutes"):
            req["minutes"] = min(req["minutes"], req["minutes_cap"])
    else:
        req["minutes"] = minutes
    return None


def usable_report(report):
    return bool(report and report.get("issue") and report["issue"] not in UNUSABLE_ISSUES)


def lookup_report(store, state, args):
    req = state["request"]
    if state.get("upstream") is not None:
        report = state["upstream"]
        return (report.get("source_assessment") or req.get("assessment_job_id"), report, None) if usable_report(report) else (
            None, None, "请先完成本次录音测评，再根据结果制定计划。")
    if job_id := req.get("assessment_job_id"):
        job = store.job(job_id)
        if job and job["kind"] == "assessment" and job["state"] == "succeeded" and usable_report(job["result"]):
            return job["result"].get("source_assessment") or job_id, job["result"], None
        return None, None, "指定测评暂无可用结果，请选择一份已完成的测评或重新上传录音。"
    session = state.get("session_id") if args.scope == "linked" else None
    if args.scope == "linked" and not session:
        return None, None, "本会话暂无可用测评，请先完成一次录音测评。"
    title = req.get("song_title")
    query = req.get("query", "")
    if not title and (match := SONG_TITLE.search(query)):
        title = match[1]
    elif not title and (match := PREVIOUS_SONG.search(query)):
        candidate = match[1].strip()
        if not re.search(r"录音|演唱|清唱|测评|报告", candidate):
            title = candidate
    title = title or args.song_title
    if title and not normalize_song_title(title):
        return None, None, "请提供可用于查找测评的歌名。"
    rows = store.planning_reports(session=session, song_title=title, segment_id=args.segment_id, unusable=UNUSABLE_ISSUES)
    if not rows:
        return None, None, "没有找到对应的测评。请告诉我歌曲或片段，或上传录音完成一次测评。"
    if title and not args.segment_id:
        sources = {row["reference"].get("audio_id") or row["reference"]["title"]
                   for _, row in rows if row.get("reference")}
        if len(sources) > 1:
            return None, None, "找到了同名歌曲的多份测评，请从历史记录选定要使用的测评。"
    report_id, report = rows[0]
    return report.get("source_assessment") or report_id, report, None


def build_practice_agent(store, skills, memory, labels):
    def basis_for(req, report=None, source=None):
        target = req.get("focus")
        if target in (None, "general"):
            query = req.get("query", "")
            explicit = not report or req.get("intent") == "practice_plan" or re.search(r"只练|专练|练习重点", query)
            target = goal_focus(req.get("goal") or (query if explicit else ""))
        focus = target or (report or {}).get("issue") or "general"
        focus = focus if focus in EXERCISES else "general"
        profile = {fact["subject"]: fact["value"] for fact in store.facts("profile")}
        goal = req.get("goal") or profile.get("goal")
        basis = prepare_evidence_practice(skills, {**req, "goal": goal}, report, focus, labels,
                                         include_secondary=target is None, preferred_minutes=profile.get("minutes"))
        progress = None if report and report.get("reference") else next(
            (fact["value"] for fact in store.facts("progress") if fact["subject"] == req["melody_id"]), None)
        basis.plan.update(focus_label=labels[basis.plan["focus"]], goal=goal, song_progress=progress,
                          source_assessment=source)
        return basis

    def decide(state):
        turns = state.get("turns", 0)
        if turns >= MAX_TURNS:
            return {"failure": "tool_limit"}
        messages = state.get("messages") or [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps({
                "assessment_required": state["required"], "suggested_scope": state["scope"],
                "planning": state["basis"].context,
            }, ensure_ascii=False)},
        ]
        try:
            reply = state["model"].chat_message(messages, tools=TOOLS, tool_choice="auto")
        except ModelError:
            return {"failure": "model_error", "turns": turns + 1}
        call = inspect_call(reply)
        if call is None:
            correction = {"role": "user", "content": "请通过工具调用通道发出恰好一个有效的工具调用。"}
            return {"call": None, "messages": [*messages, correction], "turns": turns + 1}
        return {"call": call, "messages": [*messages, reply], "turns": turns + 1}

    def perform(state, name, args):
        if name == FINISH:
            expected = ("completed" if state.get("plan") else "needs_input" if state.get("needs_input")
                        else "failed" if state.get("tool_error") else None)
            if expected and args == {"status": expected}:
                return {"status": expected}, {"status": expected}
            return {}, {"status": "rejected", "message": "请先完成计划校验，或取得缺少必要测评的结果。"}
        if state.get("plan") or state.get("needs_input") or state.get("tool_error"):
            return {}, {"status": "rejected", "message": "任务已有结果，请调用 finish_practice_plan。"}
        if name == SEARCH:
            if state.get("report_ref"):
                return {}, {"status": "rejected", "message": "本轮已取得测评，请使用该报告制定计划。"}
            search = SearchArgs.model_validate(args)
            req = state["request"]
            if not state["required"] and (WITHOUT_REPORT.search(req["query"]) or BASELINE_REQUEST.search(req["query"])):
                return {}, {"status": "rejected", "message": "本轮是通用基准练习，可以直接制定计划。"}
            if not (state.get("upstream") or req.get("assessment_job_id")) and (
                (search.song_title and normalize_song_title(search.song_title) not in normalize_song_title(
                    " ".join((req["query"], req.get("user_query", ""), req.get("song_title") or ""))))
                or (search.segment_id and search.segment_id not in req["query"])
            ):
                return {}, {"status": "rejected", "message": "只使用用户本轮明确给出的歌名或片段 ID。"}
            if state["required"] and search.scope != state["scope"]:
                return {}, {"status": "rejected", "message": f"本轮应在 {state['scope']} 范围检索。"}
            if not state["required"] and search.scope == "history" and not (
                    search.song_title or search.segment_id or SONG_TITLE.search(state["request"]["query"])):
                return {}, {"status": "rejected", "message": "通用计划可以直接制定。查历史请提供歌曲或片段。"}
            report_id, report, message = lookup_report(store, state, search)
            if not report:
                update = {"needs_input": message} if state["required"] else {}
                return update, {"status": "needs_input" if update else "not_found", "message": message}
            basis = basis_for(state["request"], report, report_id)
            return {"basis": basis, "report_ref": ASSESSMENT_REF}, {
                "status": "found", "assessment_ref": ASSESSMENT_REF, "planning": basis.context,
            }
        if name != BUILD:
            return {}, {"status": "rejected", "message": "工具名称无效。"}
        if state["required"] and not state.get("report_ref"):
            return {}, {"status": "rejected", "message": "本任务需要测评证据，请先调用 search_assessments。"}
        build = BuildArgs.model_validate(args)
        basis = state["basis"]
        if build.assessment_ref != state.get("report_ref", ""):
            raise InvalidPlan("assessment_ref")
        if build.focus != basis.plan["focus"]:
            raise InvalidPlan("focus")
        plan = compile_plan(basis, build.draft, state["model"].name)
        return {"plan": plan}, {"status": "validated", "days": plan["days"],
                                "minutes_per_day": plan["minutes_per_day"], "focus": plan["focus"]}

    def act(state):
        call = state["call"]
        name = call["name"]
        try:
            update, observation = perform(state, name, call["args"])
        except StorageError:
            message = "暂时无法读取测评或练习数据，请稍后重试。"
            update, observation = {"tool_error": message}, {"status": "failed", "message": message}
        except (ValidationError, InvalidPlan) as error:
            validation = plan_validation(error)
            update = {}
            if name == BUILD:
                attempts = state.get("invalid_drafts", 0) + 1
                update = {"validation": validation, "invalid_drafts": attempts}
                if attempts >= 2:
                    update["failure"] = "invalid_model_plan"
            observation = {"status": "invalid_plan" if name == BUILD else "rejected",
                           "validation": validation, "message": VALIDATION_HINTS[validation["check"]]}
        action = {"tool": name if name in TOOL_NAMES else "unknown_tool", "status": observation["status"]}
        if "validation" in observation:
            action["validation"] = observation["validation"]
        return {**update, "actions": [*state.get("actions", []), action],
                "messages": [*state["messages"], {"role": "tool", "tool_call_id": call["id"],
                                                 "content": json.dumps(observation, ensure_ascii=False)}]}

    graph = StateGraph(PracticeState)
    graph.add_node("decide", decide)
    graph.add_node("act_and_observe", act)
    graph.add_edge(START, "decide")
    graph.add_conditional_edges("decide", lambda state: END if state.get("failure") else
                                "act_and_observe" if state.get("call") else "decide",
                                [END, "act_and_observe", "decide"])
    graph.add_conditional_edges("act_and_observe", lambda state: END if state.get("status") or state.get("failure")
                                else "decide", [END, "decide"])
    compiled = graph.compile()

    def run(req, *, model=None, upstream=None, session_id=None):
        req = dict(req)
        if message := apply_time_constraint(req):
            return {"status": "needs_input", "answer": message, "planning_agent": {
                "status": "needs_input", "turns": 0, "actions": []}}
        query = req.get("query", "")
        without_report = bool(WITHOUT_REPORT.search(query))
        current_report = bool(LINKED_CUE.search(query))
        historical_report = bool(PAST_REPORT_CUE.search(query))
        asks_for_report = current_report or historical_report or bool(REPORT_REQUEST.search(query))
        named_report = asks_for_report and bool(SONG_TITLE.search(query) or req.get("song_title"))
        overrides_session_report = req.get("assessment_source") == "session" and (
            historical_report or named_report or (not asks_for_report and BASELINE_REQUEST.search(query))
        )
        if upstream is None and (without_report or overrides_session_report):
            req.pop("assessment_job_id", None)
        bound = upstream is not None or bool(req.get("assessment_job_id"))
        required = bound or (not without_report and asks_for_report)
        scope = "linked" if bound or current_report else "history"
        initial: PracticeState = {"request": req, "upstream": upstream, "session_id": session_id, "required": required,
                                  "scope": scope, "basis": basis_for(req)}
        state = compiled.invoke({**initial, "model": model}) if model is not None else initial
        plan, message = state.get("plan"), state.get("needs_input")
        tool_error = state.get("tool_error")
        if not state.get("status") and not message and not plan and not tool_error:
            basis = state["basis"]
            if required and not state.get("report_ref"):
                report_id, report, message = lookup_report(store, state, SearchArgs(scope=scope))
                if report:
                    basis = basis_for(req, report, report_id)
            elif (model is None and not without_report and not BASELINE_REQUEST.search(query)
                  and not goal_focus(req.get("goal") or query) and req.get("focus") in (None, "general") and memory):
                if latest := memory.latest_assessment(req["melody_id"]):
                    report_id = latest["report_id"]
                    job = store.job(report_id)
                    basis = basis_for(req, {**latest, **((job or {}).get("result") or {})}, report_id)
            if not message:
                reason = state.get("failure", "model_not_configured")
                generation = {"mode": "rules", "reason": reason}
                if validation := state.get("validation"):
                    generation.update(reason="invalid_model_plan", validation=validation)
                plan = {**basis.plan, "generation": generation}
        status = "failed" if tool_error else "needs_input" if message else "completed"
        trace = {"status": status, "turns": state.get("turns", 0), "actions": state.get("actions", []),
                 "finalized_by": "model" if state.get("status") else "program"}
        if message or tool_error:
            return {"status": status, "answer": message or tool_error, "planning_agent": trace}
        return {"status": status, "plan": plan, "planning_agent": trace, "answer":
                f"已安排 {plan['days']} 天练习，每天 {plan['minutes_per_day']} 分钟。重点是{plan['focus_label']}。"}

    return run
