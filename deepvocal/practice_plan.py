"""Build practice actions and retest criteria from a recording assessment."""

from dataclasses import dataclass
from math import isfinite
import re
from statistics import median

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .audio import MELODIES
from .knowledge import EXERCISES


FEEDBACK_FIELDS = {
    "pitch_flat": "intonation", "pitch_sharp": "intonation", "timing": "rhythm",
    "stability": "stability", "register": "summary", "general": "summary",
}
FINDINGS = {
    "pitch_flat": r"偏低|唱低|低于(?:参考|原曲|目标)",
    "pitch_sharp": r"偏高|唱高|高于(?:参考|原曲|目标)",
    "timing": (r"拖拍|抢拍|拖沓|(?:不自然|非自然)的?停顿"
               r"|(?:起音|进拍|落拍|跟拍|节奏).{0,6}(?:偏晚|偏早|稍晚|稍慢|延迟|不稳)"),
    "stability": r"下坠|下滑|发抖|颤抖|不稳|波动(?:较大|明显)",
    "register": r"破音|(?:换声|声区过渡).{0,6}(?:卡|断|不稳)",
}
CLAUSES = re.compile(r"[，。；！？,.!?;\n]|但是|但")
NEGATED = re.compile(r"(?:没有|未见|未发现|未出现|无(?:明显)?|很少)[^，。；！？]{0,13}$")
LOCATIONS = (("句尾", "尾音|尾句|句尾|收尾|最后一"), ("句首", "起音|句首|起句|开头"), ("持续音", "长音|持续音"))
UNSUPPORTED_CAUSE = re.compile(r"导致|造成|引起|引发|源于|归因|声带张力|喉部病变|气息支撑(?:减弱|不足)")
GOAL_CUES = {
    "pitch_flat": FINDINGS["pitch_flat"], "pitch_sharp": FINDINGS["pitch_sharp"],
    "register": r"换声|破音|声区过渡|假声",
    "timing": FINDINGS["timing"] + r"|节奏|节拍",
    "stability": r"长音.{0,8}(?:稳定|不稳)|稳住长音|尾音.{0,8}(?:下坠|下滑|发抖)",
}

# Each entry contains three progressively longer practice tasks and a listening criterion.
DRILLS = {
    "pitch_flat": ("先听目标音，再用轻声唱准单音。", "带回歌词，连唱目标音前后两个音。", "唱回完整乐句，留意每个目标音是否仍偏低。",
                   "回听目标音，确认起音和持续音较练习前更接近参考音高。"),
    "pitch_sharp": ("先听目标音，再用轻声唱准单音。", "连唱目标音前后两个音，避免把目标音推高。", "唱回完整乐句，留意每个目标音是否仍偏高。",
                    "回听目标音，确认起音和持续音较练习前更接近参考音高。"),
    "timing": ("跟节拍器拍出起音位置，再按拍念词。", "降低练习速度，唱准短句的进拍和收拍。", "恢复参考速度，唱完整乐句。",
               "回听同一位置，检查入拍或拖拍问题是否比练习前减轻。"),
    "stability": ("轻声唱一两个舒适短长音，留意收尾。", "带回歌词唱短句，听持续音和尾音是否平稳。", "连唱完整乐句，保持同样稳定的收尾。",
                  "回听同一个持续音和尾音，检查下坠或抖动是否比练习前减轻。"),
    "register": ("轻声滑过换声位置。", "选择换声处前后两个音，用短音连接。", "把换声处连回完整乐句。",
                 "回听同一换声位置，检查破音或中断是否比练习前减少。"),
    "general": ("在舒适音域唱短音。", "把短音接成完整乐句。", "按参考连唱同一段。",
                 "先保存一份可比较的基准录音，之后回听同一乐句的音准、起音与稳定度。"),
}

STAGES = ("分解练习", "连唱短句", "整段复测")
RATIONALE_DISPLAY_LIMIT = 320
DRILL_NAMES = {
    "pitch_flat": ("目标音校准", "连接目标音", "完整乐句音准"),
    "pitch_sharp": ("目标音校准", "连接目标音", "完整乐句音准"),
    "timing": ("拍点与念词", "慢速短句进拍", "原速乐句对齐"),
    "stability": ("短长音收尾", "短句持续音", "整句稳定收尾"),
    "register": ("轻声滑音", "换声区两音连接", "换声区连回原句"),
    "general": ("舒适短音", "短音连接", "同段基准录音"),
}


class PlanningOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class PlannedExercise(PlanningOutput):
    evidence_id: str = Field(min_length=1, max_length=30)
    step: int = Field(ge=1, le=3, strict=True)
    minutes: int = Field(ge=1, le=90, strict=True)
    instructions: str = Field(min_length=1, max_length=160)


class PlannedDay(PlanningOutput):
    day: int = Field(ge=1, le=14, strict=True)
    objective: str = Field(min_length=1, max_length=90)
    exercises: list[PlannedExercise] = Field(min_length=1, max_length=2)


class ModelPlan(PlanningOutput):
    rationale: str = Field(min_length=1)
    minutes_per_day: int = Field(ge=5, le=90, strict=True)
    sessions: list[PlannedDay] = Field(min_length=1, max_length=14)


class InvalidPlan(ValueError):
    """A well-formed plan failed a training constraint."""


def finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and isfinite(value)


def goal_focus(text):
    return next((issue for issue, cue in GOAL_CUES.items() if re.search(cue, text)), None)


def reported_observation(feedback, issue):
    description = feedback.get(FEEDBACK_FIELDS[issue]) or ""
    pattern = FINDINGS.get(issue)
    if not pattern:
        return description
    for observation in (description, feedback.get("summary") or ""):
        for clause in CLAUSES.split(observation):
            match = re.search(pattern, clause)
            if match and not NEGATED.search(clause[:match.start()]):
                return clause.strip()
    return ""


def measured_baselines(report):
    baselines = {}
    if finite_number(pitch := report.get("median_pitch_error_cents")):
        baselines["pitch"] = ("median_pitch_error_cents", pitch, "音分", "closer_to_zero")
    for field, key, unit in (("onset_error_ms", "timing", "毫秒"), ("pitch_iqr_cents", "stability", "音分")):
        values = [abs(note[field]) for note in report.get("notes", []) if finite_number(note.get(field))]
        if len(values) >= 2:
            baselines[key] = (f"median_absolute_{field}" if key == "timing" else "median_pitch_iqr_cents",
                              round(median(values), 1), unit, "decrease")
    return baselines


def retest_setup(req, report):
    settings = report.get("assessment_settings") or {}
    if reference := report.get("reference"):
        setup = {"title": reference["title"], "segment_id": reference.get("segment_id"),
                 "transpose": reference.get("transpose")}
        start = reference.get("segment_start_seconds", reference.get("start_seconds"))
        end = reference.get("segment_end_seconds", reference.get("end_seconds"))
        if finite_number(start) and finite_number(end) and 0 <= start < end:
            setup.update(start_seconds=start, end_seconds=end, audio_id=reference.get("audio_id"),
                         alignment_mode="llm" if report.get("assessment_method") == "audio_model" else "signal")
        return setup
    melody = report.get("melody_id") or req["melody_id"]
    if not report and melody == "warmup":
        return {"title": "自选一句舒适的乐句"}
    return {"title": MELODIES[melody]["title"], "melody_id": melody,
            "bpm": settings.get("bpm", report.get("bpm")), "transpose": settings.get("transpose", report.get("transpose"))}


def session_overhead(minutes):
    """Reserve the same time for warmup and recording a comparison take."""
    return max(1, minutes // 5)


def allocate_practice_minutes(budget, exercises):
    if len(exercises) == 1:
        return (budget,)
    first, second = (exercise.minutes for exercise in exercises)
    share = (budget * first + (first + second) // 2) // (first + second)
    first_minutes = min(budget - 1, max(1, share))
    return first_minutes, budget - first_minutes


def practice_action(row, step, minutes, version, *, single=False, detail=""):
    issue = row["id"]
    source_id = EXERCISES[issue][0]
    name = DRILL_NAMES[issue][step - 1]
    action = DRILLS[issue][step - 1]
    if single and issue != "general":
        name += "与原句应用"
        action += DRILLS[issue][2]
    if location := row["location"]:
        action = f"以{location}为重点。{action}"
    if detail and detail.rstrip("。！？.!?") not in action:
        action += " " + detail
    return {"name": name, "minutes": minutes, "instructions": action, "source_id": source_id,
            "skill_version": version, "evidence_ids": [issue], "step": step}


def practice_session(day, step, minutes, actions, *, single=False, objective=None):
    overhead = session_overhead(minutes)
    warmup = {"name": "舒适音区热身", "minutes": overhead,
              "instructions": EXERCISES["general"][2], "source_id": "lip"}
    review = {"name": "同段复测", "minutes": overhead,
              "instructions": "唱回完整乐句，按下方复测标准录制并回听。", "source_id": "phrases",
              "evidence_ids": [action["evidence_ids"][0] for action in actions]}
    return {"day": day, "stage": "目标练习与复测" if single else STAGES[step - 1],
            **({"objective": objective} if objective else {}),
            "exercises": [warmup, *actions, review]}


@dataclass(frozen=True)
class PracticeBasis:
    plan: dict
    context: dict
    instructions: dict
    day_range: tuple[int, int]
    minute_range: tuple[int, int]


def planning_context(plan, req, report, instructions, day_range, minute_range, *, include_secondary):
    retest = plan["retest"]
    reference = {key: retest["reference"][key] for key in ("title", "start_seconds", "end_seconds", "bpm", "transpose")
                 if retest["reference"].get(key) is not None}
    allowed = {row["id"]: row for row in plan["evidence"]}
    feedback_fields = ("summary", "intonation", "rhythm", "stability") if include_secondary else (
        FEEDBACK_FIELDS[plan["focus"]],
    )
    return {
        "task": req.get("query", ""), "goal": req.get("goal"), "primary_evidence_id": plan["focus"],
        "schedule": {"days_range": day_range, "minutes_per_day_range": minute_range,
                     "suggested_days": plan["days"], "suggested_minutes_per_day": plan["minutes_per_day"]},
        "reference": reference, "evidence": plan["evidence"], "retest": retest["criteria"],
        "assessment_feedback": {key: value for key, value in (report.get("audio_feedback") or {}).items()
                                if key in feedback_fields},
        "approved_skills": [{"evidence_id": issue, "instructions": instructions[issue][1],
                             "steps": dict(enumerate(DRILLS[issue][:3], 1))} for issue in allowed],
    }


def compile_plan(basis, draft, model_name):
    proposal = ModelPlan.model_validate(draft)
    plan, instructions = basis.plan, basis.instructions
    day_range, minute_range = basis.day_range, basis.minute_range
    allowed = {row["id"]: row for row in plan["evidence"]}
    days, minutes = len(proposal.sessions), proposal.minutes_per_day
    if (not day_range[0] <= days <= day_range[1] or not minute_range[0] <= minutes <= minute_range[1]
            or [day.day for day in proposal.sessions] != list(range(1, days + 1))):
        raise InvalidPlan("schedule")
    drill_minutes = minutes - 2 * session_overhead(minutes)
    sessions = []
    last_steps, details = {}, {}
    # A general assessment plan can have a single secondary finding. If omitted, add its first drill.
    proposed_ids = {exercise.evidence_id for draft in proposal.sessions for exercise in draft.exercises}
    supplemental = next(iter(allowed.keys() - proposed_ids - {plan["focus"]}), None)
    for draft in proposal.sessions:
        ids = {exercise.evidence_id for exercise in draft.exercises}
        if len(ids) != len(draft.exercises) or not ids <= allowed.keys() or plan["focus"] not in ids:
            raise InvalidPlan("exercise_selection")
        actions = []
        allocations = allocate_practice_minutes(drill_minutes, draft.exercises)
        for exercise, allocated in zip(draft.exercises, allocations, strict=True):
            issue, step = exercise.evidence_id, exercise.step
            previous = last_steps.get(issue, 0)
            signature = re.sub(r"[\W_]+", "", exercise.instructions)
            seen = details.setdefault(issue, set())
            if step < previous or (days <= 3 and step == previous) or not signature or signature in seen:
                raise InvalidPlan("exercise_progression")
            if UNSUPPORTED_CAUSE.search(exercise.instructions):
                raise InvalidPlan("unsupported_cause")
            if issue == plan["focus"] and ((draft.day == 1 and step != 1)
                                           or (days > 2 and draft.day == days and step != 3)):
                raise InvalidPlan("primary_progression")
            last_steps[issue] = step
            seen.add(signature)
            actions.append(practice_action(allowed[issue], step, allocated, instructions[issue][0],
                                           single=days == 1, detail=exercise.instructions))
        if supplemental and draft.day == 1:
            supplemental_minutes = max(1, drill_minutes // 3)
            actions[0]["minutes"] -= supplemental_minutes
            actions.append(practice_action(allowed[supplemental], 1, supplemental_minutes,
                                           instructions[supplemental][0], single=days == 1))
        sessions.append(practice_session(draft.day, last_steps[plan["focus"]], minutes, actions,
                                         single=days == 1, objective=draft.objective))
    rationale = proposal.rationale
    if UNSUPPORTED_CAUSE.search(rationale):
        priority = "、".join(item["label"] for item in plan["evidence"])
        first_goal = sessions[0]["objective"].rstrip("。！？.!?")
        rationale = f"这次围绕{priority}安排专项练习。第一天的目标是{first_goal}。每次都用同一片段按下方标准复测。"
    if len(rationale) > RATIONALE_DISPLAY_LIMIT:
        prefix = rationale[:RATIONALE_DISPLAY_LIMIT - 1].rstrip()
        end = max(prefix.rfind(mark) for mark in "。！？")
        rationale = prefix[:end + 1] if end >= 0 else prefix + "…"
    return {**plan, "days": days, "minutes_per_day": minutes, "sessions": sessions, "rationale": rationale,
            "generation": {"mode": "model", "model": model_name}}


def plan_validation(error):
    if isinstance(error, ValidationError):
        detail = error.errors(include_context=False, include_input=False, include_url=False)[0]
        path = detail["loc"]
        if detail["type"] == "extra_forbidden":
            path = (*path[:-1], "<extra>")
        return {"check": "schema", "field": ".".join(map(str, path)), "code": detail["type"]}
    return {"check": str(error)}


def prepare_evidence_practice(skills, req, report, focus, labels, *, include_secondary=True, preferred_minutes=None):
    """Produce linked evidence, daily actions and retest criteria for the plan node."""
    report = report or {}
    feedback = report.get("audio_feedback") or {}
    baselines = measured_baselines(report)
    diagnosis = report.get("issue")
    observations = {issue: reported_observation(feedback, issue) for issue in FINDINGS}
    findings = [issue for issue, observation in observations.items() if observation]
    if focus == "general" and findings:
        focus = findings[0]
    other_pitch = {"pitch_flat": "pitch_sharp", "pitch_sharp": "pitch_flat"}.get(focus)
    # The report's primary diagnosis is considered before supplementary observations.
    secondary = next((issue for issue in (diagnosis, *findings)
                      if issue in FINDINGS and issue not in (focus, other_pitch)), None) if include_secondary else None
    targets = [focus, secondary] if secondary else [focus]
    evidence, criteria, instructions = [], [], {}
    for issue in targets:
        observed = observations.get(issue) or (feedback.get("summary") if issue == diagnosis else "")
        backed_by_report = bool(report) and (issue == diagnosis or bool(observed))
        source = "assessment" if backed_by_report else "goal" if issue != "general" else "baseline"
        location = next((name for name, pattern in LOCATIONS if re.search(pattern, observed or "")), None)
        text = observed or (labels[issue] if source in ("assessment", "goal")
                            else "先录制同一乐句，建立复测基准。")
        row = {"id": issue, "label": labels[issue], "source": source, "observation": text, "location": location}
        metric = baselines.get("pitch" if issue in ("pitch_flat", "pitch_sharp") else issue) if backed_by_report else None
        criterion = {"evidence_id": issue, "method": "listening", "standard": DRILLS[issue][3]}
        if metric:
            key, value, unit, comparator = metric
            baseline = {"metric": key, "value": value, "unit": unit}
            row["baseline"] = baseline
            if comparator == "closer_to_zero":
                standard = f"同段音高偏差中位数较基线 {value:+g} {unit}更接近 0。"
            elif issue == "timing":
                standard = f"同一批可测起音的绝对偏差中位数低于 {value:g} {unit}。"
            else:
                standard = f"同一批持续音的音高波动中位数低于 {value:g} {unit}。"
            if value == 0:
                standard = f"同段对应指标保持在基线 0 {unit}附近，并回听确认稳定。"
            criterion.update(method="measurement", baseline=baseline, comparator=comparator if value else "maintain",
                             standard=standard)
        if location:
            criterion["standard"] = f"重点复查{location}。{criterion['standard']}"
        evidence.append(row)
        criteria.append(criterion)
        instructions[issue] = skills.load(issue)

    setup = retest_setup(req, report)
    conditions = "用同一片段、参考调、速度和测评方式，在相近录音条件下连录两遍。"
    if any(item["method"] == "measurement" for item in criteria):
        conditions += "对照相同的可测音符或起音范围。"
    focused = focus != "general"
    days = req.get("days") or (3 if secondary else 2 if focused else 1)
    automatic_minutes_limit = max(5, min(20, preferred_minutes)) if preferred_minutes else 20
    if req.get("minutes_cap"):
        automatic_minutes_limit = min(req["minutes_cap"], 90)
    suggested_minutes = preferred_minutes or (12 if secondary else 10 if focused else 8)
    minutes = req.get("minutes") or min(max(5, suggested_minutes), automatic_minutes_limit)
    day_range = (days, days) if req.get("days") or not focused else (2 if secondary else 1, 3)
    minute_range = (minutes, minutes) if req.get("minutes") else (5, automatic_minutes_limit)
    practice_minutes = minutes - 2 * session_overhead(minutes)
    secondary_minutes = max(1, practice_minutes // 3) if secondary else 0
    sessions = []
    for day in range(1, days + 1):
        step = 3 if day == days and days > 1 else 1 + (day - 1) * 3 // days
        actions = [practice_action(row, step, practice_minutes - secondary_minutes if row["id"] == focus else secondary_minutes,
                                   instructions[row["id"]][0], single=days == 1,
                                   detail=instructions[row["id"]][1] if day == 1 and instructions[row["id"]][0] > 1 else "")
                   for row in evidence]
        sessions.append(practice_session(day, step, minutes, actions, single=days == 1))
    plan = {"focus": focus, "skill_version": instructions[focus][0], "evidence": evidence, "sessions": sessions,
            "days": days, "minutes_per_day": minutes,
            "retest": {"reference": setup, "conditions": conditions, "criteria": criteria,
                       "decision": ("两遍均达到主项标准后进入下一阶段。主项仍不稳定时，保留同一乐句，回到当前专项再练。"
                                    if focused else "保留基准录音。下一次用同一乐句复测，再按反复出现的问题安排专项。")}}
    context = planning_context(plan, req, report, instructions, day_range, minute_range, include_secondary=include_secondary)
    return PracticeBasis(plan, context, instructions, day_range, minute_range)
