const field = id => document.getElementById(id);
const modes = {
  qa: {
    title: "声乐问答", description: "提一个具体问题，查看导师的回答和引用资料。",
    label: "你的问题", placeholder: "例如：换声区容易破音，应该怎么练？", send: "发送问题",
    tipTitle: "试着这样问", tip: "从一个发声问题开始。回答中的引用可以展开查看原文。",
    examples: ["什么是换声区？", "长音不稳可以怎么练？", "唱歌总是拖拍，怎么找节奏？"],
  },
  assessment: {
    title: "录音测评", description: "选择参考旋律，上传清唱录音，查看逐音偏差。",
    label: "补充说明（可选）", placeholder: "例如：我想看看后半段的音准。", send: "开始测评",
    tipTitle: "录音之前", tip: "先试听参考旋律。清唱时保持相同速度和调性，并避免把参考音录进来。",
    examples: [],
  },
  practice_plan: {
    title: "练习规划", description: "设置练习时间，按天查看热身、专项练习和复测安排。",
    label: "练习目标（可选）", placeholder: "例如：这周想练稳换声区。", send: "生成练习计划",
    tipTitle: "从一个目标开始", tip: "选定一个最想改善的问题，每天用同一段旋律复测，便于对比。",
    examples: ["练稳换声区", "改善拖拍", "练习长音稳定性"],
  },
  singing: {
    title: "示范音频", description: "选择旋律、速度与调性，生成可播放和下载的参考音。",
    label: "补充说明（可选）", placeholder: "例如：为今天的练习生成参考音。", send: "生成音频",
    tipTitle: "怎么使用参考音", tip: "先听旋律，再跟唱。需要改变音高时，可以设置移调半音数。",
    examples: [],
  },
};
const providerLabels = {local: "本地 / 服务端", openai: "OpenAI", deepseek: "DeepSeek", dashscope: "阿里云百炼", custom: "自定义服务"};
let mode = "qa";
let activeConfig = {provider: "local", model: "", key_saved: false, token: "", verified: false};
let sessionId;
let recording;
let uploadedRecording;
let selectedAssessment;
let pendingJob;
let busy = false;
let guideURL;
let guideRevision = 0;
const audioURLs = new Set();
const sessionStorageKey = "deepvocal.last-session";
const sessionDate = new Intl.DateTimeFormat("zh-CN", {month: "numeric", day: "numeric", hour: "2-digit", minute: "2-digit"});
let sessions = [];
let nextSessionOffset = null;
let nextOlderTurn = null;
let sessionListRequest = 0;

function make(tag, cls, text) {
  const element = document.createElement(tag);
  if (cls) element.className = cls;
  if (text != null) element.textContent = text;
  return element;
}

async function request(url, {payload, form, config = activeConfig, binary, method} = {}) {
  const headers = {};
  if (config.token) headers["X-DeepVocal-Token"] = config.token;
  if (config.provider !== "local" && config.key && payload) headers["X-Model-API-Key"] = config.key;
  if (payload) headers["Content-Type"] = "application/json";
  const response = await fetch(url, {
    method: method ?? (payload || form ? "POST" : "GET"), headers,
    body: payload ? JSON.stringify(payload) : form, cache: "no-store", redirect: "error",
  });
  if (response.status === 204) return;
  if (response.ok) return binary ? response.blob() : response.json();
  const detail = (await response.json().catch(() => ({}))).detail;
  const error = new Error(typeof detail === "string" ? detail : Array.isArray(detail)
    ? detail.map(item => item.loc.at(-1) + " · " + item.msg).join("；")
    : "请求失败（HTTP " + response.status + "）。");
  error.status = response.status;
  throw error;
}

function taskStatus(text, state = "") {
  field("task-status").textContent = text;
  field("task-feedback").hidden = !text;
  field("task-feedback").dataset.state = state;
  field("resume-job").hidden = busy || !pendingJob;
}

function setBusy(value) {
  busy = value;
  field("task-fields").disabled = value;
  field("auto-route").disabled = value;
  field("send").disabled = value || Boolean(pendingJob);
  field("new-session").disabled = value;
  field("open-settings").disabled = value;
  field("hero-assessment").disabled = value;
  field("older-turns").disabled = value;
  document.querySelectorAll("[data-mode], .suggestion, .history-item, .history-delete").forEach(button => { button.disabled = value; });
  field("resume-job").hidden = value || !pendingJob;
  field("chat").setAttribute("aria-busy", value);
}

function updatePanels() {
  const isAssessment = mode === "assessment";
  const isPractice = mode === "practice_plan";
  const isSinging = mode === "singing";
  field("reference-fields").hidden = mode === "qa";
  field("recording-fields").hidden = !isAssessment;
  field("plan-fields").hidden = !isPractice && !(isAssessment && field("also-plan").checked);
  field("lyrics-group").hidden = !isSinging;
  field("auto-group").hidden = mode !== "qa";
  for (const id of ["tempo-group", "transpose-group", "guide-group"]) field(id).hidden = isPractice;
  for (const id of ["bpm", "transpose"]) field(id).disabled = !isSinging && !isAssessment;
  field("offset").disabled = !isAssessment;
  for (const id of ["days", "minutes"]) field(id).disabled = field("plan-fields").hidden;
  field("query").required = mode === "qa";
  field("use-assessment-group").hidden = !isPractice || selectedAssessment?.melody !== field("melody").value;
}

function selectMode(next, focus = false) {
  mode = next;
  const config = modes[mode];
  field("task-title").textContent = config.title;
  field("task-description").textContent = config.description;
  field("query-label").textContent = config.label;
  field("query").placeholder = config.placeholder;
  field("send-label").textContent = config.send;
  field("tips-title").textContent = config.tipTitle;
  field("tips-text").textContent = config.tip;
  field("suggestions").replaceChildren(...config.examples.map(example => {
    const button = make("button", "suggestion", example);
    button.type = "button";
    button.addEventListener("click", () => {
      field("query").value = example;
      field("query").focus();
    });
    button.disabled = busy;
    return button;
  }));
  document.querySelectorAll("[data-mode]").forEach(button => button.setAttribute("aria-pressed", button.dataset.mode === mode));
  updatePanels();
  if (focus) field("task-title").focus();
}

document.querySelectorAll("[data-mode]").forEach(button => {
  button.addEventListener("click", () => selectMode(button.dataset.mode, true));
});
field("hero-assessment").addEventListener("click", () => selectMode("assessment", true));
field("also-plan").addEventListener("change", updatePanels);
field("melody").addEventListener("change", updatePanels);

let verified = false;
let savingSettings = false;
let testingModel = false;
function updateSettings() {
  const provider = field("provider").value;
  const savedKey = provider === activeConfig.provider && activeConfig.key_saved;
  field("provider-fields").hidden = provider === "local";
  field("test").hidden = provider === "local";
  field("custom-hint").hidden = provider !== "custom";
  field("key-hint").hidden = !savedKey;
  field("model").required = provider !== "local";
  field("key").required = provider !== "local" && provider !== "custom" && !savedKey;
  field("key").placeholder = savedKey ? "已保存，输入新 Key 可替换" : "";
}
function modelStatus(text, state = "") {
  field("model-status").textContent = text;
  field("model-status").hidden = !text;
  field("model-status").dataset.state = state;
}
function draftConfig() {
  const provider = field("provider").value;
  return {provider, model: provider === "local" ? "" : field("model").value.trim(),
    key: provider === "local" ? "" : field("key").value.trim(), token: field("token").value.trim(), verified};
}
function applyModelSettings(settings, token, connectionVerified = false) {
  activeConfig = {...settings, token, verified: connectionVerified};
  const status = connectionVerified ? " · 已验证" : settings.key_saved ? " · Key 已保存" : "";
  field("connection-label").textContent = providerLabels[settings.provider] + status;
  field("connection").dataset.connected = String(connectionVerified);
}

async function restoreModelSettings() {
  const previous = activeConfig;
  try {
    const settings = await request("/api/model/settings", {config: previous});
    if (previous === activeConfig) {
      const sameModel = previous.provider === settings.provider && previous.model === settings.model;
      applyModelSettings(settings, previous.token, sameModel && previous.verified);
    }
  } catch (error) {
    if (previous === activeConfig && error.status === 401) {
      field("connection-label").textContent = "需要访问令牌";
    }
  }
}
async function refreshHealth() {
  const snapshot = activeConfig;
  try {
    const result = await request("/api/health", {config: snapshot});
    if (snapshot === activeConfig) field("singing-hint").textContent = result.singing_configured
      ? "歌声合成服务已配置。歌词留空时生成参考音。"
      : "当前服务未配置歌词合成，留空可生成参考音。";
  } catch (error) {
    if (snapshot === activeConfig && error.status === 401) {
      field("connection-label").textContent = "需要访问令牌";
      field("connection").dataset.connected = "false";
    }
  }
}
field("open-settings").addEventListener("click", async () => {
  await restoreModelSettings();
  if (busy || field("settings-dialog").open) return;
  for (const id of ["provider", "model", "token"]) field(id).value = activeConfig[id];
  field("key").value = "";
  verified = activeConfig.verified;
  updateSettings();
  modelStatus(verified ? "连接已验证。" : "");
  field("settings-dialog").showModal();
});
field("close-settings").addEventListener("click", () => field("settings-dialog").close());
field("settings-dialog").addEventListener("cancel", event => {
  if (savingSettings) event.preventDefault();
});
field("settings-dialog").addEventListener("close", () => {
  field("key").value = "";
  field("token").value = "";
});
field("provider").addEventListener("change", () => {
  const provider = field("provider").value;
  field("model").value = provider === activeConfig.provider ? activeConfig.model : provider === "dashscope" ? "qwen-plus" : "";
  field("key").value = "";
  updateSettings();
});
field("settings").addEventListener("input", () => {
  verified = false;
  modelStatus("");
});
field("test").addEventListener("click", async () => {
  if (savingSettings || testingModel || !field("settings").reportValidity()) return;
  const config = draftConfig();
  const useSaved = !config.key && config.provider === activeConfig.provider && config.provider !== "local";
  if (!config.model || (config.provider !== "custom" && !config.key && !activeConfig.key_saved)) {
    modelStatus("请填写模型 ID 和 API Key。", "error");
    return;
  }
  testingModel = true;
  field("test").disabled = true;
  modelStatus("正在测试连接…");
  try {
    await request("/api/model/test", {payload: {provider: config.provider, model: config.model, use_saved_model: useSaved}, config});
    const draft = draftConfig();
    if (field("settings-dialog").open && ["provider", "model", "key", "token"].every(key => draft[key] === config[key])) {
      verified = true;
      modelStatus("连接成功，可以保存设置。");
    }
  } catch (error) {
    const draft = draftConfig();
    if (field("settings-dialog").open && ["provider", "model", "key", "token"].every(key => draft[key] === config[key])) {
      modelStatus(error.message, "error");
    }
  } finally {
    testingModel = false;
    field("test").disabled = savingSettings;
  }
});
field("settings").addEventListener("submit", async event => {
  event.preventDefault();
  if (savingSettings) return;
  const config = draftConfig();
  const savedKey = config.provider === activeConfig.provider && activeConfig.key_saved;
  if (config.provider !== "local" && (!config.model || (config.provider !== "custom" && !config.key && !savedKey))) {
    modelStatus("请填写模型 ID 和 API Key。", "error");
    return;
  }
  savingSettings = true;
  for (const id of ["save-model", "test", "close-settings"]) field(id).disabled = true;
  modelStatus("正在保存设置…");
  try {
    const backend = "/api/model/settings";
    const settings = config.provider === "local" && activeConfig.provider === "local" && config.token !== activeConfig.token
      ? await request(backend, {config})
      : await request(backend, {method: "PUT", payload: {provider: config.provider, model: config.model || null}, config});
    applyModelSettings(settings, config.token, config.verified);
    field("settings-dialog").close();
    refreshHealth();
    loadSessions({restore: true});
  } catch (error) {
    modelStatus(error.message, "error");
  } finally {
    savingSettings = false;
    field("save-model").disabled = field("close-settings").disabled = false;
    field("test").disabled = testingModel;
  }
});

function setRecording(file) {
  if (file && (!/\.wav$/i.test(file.name) || file.size === 0 || file.size > 12 * 1024 * 1024)) {
    field("recording").value = "";
    setRecording();
    taskStatus("请选择 12 MiB 以内的 WAV 录音。", "error");
    return;
  }
  recording = file;
  uploadedRecording = undefined;
  field("file-name").textContent = file?.name || "选择或拖入清唱录音";
  field("file-detail").textContent = file ? (file.size / 1024).toFixed(1) + " KiB · WAV" : "16 位 PCM WAV · 最长 60 秒 · 最大 12 MiB";
  field("remove-recording").hidden = !file;
  if (!pendingJob) taskStatus("");
}
field("recording").addEventListener("change", event => setRecording(event.target.files[0]));
field("remove-recording").addEventListener("click", () => {
  field("recording").value = "";
  setRecording();
  field("recording").focus();
});
const dropZone = field("drop-zone");
dropZone.addEventListener("dragover", event => {
  event.preventDefault();
  if (!busy) dropZone.classList.add("dragging");
});
dropZone.addEventListener("dragleave", event => {
  if (!dropZone.contains(event.relatedTarget)) dropZone.classList.remove("dragging");
});
dropZone.addEventListener("drop", event => {
  event.preventDefault();
  dropZone.classList.remove("dragging");
  if (!busy && event.dataTransfer.files.length) {
    field("recording").value = "";
    setRecording(event.dataTransfer.files[0]);
  }
});

function clearGuide() {
  guideRevision++;
  field("guide-output").replaceChildren();
  if (guideURL) {
    URL.revokeObjectURL(guideURL);
    audioURLs.delete(guideURL);
    guideURL = undefined;
  }
}
function melodySpec() {
  return {melody_id: field("melody").value, bpm: Number(field("bpm").value), transpose: Number(field("transpose").value)};
}
function addAudio(parent, blob, filename) {
  const url = URL.createObjectURL(blob);
  audioURLs.add(url);
  const player = make("audio");
  player.controls = true;
  player.src = url;
  player.setAttribute("aria-label", filename ? "生成的音频" : "参考旋律");
  const controls = make("div", "audio-playback");
  controls.append(player);
  if (filename) {
    const link = make("a", "button quiet", "下载 WAV");
    link.href = url;
    link.download = filename;
    controls.append(link);
  }
  parent.append(controls);
  return {url, player};
}
for (const id of ["melody", "bpm", "transpose"]) field(id).addEventListener("input", clearGuide);
field("guide").addEventListener("click", async () => {
  if (!["bpm", "transpose"].every(id => field(id).reportValidity())) return;
  const {melody_id, bpm, transpose} = melodySpec();
  const version = guideRevision;
  field("guide").disabled = true;
  try {
    const blob = await request("/api/melodies/" + melody_id + "/guide.wav?" + new URLSearchParams({bpm, transpose}), {binary: true});
    if (version !== guideRevision) return;
    clearGuide();
    const audio = addAudio(field("guide-output"), blob);
    guideURL = audio.url;
    audio.player.play().catch(() => {});
  } catch (error) {
    if (version === guideRevision) taskStatus(error.message, "error");
  } finally {
    field("guide").disabled = false;
  }
});

function renderEvidence(parent, body, answer, evidence) {
  const panel = make("section", "result sources");
  panel.append(make("h3", "", "引用资料 · " + evidence.length));
  const detailsById = new Map();
  for (const [index, source] of evidence.entries()) {
    const details = make("details");
    const metadata = make("p", "source-meta", source.source);
    if (/^https?:\/\//i.test(source.source_url || "")) {
      const original = make("a", "", "查看原文");
      original.href = source.source_url;
      original.target = "_blank";
      original.rel = "noopener noreferrer";
      metadata.append(" · ", original);
    }
    details.append(make("summary", "", (index + 1) + ". " + source.title),
      make("p", "", source.text), metadata);
    if (source.path?.length) {
      const paths = make("ul", "source-meta exercises");
      for (const [from, relation, to] of source.path) paths.append(make("li", "", from + " → " + relation + " → " + to));
      details.append(paths);
    }
    detailsById.set(source.id, {details, index});
    panel.append(details);
  }
  for (const part of answer.split(/(\[[a-z0-9_-]+\])/gi)) {
    const match = detailsById.get(part.slice(1, -1));
    if (part.startsWith("[") && match) {
      const button = make("button", "source-link", "[" + (match.index + 1) + "]");
      button.type = "button";
      button.setAttribute("aria-label", "查看引用 " + (match.index + 1));
      button.addEventListener("click", () => {
        match.details.open = true;
        match.details.querySelector("summary").focus();
      });
      body.append(button);
    } else body.append(document.createTextNode(part));
  }
  parent.append(panel);
}

const signed = (value, unit) => Number.isFinite(value) ? (value > 0 ? "+" : "") + value + " " + unit : "—";
function renderAssessment(parent, report, jobId) {
  const panel = make("section", "result");
  panel.append(make("h3", "", "音准与节奏"));
  const metrics = make("dl", "metrics");
  for (const [label, value] of [
    ["有效音符", report.matched_notes + " / " + report.total_notes],
    ["音高偏差中位数", signed(report.median_pitch_error_cents, "音分")],
    ["±50 音分内的帧", report.within_50_cents_ratio == null ? "—" : Math.round(report.within_50_cents_ratio * 100) + "%"],
  ]) {
    const metric = make("div");
    metric.append(make("dt", "", label), make("dd", "", value));
    metrics.append(metric);
  }
  panel.append(metrics);
  if (report.notes?.length) {
    const details = make("details", "note-detail");
    details.append(make("summary", "", "查看 " + report.notes.length + " 个音符的偏差"));
    const wrapper = make("div");
    const table = make("table");
    const heading = table.createTHead().insertRow();
    for (const label of ["音符", "参考起点", "音高偏差", "起音偏差"]) {
      const cell = make("th", "", label);
      cell.scope = "col";
      heading.append(cell);
    }
    const tbody = table.createTBody();
    for (const note of report.notes) {
      const row = tbody.insertRow();
      const name = ["C", "C♯", "D", "D♯", "E", "F", "F♯", "G", "G♯", "A", "A♯", "B"][note.target_midi % 12] + (Math.floor(note.target_midi / 12) - 1);
      row.append(make("td", "", note.note + " · " + name), make("td", "", note.start_seconds + " s"),
        make("td", Math.abs(note.pitch_error_cents) > 50 ? "outside" : "", signed(note.pitch_error_cents, "音分")),
        make("td", "", signed(note.onset_error_ms, "ms")));
    }
    wrapper.append(table);
    details.append(wrapper);
    panel.append(details);
  }
  const button = make("button", "button quiet", "用这份测评规划练习");
  button.type = "button";
  button.addEventListener("click", () => {
    if (busy) return;
    selectedAssessment = {id: jobId, melody: report.melody_id};
    field("melody").value = report.melody_id;
    field("use-assessment").checked = true;
    selectMode("practice_plan", true);
  });
  if (jobId && !report.plan) panel.append(button);
  parent.append(panel);
}

function renderPlan(parent, plan) {
  const panel = make("section", "result plan");
  panel.append(make("h3", "", plan.days + " 天练习计划 · 每天 " + plan.minutes_per_day + " 分钟"),
    make("p", "plan-intro", "练习重点：" + plan.focus_label + (plan.goal ? " · 目标：" + plan.goal : "")));
  for (const day of plan.sessions) {
    const details = make("details");
    details.open = day.day === 1;
    const list = make("ol", "exercises");
    for (const exercise of day.exercises) {
      const item = make("li", "", exercise.name);
      item.append(make("span", "minutes", exercise.minutes + " 分钟"), make("small", "", exercise.instructions));
      list.append(item);
    }
    details.append(make("summary", "", "第 " + day.day + " 天"), list);
    panel.append(details);
  }
  parent.append(panel);
}
function createMessage(text, own = false, result = {}) {
  const entry = make("article", own ? "message user" : "message");
  entry.append(make("span", "message-label", own ? "你" : "DeepVocal"));
  const body = make("p");
  entry.append(body);
  if (result.evidence?.length) renderEvidence(entry, body, text, result.evidence);
  else body.textContent = text;
  if (result.assessment) renderAssessment(entry, result.assessment, result.assessmentJobId);
  if (result.plan) renderPlan(entry, result.plan);
  return entry;
}

function message(text, own = false, result = {}) {
  const entry = createMessage(text, own, result);
  const transcript = field("transcript");
  transcript.append(entry);
  transcript.scrollTop = entry.offsetTop;
  return entry;
}

function clearConversation() {
  selectedAssessment = pendingJob = undefined;
  nextOlderTurn = null;
  field("older-turns").hidden = true;
  field("chat").reset();
  setRecording();
  clearGuide();
  for (const url of audioURLs) URL.revokeObjectURL(url);
  audioURLs.clear();
  field("transcript").replaceChildren();
  taskStatus("");
}

function rememberSession(id) {
  sessionId = id;
  try {
    if (id) localStorage.setItem(sessionStorageKey, id);
    else localStorage.removeItem(sessionStorageKey);
  } catch { /* The session is still saved on the server. */ }
  updateActiveSession();
}

function rememberedSession() {
  try { return localStorage.getItem(sessionStorageKey); }
  catch { return null; }
}

function updateActiveSession() {
  document.querySelectorAll(".history-item").forEach(button => {
    if (button.dataset.sessionId === sessionId) button.setAttribute("aria-current", "true");
    else button.removeAttribute("aria-current");
  });
}

function historyStatus(text, retry = false) {
  field("history-status").textContent = text;
  field("history-status").hidden = !text;
  field("retry-history").hidden = !retry;
}

function renderSessions() {
  field("history-list").replaceChildren(...sessions.map(session => {
    const item = make("li", "history-row");
    const button = make("button", "history-item");
    button.type = "button";
    button.dataset.sessionId = session.id;
    button.title = session.title;
    button.disabled = busy;
    const date = new Date(session.updated_at * 1000);
    const time = make("time", "", sessionDate.format(date));
    time.dateTime = date.toISOString();
    button.append(make("span", "", session.title), time);
    button.addEventListener("click", () => openSession(session.id));
    const remove = make("button", "history-delete", "删除");
    remove.type = "button";
    remove.disabled = busy;
    remove.setAttribute("aria-label", "删除会话 " + session.title);
    remove.addEventListener("click", () => deleteSession(session, remove));
    item.append(button, remove);
    return item;
  }));
  field("more-sessions").hidden = nextSessionOffset === null;
  updateActiveSession();
}

async function deleteSession(session, trigger) {
  if (busy || !window.confirm(`确定删除「${session.title}」及其全部消息吗？删除后无法恢复。`)) return;
  let focus = trigger;
  setBusy(true);
  historyStatus("正在删除会话…");
  try {
    await request("/api/sessions/" + encodeURIComponent(session.id), {method: "DELETE"});
    // Ignore list responses started before the deletion and keep the loaded pages in sync.
    sessionListRequest += 1;
    const index = sessions.findIndex(item => item.id === session.id);
    if (index >= 0) {
      sessions.splice(index, 1);
      if (nextSessionOffset !== null) nextSessionOffset -= 1;
    }
    field("more-sessions").disabled = false;
    renderSessions();
    if (sessionId === session.id) {
      startNewSession();
      focus = field("query");
    } else {
      if (!sessionId && rememberedSession() === session.id) rememberSession(undefined);
      const buttons = field("history-list").querySelectorAll(".history-item");
      focus = buttons[Math.min(Math.max(index, 0), buttons.length - 1)] ?? field("new-session");
    }
    historyStatus(sessions.length ? "已删除会话。" : "已删除会话。还没有历史对话。");
  } catch (error) {
    historyStatus(error.status ? error.message : "无法删除会话，请重试。");
  } finally {
    setBusy(false);
    focus.focus();
  }
}

async function loadSessions({more = false, restore = false} = {}) {
  const offset = more ? nextSessionOffset : 0;
  if (offset === null) return;
  const requestId = ++sessionListRequest;
  const button = field("more-sessions");
  button.disabled = true;
  if (!sessions.length) historyStatus("正在读取会话…");
  try {
    const page = await request("/api/sessions?" + new URLSearchParams({offset}));
    if (requestId !== sessionListRequest) return;
    sessions = more ? [...sessions, ...page.sessions.filter(item => !sessions.some(saved => saved.id === item.id))] : page.sessions;
    nextSessionOffset = page.next_offset;
    renderSessions();
    historyStatus(sessions.length ? "" : "还没有历史对话。发送消息后会自动保存。");
    const savedId = restore && !sessionId && rememberedSession();
    if (savedId && !busy) await openSession(savedId, true);
  } catch (error) {
    if (requestId !== sessionListRequest) return;
    historyStatus(error.status === 401 ? "请在模型设置中填写服务访问令牌。" : "无法读取历史对话。", true);
  } finally {
    if (requestId === sessionListRequest) button.disabled = false;
  }
}

function audioResult(job, loadNow = false) {
  const kind = job.result.type === "singing" ? "歌声" : "参考音";
  const entry = createMessage(kind + "已生成。");
  const button = make("button", "button quiet", "加载音频");
  button.type = "button";
  const status = make("p", "hint");
  status.setAttribute("role", "status");
  async function load() {
    button.disabled = true;
    status.textContent = "";
    try {
      const blob = await request(job.result.audio_url, {binary: true});
      if (!entry.isConnected) return;
      addAudio(entry, blob, "deepvocal-" + job.result.type + ".wav");
      button.remove();
      status.remove();
    } catch (error) {
      button.disabled = false;
      status.textContent = error.message;
    }
  }
  button.addEventListener("click", load);
  entry.append(button, status);
  if (loadNow) queueMicrotask(load);
  return entry;
}

function jobMessage(job, loadAudio = false) {
  if (job.state === "failed") return createMessage(job.error || "任务未完成。");
  if (job.state === "succeeded") {
    if (job.kind === "assessment") {
      const report = job.result;
      return createMessage(report.summary, false, {assessment: report, assessmentJobId: job.id, plan: report.plan});
    }
    return audioResult(job, loadAudio);
  }
  const entry = createMessage("任务已提交，可以查询最新进度。");
  const button = make("button", "button quiet", "查询任务进度");
  const status = make("p", "hint");
  status.setAttribute("role", "status");
  button.type = "button";
  button.addEventListener("click", async () => {
    button.disabled = true;
    try {
      const current = await request("/api/jobs/" + job.id);
      if (!entry.isConnected) return;
      if (current.state === "succeeded" || current.state === "failed") {
        if (current.state === "succeeded" && current.kind === "assessment" && !selectedAssessment) {
          selectedAssessment = {id: current.id, melody: current.result.melody_id};
          updatePanels();
        }
        const result = jobMessage(current);
        entry.replaceWith(result);
        result.querySelector("button")?.focus();
      } else {
        status.textContent = current.state === "queued" ? "仍在排队。" : "仍在处理。";
      }
    } catch (error) {
      status.textContent = error.message;
    } finally {
      button.disabled = false;
    }
  });
  entry.append(button, status);
  return entry;
}

function showTurns(page, prepend = false) {
  const fragment = document.createDocumentFragment();
  for (const turn of page.turns) {
    fragment.append(createMessage(turn.query, true), createMessage(turn.result.answer, false, turn.result));
    if (turn.job) fragment.append(jobMessage(turn.job));
  }
  if (!selectedAssessment) {
    const assessment = page.turns.slice().reverse().find(turn => turn.job?.kind === "assessment" && turn.job.state === "succeeded");
    if (assessment) selectedAssessment = {id: assessment.job.id, melody: assessment.job.result.melody_id};
  }
  const transcript = field("transcript");
  const previousHeight = transcript.scrollHeight;
  const previousTop = transcript.scrollTop;
  if (prepend) transcript.prepend(fragment);
  else transcript.replaceChildren(fragment);
  transcript.scrollTop = prepend ? previousTop + transcript.scrollHeight - previousHeight : transcript.scrollHeight;
  nextOlderTurn = page.next_before;
  field("older-turns").hidden = nextOlderTurn === null;
  updatePanels();
}

async function openSession(id, restoring = false) {
  if (busy || id === sessionId) return;
  setBusy(true);
  historyStatus("正在打开会话…");
  try {
    const page = await request("/api/sessions/" + encodeURIComponent(id) + "/turns");
    clearConversation();
    rememberSession(id);
    const lastMode = page.turns.at(-1)?.result.steps?.[0];
    if (modes[lastMode]) selectMode(lastMode, !restoring);
    showTurns(page);
    if (mode === "practice_plan" && selectedAssessment) {
      field("melody").value = selectedAssessment.melody;
      updatePanels();
    }
    historyStatus("");
    if (!restoring && matchMedia("(max-width: 730px)").matches) setHistoryExpanded(false);
  } catch (error) {
    if (restoring && error.status === 404) {
      rememberSession(undefined);
      historyStatus(sessions.length ? "" : "还没有历史对话。发送消息后会自动保存。");
    } else historyStatus(error.message, true);
  } finally {
    setBusy(false);
  }
}

function setHistoryExpanded(expanded) {
  document.querySelector(".history").dataset.expanded = String(expanded);
  const button = field("toggle-history");
  button.setAttribute("aria-expanded", expanded);
  button.textContent = expanded ? "收起" : "展开";
}

field("toggle-history").addEventListener("click", () => {
  setHistoryExpanded(field("toggle-history").getAttribute("aria-expanded") !== "true");
});
field("more-sessions").addEventListener("click", () => loadSessions({more: true}));
field("retry-history").addEventListener("click", () => loadSessions({restore: true}));
field("older-turns").addEventListener("click", async () => {
  if (busy || !sessionId || nextOlderTurn === null) return;
  setBusy(true);
  try {
    const page = await request("/api/sessions/" + encodeURIComponent(sessionId) + "/turns?before=" + nextOlderTurn);
    showTurns(page, true);
    if (!pendingJob) taskStatus("");
  } catch (error) {
    taskStatus(error.message, "error");
  } finally {
    setBusy(false);
  }
});

function startNewSession() {
  clearConversation();
  rememberSession(undefined);
  selectMode(mode);
}

field("new-session").addEventListener("click", () => {
  startNewSession();
  field("query").focus();
});

async function finishJob() {
  for (let attempt = 0; attempt < 100; attempt++) {
    const data = await request("/api/jobs/" + pendingJob.id);
    if (data.state === "failed") {
      pendingJob = undefined;
      throw new Error(data.error || "任务未完成，请检查输入后重试。");
    }
    if (data.state === "succeeded") {
      const result = data.result;
      if (data.kind === "assessment") {
        selectedAssessment = {id: data.id, melody: result.melody_id};
        field("use-assessment").checked = true;
        updatePanels();
      }
      const transcript = field("transcript");
      const resultMessage = jobMessage(data, true);
      transcript.append(resultMessage);
      transcript.scrollTop = resultMessage.offsetTop;
      pendingJob = undefined;
      taskStatus("");
      return;
    }
    taskStatus(data.state === "queued" ? "任务正在排队…" : "正在处理音频…");
    await new Promise(resolve => setTimeout(resolve, 800));
  }
  taskStatus("任务仍在处理，可继续查询。");
}

field("resume-job").addEventListener("click", async () => {
  if (busy || !pendingJob) return;
  setBusy(true);
  try { await finishJob(); }
  catch (error) {
    if (error.status === 404) pendingJob = undefined;
    taskStatus(error.message, "error");
  } finally { setBusy(false); }
});

field("chat").addEventListener("submit", async event => {
  event.preventDefault();
  if (busy || pendingJob) return;
  const query = field("query").value.trim();
  const selected = mode;
  if (selected === "qa" && !query) {
    taskStatus("请填写你的问题。", "error");
    field("query").focus();
    return;
  }
  if (selected === "assessment" && !recording) {
    taskStatus("请先选择一份 WAV 清唱录音。", "error");
    field("recording").focus();
    return;
  }
  const combined = selected === "assessment" && field("also-plan").checked;
  const planning = selected === "practice_plan" || combined;
  const payload = {query: query || modes[selected].title, intent: selected, session_id: sessionId};
  if (selected !== "qa") payload.melody_id = field("melody").value;
  if (selected === "assessment" || selected === "singing") Object.assign(payload, melodySpec());
  if (selected === "assessment") payload.offset_seconds = Number(field("offset").value);
  if (selected === "singing" && field("lyrics").value.trim()) payload.lyrics = field("lyrics").value.trim();
  if (selected === "practice_plan" && !field("use-assessment-group").hidden && field("use-assessment").checked) {
    payload.assessment_job_id = selectedAssessment.id;
  }
  if (planning) {
    for (const id of ["days", "minutes"]) if (field(id).value) payload[id] = Number(field(id).value);
    if (field("focus").value) payload.focus = field("focus").value;
    if (selected === "practice_plan" && query) payload.goal = query.slice(0, 240);
  }
  if (combined) {
    payload.intent = "auto";
    payload.query = "测评并安排练习" + (query ? "。 " + query.slice(0, 2380) : "");
  }
  if (selected === "qa" && field("auto-route").checked) payload.intent = "auto";
  setBusy(true);
  taskStatus("正在提交任务…");
  try {
    await initialModelSettings;
    if (activeConfig.provider !== "local" && selected === "qa") {
      payload.provider = activeConfig.provider;
      payload.model = activeConfig.model;
      payload.use_saved_model = true;
    }
    if (selected === "assessment") {
      if (!uploadedRecording) {
        taskStatus("正在上传录音…");
        const form = new FormData();
        form.append("file", recording);
        uploadedRecording = (await request("/api/audio", {form})).id;
      }
      payload.audio_id = uploadedRecording;
    }
    const result = await request("/api/chat", {payload});
    rememberSession(result.session_id);
    loadSessions();
    message(payload.query, true);
    message(result.answer, false, result);
    if (field("query").value.trim() === query) field("query").value = "";
    if (result.job) {
      pendingJob = result.job;
      await finishJob();
    } else taskStatus("");
  } catch (error) {
    if (error.status === 410) {
      startNewSession();
      field("query").value = query;
      loadSessions();
    }
    if (error.status === 404) pendingJob = undefined;
    taskStatus(error.message, "error");
  } finally {
    setBusy(false);
    selectMode(mode);
  }
});
field("query").addEventListener("keydown", event => {
  if (event.key === "Enter" && (event.ctrlKey || event.metaKey) && !event.isComposing && !busy && !pendingJob) {
    event.preventDefault();
    field("chat").requestSubmit();
  }
});
selectMode(mode);
const initialModelSettings = restoreModelSettings();
initialModelSettings.then(refreshHealth);
loadSessions({restore: true});
