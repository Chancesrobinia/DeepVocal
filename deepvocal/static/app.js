const field = id => document.getElementById(id);
const modes = {
  auto: {
    title: "声乐对话", description: "描述你的需求，DeepVocal 会自动分配任务。",
    label: "消息", placeholder: "例如：假声怎么练？也帮我测一下这段录音。", send: "发送",
    tipTitle: "可以一起问", tip: "添加清唱录音，把测评和声乐问题一起发给 DeepVocal。",
    examples: ["假声怎么练？", "给我安排换声区练习计划", "生成一段示范音频"],
  },
  qa: {
    title: "声乐问答", description: "提一个具体问题，查看导师的回答和引用资料。",
    label: "你的问题", placeholder: "例如：换声区容易破音，应该怎么练？", send: "发送问题",
    tipTitle: "试着这样问", tip: "从一个发声问题开始。回答中的引用可以展开查看原文。",
    examples: ["什么是换声区？", "长音不稳可以怎么练？", "唱歌总是拖拍，怎么找节奏？"],
  },
  assessment: {
    title: "录音测评", description: "上传你的演唱片段，手边有对比歌声也可以一并附上。",
    label: "测评要求", placeholder: "例如：我唱的是《晴天》，帮我看看副歌的音准。", send: "开始测评",
    tipTitle: "录音之前", tip: "录下一段旋律清晰的清唱，避免把伴奏或参考歌声录进去。",
    examples: [],
  },
  practice_plan: {
    title: "练习规划", description: "根据录音测评或你的目标，直接安排练习和复测。",
    label: "练习目标（可选）", placeholder: "例如：这周想练稳换声区。", send: "生成练习计划",
    tipTitle: "从一个目标开始", tip: "可以说说最想改善的地方。已有测评时，也可以直接生成计划。",
    examples: ["练稳换声区", "改善拖拍", "练习长音稳定性"],
  },
  singing: {
    title: "歌曲翻唱", description: "上传自己的录音，在消息里指定歌曲。生成后可以试听和下载。",
    label: "翻唱要求", placeholder: "例如：用我的音色翻唱《晴天》。", send: "生成翻唱",
    tipTitle: "选择录音", tip: "上传清晰的单人录音作为音色参考。曲库没有对应歌曲时，可以附上清晰的参考主唱。",
    examples: ["用我的音色翻唱《晴天》"],
  },
};
const providerLabels = {local: "本地 / 服务端", openai: "OpenAI", deepseek: "DeepSeek", dashscope: "阿里云百炼", custom: "自定义服务"};
const agentChips = document.querySelectorAll(".agent-chip");
let mode = "auto";
let audioToolsOpen = false;
let singingGuide = false;
let pendingCover = null;
let activeConfig = {provider: "local", model: "", key_saved: false, token: "", verified: false};
let sessionId;
let recording;
let uploadedRecording;
let recordingOffset = 0;
let uploadedReference;
let referencePreviewURL;
let selectedAssessmentId;
let explicitlySelectedAssessmentId;
let pendingJobs = [];
let busy = false;
let guideURL;
let guideRevision = 0;
const mediaURLs = new Set();
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

async function readChatStream(response, onEvent) {
  if (!response.headers.get("content-type")?.startsWith("text/event-stream") || !response.body) {
    throw new TypeError("Unexpected chat response format");
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  try {
    for (;;) {
      const {value, done} = await reader.read();
      buffer += decoder.decode(value, {stream: !done});
      let match;
      while ((match = /\r?\n\r?\n/.exec(buffer))) {
        const frame = buffer.slice(0, match.index);
        buffer = buffer.slice(match.index + match[0].length);
        const data = frame.split(/\r?\n/).filter(line => line.startsWith("data:")).map(line => line.slice(5)).join("\n");
        if (!data.trim()) continue;
        let event;
        try { event = JSON.parse(data); }
        catch (cause) { throw new TypeError("Invalid chat stream event", {cause}); }
        if (event.type === "done") {
          if (!event.result?.session_id || typeof event.result.answer !== "string") {
            throw new TypeError("Invalid completed chat result");
          }
          return event.result;
        }
        if (event.type === "error") {
          throw Object.assign(new Error(event.detail), {status: event.status});
        }
        onEvent(event);
      }
      if (done) throw new TypeError("Chat stream ended before completion");
    }
  } finally {
    await reader.cancel().catch(() => {});
    reader.releaseLock();
  }
}

async function request(url, {payload, form, config = activeConfig, binary, method, onEvent} = {}) {
  const headers = {};
  if (config.token) headers["X-DeepVocal-Token"] = config.token;
  if (config.provider !== "local" && config.key && payload) headers["X-Model-API-Key"] = config.key;
  if (payload) headers["Content-Type"] = "application/json";
  const verb = method ?? (payload || form ? "POST" : "GET");
  const options = {
    method: verb, headers,
    body: payload ? JSON.stringify(payload) : form, cache: "no-store", redirect: "error",
  };
  let response;
  let body;
  for (let attempt = 0; ; attempt++) {
    try {
      response = await fetch(url, options);
      if (response.status === 204) return;
      if (response.ok && binary) return await response.blob();
      if (response.ok && onEvent) return await readChatStream(response, onEvent);
      body = await response.text();
      break;
    } catch (cause) {
      if (!(cause instanceof TypeError)) throw cause;
      if (verb === "GET" && attempt < 2) {
        await new Promise(resolve => setTimeout(resolve, 250 * (attempt + 1)));
        continue;
      }
      const message = verb === "GET" || url === "/api/model/test"
        ? "与 DeepVocal 的连接中断，请确认本地服务正在运行后重试。"
        : url === "/api/chat" || url === "/api/chat/stream"
          ? "连接中断，无法确认任务是否已提交。请先查看历史对话，再决定是否重发。"
          : "与 DeepVocal 的连接中断，无法确认操作结果。请检查后重试。";
      const error = new Error(message, {cause});
      error.requestPath = url;
      throw error;
    }
  }
  let data;
  try { data = JSON.parse(body); }
  catch {
    if (response.ok) throw new Error("DeepVocal 返回了无法读取的数据，请刷新后重试。");
  }
  if (response.ok) return data;
  const detail = data?.detail;
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
  field("resume-job").hidden = busy || !pendingJobs.length;
}

function setBusy(value) {
  busy = value;
  field("task-fields").disabled = value;
  field("attach-audio").disabled = value;
  field("send").disabled = value || pendingJobs.length > 0;
  field("new-session").disabled = value;
  field("open-settings").disabled = value;
  field("hero-chat").disabled = value;
  field("older-turns").disabled = value;
  document.querySelectorAll(".suggestion, .history-item, .history-delete").forEach(button => { button.disabled = value; });
  field("resume-job").hidden = value || !pendingJobs.length;
  field("chat").setAttribute("aria-busy", value);
}

function updatePanels() {
  const isAuto = mode === "auto";
  const isQA = mode === "qa";
  const isAssessment = mode === "assessment";
  const isPractice = mode === "practice_plan";
  const isSinging = mode === "singing";
  const showGuide = isSinging && singingGuide;
  const showCover = (isSinging && !singingGuide) || (isAuto && pendingCover);
  const hasReferenceFile = Boolean(uploadedReference || field("reference-file").files[0]);
  const hasAudioDraft = Boolean(recording || uploadedRecording || hasReferenceFile);
  const showRecording = isAssessment || Boolean(showCover) || (isAuto && (audioToolsOpen || hasAudioDraft));
  const hasReference = showRecording && hasReferenceFile;
  field("reference-fields").hidden = !showGuide;
  field("singing-mode-toggle").hidden = !isSinging;
  field("singing-mode-toggle").textContent = showGuide ? "返回歌曲翻唱" : "生成练习参考音";
  field("singing-mode-toggle").setAttribute("aria-expanded", showGuide);
  field("recording-fields").hidden = !showRecording;
  field("attach-audio").hidden = !isAuto || hasAudioDraft || Boolean(pendingCover);
  field("attach-audio").setAttribute("aria-expanded", showRecording);
  field("attach-audio-label").textContent = showRecording ? "收起录音设置" : "添加录音";
  field("reference-options").hidden = !hasReference;
  field("reference-file-label").textContent = showCover ? "参考主唱（可选）" : "对比歌声（可选）";
  field("recording-purpose").textContent = showCover ? "翻唱会在本机处理。本人录音只用于提取音色。"
    : isAuto ? "附上录音后，在消息里说明你想测评还是翻唱。"
      : "歌曲测评会将演唱录音与参考音频交给已配置的百炼音频模型。";
  field("preview-saved-reference").hidden = !hasReference || !uploadedReference || !!referencePreviewURL;
  field("plan-context").hidden = !isPractice || !selectedAssessmentId;
  field("plan-context").textContent = selectedAssessmentId && selectedAssessmentId === explicitlySelectedAssessmentId
    ? "将使用所选录音测评" : "本会话有可用的录音测评";
  field("lyrics-group").hidden = !showGuide;
  field("query").placeholder = showGuide ? "例如：为今天的练习生成参考音。" : modes[mode].placeholder;
  field("send-label").textContent = showGuide ? "生成参考音" : modes[mode].send;
  for (const id of ["tempo-group", "guide-group", "transpose-group"]) field(id).hidden = !showGuide;
  for (const id of ["bpm", "transpose"]) field(id).disabled = !showGuide;
  field("recording").disabled = !showRecording;
  field("reference-file").disabled = !showRecording;
  field("query").required = isQA;
}

field("singing-mode-toggle").addEventListener("click", () => {
  singingGuide = !singingGuide;
  updatePanels();
});

function selectMode(next, focus = false) {
  mode = next;
  const config = modes[mode];
  field("task-title").textContent = config.title;
  field("task-description").textContent = config.description;
  field("query-label").textContent = config.label;
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
  for (const button of agentChips) button.setAttribute("aria-pressed", button.dataset.agent === mode);
  updatePanels();
  if (focus) field("task-title").focus();
}

for (const button of agentChips) {
  button.addEventListener("click", () => selectMode(button.dataset.agent === mode ? "auto" : button.dataset.agent));
}
field("attach-audio").addEventListener("click", () => {
  audioToolsOpen = !audioToolsOpen;
  updatePanels();
  if (audioToolsOpen) field("recording").focus();
});
field("hero-chat").addEventListener("click", () => field("query").focus());

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

const audioExtension = /\.(wav|m4a)$/i;
const recordingFileHint = "WAV（16 位 PCM）或 M4A · 最长 60 秒 · 最大 12 MiB";

function setRecording(file) {
  const format = file?.name.match(audioExtension)?.[1].toUpperCase();
  if (file && (!format || file.size === 0 || file.size > 12 * 1024 * 1024)) {
    field("recording").value = "";
    setRecording();
    taskStatus("请选择 12 MiB 以内的 WAV 或 M4A 录音。", "error");
    return;
  }
  recording = file;
  uploadedRecording = undefined;
  recordingOffset = 0;
  field("file-name").textContent = file?.name || "选择或拖入清唱录音";
  field("file-detail").textContent = file ? (file.size / 1024).toFixed(1) + " KiB · " + format : recordingFileHint;
  field("remove-recording").hidden = !file;
  updatePanels();
  if (!pendingJobs.length) taskStatus("");
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
    mediaURLs.delete(guideURL);
    guideURL = undefined;
  }
}
function melodySpec() {
  return {melody_id: field("melody").value, bpm: Number(field("bpm").value), transpose: Number(field("transpose").value)};
}

const referenceFileHint = "WAV（16 位 PCM）或 M4A · 最长 8 分钟。建议使用清晰的单人主唱或独唱参考。";
function setReferencePreview(file) {
  const player = field("reference-preview");
  if (referencePreviewURL) URL.revokeObjectURL(referencePreviewURL);
  referencePreviewURL = file ? URL.createObjectURL(file) : undefined;
  player.hidden = !referencePreviewURL;
  if (referencePreviewURL) player.src = referencePreviewURL;
  else {
    player.removeAttribute("src");
    player.load();
  }
}
function setReferenceAttachment(file) {
  uploadedReference = undefined;
  field("reference-file-info").textContent = file?.name || referenceFileHint;
  setReferencePreview(file);
  updatePanels();
}

function clearReference() {
  field("reference-file").value = "";
  setReferenceAttachment();
}

field("reference-file").addEventListener("change", event => {
  const file = event.target.files[0];
  if (file && (!audioExtension.test(file.name) || !file.size || file.size > 96 * 1024 * 1024)) {
    clearReference();
    taskStatus("请选择 96 MiB 以内的 WAV 或 M4A 对比歌声。", "error");
    return;
  }
  setReferenceAttachment(file);
  taskStatus("");
});
field("remove-reference").addEventListener("click", () => {
  clearReference();
  field("reference-file").focus();
});
field("preview-saved-reference").addEventListener("click", async event => {
  const button = event.currentTarget;
  const reference = uploadedReference;
  if (!reference || busy) return;
  button.disabled = true;
  try {
    const blob = await request("/api/audio/" + encodeURIComponent(reference.id), {binary: true});
    if (uploadedReference !== reference || referencePreviewURL) return;
    setReferencePreview(blob);
    updatePanels();
  } catch (error) {
    taskStatus(error.message, "error");
  } finally {
    button.disabled = false;
  }
});

function useAssessmentReference(reference, {range, recordingId, offset, alignmentMode} = {}) {
  if (busy || pendingJobs.length) return;
  if (uploadedReference?.id !== reference.audio_id) setReferencePreview();
  field("reference-file").value = "";
  uploadedReference = {id: reference.audio_id, range, transpose: reference.transpose, alignmentMode};
  field("reference-file-info").textContent = range
    ? reference.title + " · 已选片段 " + seconds(range[0]) + "–" + seconds(range[1])
    : reference.title;
  useRecordingForRetry(recordingId, offset);
  selectMode("assessment", true);
  if (alignmentMode === "dtw") taskStatus("已选择备用对齐，点击开始测评查找候选片段。");
  else taskStatus(recordingId ? "已选参考片段，点击开始测评。" : "已选参考片段，请上传新的演唱录音。");
}

function useRecordingForRetry(retryAudioId, offset = 0) {
  if (!retryAudioId || retryAudioId !== uploadedRecording) {
    field("recording").value = "";
    setRecording();
    if (retryAudioId) {
      uploadedRecording = retryAudioId;
      field("file-name").textContent = "复用该次演唱录音";
      field("remove-recording").hidden = false;
    }
  }
  recordingOffset = retryAudioId ? offset : 0;
}

function uploadComparisonForRetry(report) {
  if (busy || pendingJobs.length) return;
  useRecordingForRetry(report.recording_audio_id, report.offset_seconds);
  clearReference();
  const input = field("reference-file");
  selectMode("assessment", true);
  input.focus();
  input.click();
}

function comparisonUploadButton(report, label = "上传对比歌声") {
  const button = make("button", "button quiet", label);
  button.type = "button";
  button.addEventListener("click", () => uploadComparisonForRetry(report));
  return button;
}

function addAudio(parent, blob, filename) {
  const url = URL.createObjectURL(blob);
  mediaURLs.add(url);
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

function renderEvidenceFigure(details, figure) {
  if (!/^\/api\/knowledge\/figures\/[a-z0-9_-]+\/[a-z0-9_-]+$/.test(figure.url || "")) return;
  const panel = make("figure", "source-figure");
  const caption = make("figcaption", "", figure.caption);
  const status = make("p", "source-meta");
  status.setAttribute("role", "status");
  panel.append(caption, status);
  if (figure.credit) panel.append(make("p", "source-meta", figure.credit));
  details.append(panel);
  let loading = false;
  async function load() {
    if (!details.open || loading) return;
    loading = true;
    status.textContent = "正在加载图版…";
    try {
      const blob = await request(figure.url, {binary: true});
      if (!panel.isConnected) return;
      const url = URL.createObjectURL(blob);
      mediaURLs.add(url);
      const image = make("img");
      image.src = url;
      image.alt = figure.description;
      const viewport = make("div", "source-figure-viewport");
      viewport.append(image);
      const original = make("button", "source-link", "原图尺寸");
      original.type = "button";
      original.setAttribute("aria-pressed", "false");
      original.addEventListener("click", () => {
        const expanded = viewport.classList.toggle("full-size");
        original.setAttribute("aria-pressed", String(expanded));
        viewport.tabIndex = expanded ? 0 : -1;
      });
      status.replaceWith(original, viewport);
      details.removeEventListener("toggle", load);
    } catch {
      const retry = make("button", "source-link", "重试");
      retry.type = "button";
      retry.addEventListener("click", load);
      status.replaceChildren("图版加载失败。", retry);
      loading = false;
    }
  }
  details.addEventListener("toggle", load);
}

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
    for (const figure of source.figures || []) renderEvidenceFigure(details, figure);
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

function renderRelatedFigures(parent, sources) {
  const panel = make("section", "result sources");
  panel.append(make("h3", "", "相关图版"));
  for (const source of sources) {
    const details = make("details");
    details.append(make("summary", "", source.title), make("p", "source-meta", source.source));
    for (const figure of source.figures) renderEvidenceFigure(details, figure);
    panel.append(details);
  }
  parent.append(panel);
}

const signed = (value, unit) => Number.isFinite(value) ? (value > 0 ? "+" : "") + value + " " + unit : "—";
const seconds = value => Number.isFinite(value) ? value.toFixed(2) + " s" : "—";

function segmentPlayback(parent, label, audioId, start, end) {
  if (!audioId || !Number.isFinite(start) || !Number.isFinite(end)) return;
  const button = make("button", "button quiet", label);
  const status = make("span", "hint");
  const output = make("div");
  let player;
  button.type = "button";
  button.addEventListener("click", async () => {
    button.disabled = true;
    try {
      if (!player) {
        const blob = await request("/api/audio/" + encodeURIComponent(audioId), {binary: true});
        if (!output.isConnected) return;
        player = addAudio(output, blob).player;
        player.setAttribute("aria-label", label);
        player.addEventListener("timeupdate", () => {
          if (player.currentTime >= end) player.pause();
        });
      }
      player.currentTime = start;
      await player.play();
      status.textContent = "";
    } catch (error) {
      status.textContent = error.message;
    } finally {
      button.disabled = false;
    }
  });
  parent.append(button, status, output);
}

function renderReferenceMatch(panel, report) {
  const reference = report.reference;
  if (reference) panel.append(make("p", "source-meta", reference.title));
  if (reference?.selection === "library") {
    const shift = reference.detected_transpose;
    panel.append(make("p", "hint", "已从曲库自动匹配。估计整体调高差为 " + signed(shift, "半音") + "，音准按参考原调计算。"));
  }
  if (report.candidates?.length) {
    panel.append(make("p", "", "试听候选片段后，选择正确的一段再测评。"));
    for (const candidate of report.candidates) {
      const entry = make("div", "inline-controls");
      const {start_seconds: start, end_seconds: end} = candidate;
      const selected = candidate.audio_id ? candidate : reference;
      const title = !reference && candidate.title ? candidate.title + " " : "";
      const select = make("button", "button quiet", "选用 " + title + seconds(start) + "–" + seconds(end));
      select.type = "button";
      select.addEventListener("click", () => useAssessmentReference(selected, {
        range: [start, end], recordingId: report.recording_audio_id, offset: report.offset_seconds,
      }));
      entry.append(select);
      segmentPlayback(entry, "试听", selected.audio_id, start, end);
      panel.append(entry);
    }
  } else if (reference?.clip_audio_id) {
    panel.append(make("p", "", "原曲片段 " + seconds(reference.start_seconds) + "–" + seconds(reference.end_seconds)));
    const play = make("div", "inline-controls");
    segmentPlayback(play, "试听对齐片段", reference.clip_audio_id, 0, reference.end_seconds - reference.start_seconds);
    segmentPlayback(play, "试听本次演唱", report.recording_audio_id, report.offset_seconds, report.recording_end_seconds);
    const select = make("button", "button quiet", "用同一参考片段复测");
    select.type = "button";
    select.addEventListener("click", () => useAssessmentReference(reference, {
      range: [reference.start_seconds, reference.end_seconds],
    }));
    play.append(select);
    panel.append(play);
  } else if (report.alignment) {
    const {recording_start_seconds: start, recording_end_seconds: end, relative_speed: speed} = report.alignment;
    panel.append(make("p", "", "参考 " + seconds(reference.start_seconds) + "–" + seconds(reference.end_seconds)
      + " · 演唱 " + seconds(start) + "–" + seconds(end)
      + " · 估计整体速度 " + Math.round(speed * 100) + "%"));
    const play = make("div", "inline-controls");
    segmentPlayback(play, "试听参考片段", reference.audio_id, reference.start_seconds, reference.end_seconds);
    segmentPlayback(play, "试听本次演唱", report.recording_audio_id, start, end);
    panel.append(play);
    const select = make("button", "button quiet", "用同一参考片段复测");
    select.type = "button";
    select.addEventListener("click", () => useAssessmentReference(reference, {
      range: [reference.segment_start_seconds, reference.segment_end_seconds],
    }));
    panel.append(select);
  } else if (reference && ["alignment_uncertain", "reference_unusable"].includes(report.issue)) {
    panel.append(comparisonUploadButton(report, "更换对比歌声"));
  }
  if (reference && report.issue === "alignment_uncertain" && report.alignment_method === "llm") {
    const backup = make("button", "button quiet", "尝试备用对齐");
    backup.type = "button";
    backup.addEventListener("click", () => useAssessmentReference(reference, {
      recordingId: report.recording_audio_id, offset: report.offset_seconds, alignmentMode: "dtw",
    }));
    panel.append(backup);
  }
}

function renderAssessment(parent, report, jobId) {
  const panel = make("section", "result");
  if (report.issue === "song_title_required") {
    const retry = make("button", "button quiet", "补充歌名");
    retry.type = "button";
    retry.addEventListener("click", () => {
      if (busy || pendingJobs.length) return;
      useRecordingForRetry(report.recording_audio_id, report.offset_seconds);
      clearReference();
      selectMode("assessment", true);
    });
    panel.append(retry);
    parent.append(panel);
    return;
  }
  if (report.issue === "reference_not_found" || (report.issue === "alignment_uncertain" && !report.reference && !report.candidates?.length)) {
    panel.append(comparisonUploadButton(report));
    parent.append(panel);
    return;
  }
  if (["reference_mismatch", "reference_segment_mismatch"].includes(report.issue)) {
    panel.append(comparisonUploadButton(report, "重新上传对比歌声"));
    parent.append(panel);
    return;
  }
  panel.append(make("h3", "", report.reference ? "片段对齐与音准" : "音准与节奏"));
  if (report.reference || report.candidates?.length) renderReferenceMatch(panel, report);
  if (report.audio_feedback?.assessable) {
    for (const [name, key] of [["音准", "intonation"], ["节奏", "rhythm"], ["稳定性", "stability"], ["练习", "exercise"]]) {
      if (report.audio_feedback[key]) panel.append(make("h4", "", name), make("p", "", report.audio_feedback[key]));
    }
  } else if (report.audio_feedback || report.issue === "alignment_uncertain" || (report.reference && !report.notes?.length)) {
    parent.append(panel);
    return;
  } else {
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
    if (report.comparison) {
      const {previous, current, absolute_bias_reduction_cents: change} = report.comparison;
      const direction = change > 0 ? "偏差幅度减小" : change < 0 ? "偏差幅度增大" : "偏差幅度持平";
      const difference = change === 0 ? "" : " " + Math.abs(change) + " 音分";
      const date = new Date(previous.observed * 1000).toLocaleString();
      panel.append(make("p", "source-meta", (report.reference ? "同片段复测" : "同调同速复测") + "（上次 " + date + "） · " +
        signed(previous.median_pitch_error_cents, "音分") + " → " +
        signed(current.median_pitch_error_cents, "音分") + "，" + direction + difference + "。"));
    }
    if (report.notes?.length) {
      const details = make("details", "note-detail");
      details.append(make("summary", "", "查看 " + report.notes.length + " 个音符的偏差"));
      const wrapper = make("div");
      const table = make("table");
      const heading = table.createTHead().insertRow();
      const columns = report.reference ? ["音符", "参考起点", "演唱对应位置", "音高偏差", "音高波动", "对齐后起音偏差"]
        : ["音符", "参考起点", "音高偏差", "起音偏差"];
      for (const label of columns) {
        const cell = make("th", "", label);
        cell.scope = "col";
        heading.append(cell);
      }
      const tbody = table.createTBody();
      for (const note of report.notes) {
        const row = tbody.insertRow();
        const name = ["C", "C♯", "D", "D♯", "E", "F", "F♯", "G", "G♯", "A", "A♯", "B"][note.target_midi % 12] + (Math.floor(note.target_midi / 12) - 1);
        row.append(make("td", "", note.note + " · " + name));
        if (report.reference) row.append(make("td", "", seconds(note.reference_start_seconds)));
        row.append(make("td", "", seconds(note.start_seconds)),
          make("td", Math.abs(note.pitch_error_cents) > 50 ? "outside" : "", signed(note.pitch_error_cents, "音分")));
        if (report.reference) row.append(make("td", "", note.pitch_iqr_cents + " 音分"));
        row.append(make("td", "", signed(note.onset_error_ms, "ms")));
      }
      wrapper.append(table);
      details.append(wrapper);
      panel.append(details);
    }
  }
  const button = make("button", "button quiet", "用这份测评规划练习");
  button.type = "button";
  button.addEventListener("click", () => {
    if (busy) return;
    selectedAssessmentId = jobId;
    explicitlySelectedAssessmentId = jobId;
    selectMode("practice_plan", true);
  });
  if (jobId && !report.plan && report.issue !== "insufficient_audio") panel.append(button);
  parent.append(panel);
}

function renderPlan(parent, plan) {
  const panel = make("section", "result plan");
  panel.append(make("h3", "", plan.days + " 天练习计划 · 每天 " + plan.minutes_per_day + " 分钟"),
    make("p", "plan-intro", "练习重点：" + plan.focus_label + (plan.goal ? " · 目标：" + plan.goal : "")));
  if (plan.rationale) {
    panel.append(make("h4", "", "为什么这样安排"), make("p", "", plan.rationale));
  } else if (plan.generation?.mode === "rules") {
    const notices = {
      model_not_configured: "当前未配置规划模型，已按练习规则生成。",
      model_error: "规划模型暂不可用，已按练习规则生成。",
      tool_limit: "规划工具调用未完成，已按练习规则生成。",
      invalid_model_plan: "规划模型的结果未通过校验，已按练习规则生成。",
    };
    const checks = {
      schema: "模型返回的计划格式不完整",
      schedule: "模型给出的天数或时长超出本次练习范围",
      exercise_selection: "模型给出的练习项目与本次重点不符",
      daily_duration: "模型给出的各项时长与当天总时长不符",
      exercise_progression: "模型给出的练习动作重复或顺序不合理",
      unsupported_cause: "模型给出的练习说明包含测评证据无法支持的原因",
      primary_progression: "模型给出的主项安排缺少从分解到原句的过渡",
      evidence_coverage: "模型给出的练习安排遗漏了本次重点",
      assessment_ref: "模型给出的计划没有关联本次测评",
      focus: "模型给出的练习重点与本次要求不符",
    };
    const problem = checks[plan.generation.validation?.check];
    panel.append(make("p", "hint", problem ? problem + "，已按练习规则生成。" : notices[plan.generation.reason]));
  }
  if (plan.evidence?.length) {
    const list = make("ul", "exercises");
    for (const item of plan.evidence) {
      const source = {assessment: "测评", goal: "练习目标", baseline: "建立基准"}[item.source];
      list.append(make("li", "", source + " · " + item.observation));
    }
    panel.append(make("h4", "", "练习依据"), list);
  }
  for (const day of plan.sessions) {
    const details = make("details");
    details.open = day.day === 1;
    const list = make("ol", "exercises");
    for (const exercise of day.exercises) {
      const item = make("li", "", exercise.name);
      item.append(make("span", "minutes", exercise.minutes + " 分钟"), make("small", "", exercise.instructions));
      list.append(item);
    }
    details.append(make("summary", "", "第 " + day.day + " 天" + (day.stage ? " · " + day.stage : "")));
    if (day.objective) details.append(make("p", "source-meta", day.objective));
    details.append(list);
    panel.append(details);
  }
  if (plan.retest) {
    const {reference, conditions, criteria, decision} = plan.retest;
    const target = [reference.title];
    if (reference.start_seconds != null) target.push(seconds(reference.start_seconds) + "–" + seconds(reference.end_seconds));
    if (reference.bpm != null) target.push(reference.bpm + " BPM");
    if (reference.transpose != null) target.push("移调 " + reference.transpose + " 半音");
    const list = make("ul", "exercises");
    for (const {standard} of criteria) list.append(make("li", "", standard));
    panel.append(make("h4", "", "复测标准"), make("p", "source-meta", target.join(" · ")),
      make("p", "", conditions), list, make("p", "", decision));
    if (reference.audio_id && reference.start_seconds != null) {
      const retry = make("button", "button quiet", "用同一片段复测");
      retry.type = "button";
      retry.addEventListener("click", () => useAssessmentReference(reference, {
        range: [reference.start_seconds, reference.end_seconds], alignmentMode: reference.alignment_mode,
      }));
      panel.append(retry);
    }
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
  if (result.related_figures?.length) renderRelatedFigures(entry, result.related_figures);
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
  selectedAssessmentId = undefined;
  explicitlySelectedAssessmentId = undefined;
  pendingJobs = [];
  nextOlderTurn = null;
  audioToolsOpen = false;
  singingGuide = false;
  pendingCover = null;
  field("older-turns").hidden = true;
  field("chat").reset();
  setRecording();
  clearReference();
  clearGuide();
  for (const url of mediaURLs) URL.revokeObjectURL(url);
  mediaURLs.clear();
  field("transcript").replaceChildren();
  selectMode("auto");
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
  const kind = job.result.type === "cover"
    ? (job.result.song_title ? `《${job.result.song_title}》的翻唱` : "翻唱")
    : job.result.type === "singing" ? "歌声" : "参考音";
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
      const summary = report.plan_error ? report.summary + "\n\n" + report.plan_error : report.summary;
      return createMessage(summary, false, {assessment: report, assessmentJobId: job.id, plan: report.plan});
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
        if (!selectedAssessmentId && usableAssessmentJob(current)) {
          selectedAssessmentId = current.id;
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

function jobsFor(result) {
  return result.jobs || (result.job ? [result.job] : []);
}

function usableAssessmentJob(job) {
  return job.kind === "assessment" && job.state === "succeeded"
    && job.result && !["insufficient_audio", "alignment_uncertain", "reference_unusable", "reference_not_found",
      "song_title_required", "reference_mismatch", "reference_segment_mismatch"].includes(job.result.issue);
}

function showTurns(page, prepend = false) {
  const fragment = document.createDocumentFragment();
  for (const turn of page.turns) {
    fragment.append(createMessage(turn.query, true), createMessage(turn.result.answer, false, turn.result));
    for (const job of jobsFor(turn)) fragment.append(jobMessage(job));
  }
  if (!prepend) {
    const cover = page.turns.at(-1)?.result.singing_agent;
    pendingCover = cover?.status === "needs_input" ? cover : null;
  }
  if (!selectedAssessmentId) {
    const assessment = page.turns.slice().reverse().flatMap(jobsFor).find(usableAssessmentJob);
    if (assessment) selectedAssessmentId = assessment.id;
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
    showTurns(page);
    historyStatus("");
    if (!restoring) field("task-title").focus();
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
    if (!pendingJobs.length) taskStatus("");
  } catch (error) {
    taskStatus(error.message, "error");
  } finally {
    setBusy(false);
  }
});

function startNewSession() {
  clearConversation();
  rememberSession(undefined);
}

field("new-session").addEventListener("click", () => {
  startNewSession();
  field("query").focus();
});

async function finishJobs() {
  for (let attempt = 0; attempt < 100; attempt++) {
    const checks = await Promise.allSettled(pendingJobs.map(job => request("/api/jobs/" + job.id)));
    const remaining = [];
    let retryError;
    const transcript = field("transcript");
    for (const [index, check] of checks.entries()) {
      if (check.status === "rejected") {
        if (check.reason.status === 404) message(check.reason.message);
        else {
          remaining.push(pendingJobs[index]);
          retryError ||= check.reason;
        }
        continue;
      }
      const job = check.value;
      if (job.state === "queued" || job.state === "running") {
        remaining.push(job);
        continue;
      }
      if (usableAssessmentJob(job)) {
        selectedAssessmentId = job.id;
        if (mode === "auto" && uploadedRecording && job.result.recording_audio_id === uploadedRecording) {
          audioToolsOpen = false;
          field("recording").value = "";
          setRecording();
        }
        updatePanels();
      }
      const resultMessage = jobMessage(job, true);
      transcript.append(resultMessage);
      transcript.scrollTop = resultMessage.offsetTop;
    }
    pendingJobs = remaining;
    if (retryError) throw retryError;
    if (!pendingJobs.length) {
      taskStatus("");
      return;
    }
    const cover = pendingJobs.find(job => job.kind === "voice_conversion" && job.result?.segments_total);
    taskStatus(cover ? `正在生成翻唱，已完成 ${cover.result.segments_done}/${cover.result.segments_total} 段…`
      : pendingJobs.every(job => job.state === "queued") ? "任务正在排队…" : "正在处理音频…");
    await new Promise(resolve => setTimeout(resolve, 800));
  }
  taskStatus("任务仍在处理，可继续查询。");
}

field("resume-job").addEventListener("click", async () => {
  if (busy || !pendingJobs.length) return;
  setBusy(true);
  try { await finishJobs(); }
  catch (error) {
    taskStatus(error.message, "error");
  } finally { setBusy(false); }
});

field("chat").addEventListener("submit", async event => {
  event.preventDefault();
  if (busy || pendingJobs.length) return;
  const query = field("query").value.trim();
  const selected = mode;
  const auto = selected === "auto";
  const cover = selected === "singing" && !singingGuide;
  const canAttach = auto || selected === "assessment" || cover;
  const hasRecording = Boolean(recording || uploadedRecording);
  const submitRecording = canAttach && hasRecording;
  const referenceFile = field("reference-file").files[0];
  const hasReference = canAttach && Boolean(uploadedReference || referenceFile);
  if ((selected === "assessment" || hasReference) && !hasRecording && !((cover || auto) && pendingCover?.recording_audio_id)) {
    taskStatus("请先选择一份 WAV 或 M4A 清唱录音。", "error");
    field("recording").focus();
    return;
  }
  if ((selected === "qa" || (auto && !submitRecording && !hasReference)) && !query) {
    taskStatus(auto ? "请输入消息或添加录音。" : "请填写你的问题。", "error");
    field("query").focus();
    return;
  }
  const planning = selected === "practice_plan";
  const streamReply = auto || selected === "qa";
  const payload = {
    query: query || (cover || (auto && pendingCover) ? "歌曲翻唱" : auto ? "请测评这段录音。" : modes[selected].title),
    intent: selected, session_id: sessionId,
  };
  if (selected === "singing") payload.singing_mode = cover ? "cover" : "guide";
  if (auto && pendingCover && !query) payload.singing_mode = "cover";
  if (selected === "singing" && singingGuide) Object.assign(payload, melodySpec());
  if (submitRecording && recordingOffset) payload.offset_seconds = recordingOffset;
  if (selected === "singing" && singingGuide && field("lyrics").value.trim()) payload.lyrics = field("lyrics").value.trim();
  if (planning) {
    if (selectedAssessmentId) {
      payload.assessment_job_id = selectedAssessmentId;
      payload.assessment_source = selectedAssessmentId === explicitlySelectedAssessmentId ? "explicit" : "session";
    }
    if (query) payload.goal = query.slice(0, 240);
  }
  setBusy(true);
  taskStatus("正在提交任务…");
  let userEntry;
  let preview;
  let draft = "";
  let notice = "";
  try {
    await initialModelSettings;
    if (activeConfig.provider !== "local" && (streamReply || submitRecording || planning || cover)) {
      payload.provider = activeConfig.provider;
      payload.model = activeConfig.model;
      payload.use_saved_model = true;
    }
    if (hasReference) {
      if (!uploadedReference) {
        taskStatus("正在上传参考歌曲…");
        const form = new FormData();
        form.append("file", referenceFile);
        uploadedReference = await request("/api/audio/reference", {form});
      }
      payload.reference_audio_id = uploadedReference.id;
      if (uploadedReference.alignmentMode) payload.alignment_mode = uploadedReference.alignmentMode;
      if (uploadedReference.range) {
        [payload.reference_start_seconds, payload.reference_end_seconds] = uploadedReference.range;
        if (uploadedReference.transpose) payload.transpose = uploadedReference.transpose;
      }
    }
    if (submitRecording) {
      if (!uploadedRecording) {
        taskStatus("正在上传录音…");
        const form = new FormData();
        form.append("file", recording);
        uploadedRecording = (await request("/api/audio", {form})).id;
      }
      payload.audio_id = uploadedRecording;
    }
    if (auto) taskStatus("正在分析任务…");
    const result = streamReply ? await request("/api/chat/stream", {payload, onEvent(event) {
      if (!preview) {
        userEntry = message(payload.query, true);
        preview = message("等待回答…");
        taskStatus("");
      }
      const transcript = field("transcript");
      const follow = transcript.scrollHeight - transcript.scrollTop - transcript.clientHeight < 48;
      const content = preview.querySelector("p");
      if (event.type === "status" && event.phase === "retrieving") {
        draft = "";
        content.textContent = "正在检索知识库…";
      } else if (event.type === "status" && event.phase === "answering") {
        draft = "";
        content.textContent = "正在生成回答…";
      } else if (event.type === "status" && event.phase === "fallback") {
        draft = "";
        notice = event.notice + "\n\n";
        content.textContent = notice + "正在生成回答…";
      } else if (event.type === "delta" && typeof event.text === "string") {
        draft += event.text;
        if (!"NO_EVIDENCE".startsWith(draft.trim())) {
          content.textContent = notice + draft.replace(/\[(?:资料\s*)?\s*[a-z0-9_-]+(?:\s*\]|$)/gi, "");
        }
      }
      if (follow) transcript.scrollTop = transcript.scrollHeight;
    }}) : await request("/api/chat", {payload});
    rememberSession(result.session_id);
    loadSessions();
    if (preview) {
      const transcript = field("transcript");
      const follow = transcript.scrollHeight - transcript.scrollTop - transcript.clientHeight < 48;
      preview.replaceWith(createMessage(result.answer, false, result));
      if (follow) transcript.scrollTop = transcript.scrollHeight;
      preview = userEntry = undefined;
    } else {
      message(payload.query, true);
      message(result.answer, false, result);
    }
    if (field("query").value.trim() === query) field("query").value = "";
    pendingJobs = jobsFor(result);
    pendingCover = result.singing_agent?.status === "needs_input" ? result.singing_agent : null;
    if (hasReference && pendingJobs.some(job => ["assessment", "voice_conversion"].includes(job.kind))) clearReference();
    if (pendingJobs.length) await finishJobs();
    else taskStatus("");
  } catch (error) {
    preview?.remove();
    userEntry?.remove();
    if (error.requestPath === "/api/chat" || error.requestPath === "/api/chat/stream") loadSessions();
    if (error.status === 410) {
      startNewSession();
      field("query").value = query;
      loadSessions();
    }
    taskStatus(error.message, "error");
  } finally {
    setBusy(false);
    selectMode(mode);
  }
});
field("query").addEventListener("keydown", event => {
  if (event.key === "Enter" && (event.ctrlKey || event.metaKey) && !event.isComposing && !busy && !pendingJobs.length) {
    event.preventDefault();
    field("chat").requestSubmit();
  }
});
selectMode(mode);
const initialModelSettings = restoreModelSettings();
initialModelSettings.then(refreshHealth);
loadSessions({restore: true});
