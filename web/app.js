/* Kicklab: browser-only presentation of the local, measured analysis reports. */
"use strict";

const $ = (id) => document.getElementById(id);
const state = {
  selectedFile: null,
  jobId: null,
  report: null,
  currentFrame: 0,
  metric: "knee_flexion",
  pollingTimer: null,
  selectionVersion: 0,
  playbackCallback: null,
  animationFrame: null,
  graph: null,
  overlay: true,
  restoreFrame: 0,
  jobs: [],
  pollFailures: 0,
  sessionRefreshVersion: 0,
  pendingSeekFrame: null,
  frameLoopVersion: 0,
};
const video = $("analysis-video");
const canvas = $("pose-canvas");
const canvasContext = canvas.getContext("2d");
const PHASE_COLORS = {
  approach: "#c7d7b5",
  plant: "#b2c997",
  plant_foot: "#b2c997",
  plant_foot_placement: "#b2c997",
  backswing: "#96b47e",
  forward_swing: "#5f8b61",
  contact: "#dab977",
  ball_contact: "#dab977",
  follow_through: "#91b4a0",
  unknown: "#e3e9dc",
};
const METRIC_SHORT_NAMES = {
  knee_flexion: "Kicking knee",
  hip_angle: "Hip angle",
  plant_knee_flexion: "Plant knee",
  torso_lean: "Torso lean",
  foot_speed: "Foot speed",
  plant_displacement: "Plant movement",
};
const SHOT_NAMES = { shot: "General shot", penalty: "Penalty", free_kick: "Free kick" };
const VIEW_NAMES = { side: "Side view", oblique: "Angled view", front: "Front / rear view" };
const CONNECTIONS = [
  ["left_shoulder", "right_shoulder"], ["left_shoulder", "left_elbow"],
  ["left_elbow", "left_wrist"], ["right_shoulder", "right_elbow"],
  ["right_elbow", "right_wrist"], ["left_shoulder", "left_hip"],
  ["right_shoulder", "right_hip"], ["left_hip", "right_hip"],
  ["left_hip", "left_knee"], ["left_knee", "left_ankle"],
  ["left_ankle", "left_heel"], ["left_heel", "left_foot_index"],
  ["left_ankle", "left_foot_index"], ["right_hip", "right_knee"],
  ["right_knee", "right_ankle"], ["right_ankle", "right_heel"],
  ["right_heel", "right_foot_index"], ["right_ankle", "right_foot_index"],
];

function element(tag, className, content) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (content !== undefined) node.textContent = content;
  return node;
}
function svgElement(tag, attributes = {}, content) {
  const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, String(value));
  if (content !== undefined) node.textContent = content;
  return node;
}
function finite(value) { return typeof value === "number" && Number.isFinite(value); }
function clamp(value, min, max) { return Math.max(min, Math.min(max, value)); }
function number(value, digits = 1) {
  if (!finite(value)) return "—";
  return value.toLocaleString(undefined, { maximumFractionDigits: digits, minimumFractionDigits: 0 });
}
function timeLabel(seconds, precise = true) {
  if (!finite(seconds)) return "0:00";
  const time = Math.max(0, seconds);
  const minutes = Math.floor(time / 60);
  const wholeSeconds = Math.floor(time % 60).toString().padStart(2, "0");
  const fraction = Math.floor((time % 1) * 100 + 0.00001).toString().padStart(2, "0");
  return `${minutes}:${wholeSeconds}${precise ? `.${fraction}` : ""}`;
}
function humanPhase(id) {
  if (!id || id === "unknown" || id === "unclassified") return "Phase not identified";
  return String(id).replace(/_/g, " ").replace(/^\w/, (c) => c.toUpperCase());
}
function phaseColor(id) { return PHASE_COLORS[id] || "#a8bfa0"; }
function maxFrame() { return Math.max(0, (state.report?.frames?.length || 1) - 1); }
function fps() { return state.report?.metadata?.fps || 30; }
function currentData() { return state.report?.frames?.[state.currentFrame]; }
function currentDefinition() {
  return state.report?.metric_definitions?.find((metric) => metric.key === state.metric)
    || { key: state.metric, label: METRIC_SHORT_NAMES[state.metric] || state.metric, unit: "", description: "" };
}
function showError(message) {
  $("global-error").textContent = message;
  $("global-error").hidden = !message;
}
function announce(message) { $("announcement").textContent = message; }
function showPanel(name) {
  for (const panel of ["upload", "processing", "results"]) $( `${panel}-panel` ).hidden = panel !== name;
}
function stopPolling() {
  if (state.pollingTimer) clearTimeout(state.pollingTimer);
  state.pollingTimer = null;
}
function setConnection(online) {
  $("connection-status").classList.toggle("offline", !online);
  $("connection-status").lastChild.textContent = online ? " Local processing" : " Connection lost";
}
function errorMessage(body, fallback) {
  if (typeof body?.detail === "string") return body.detail;
  if (Array.isArray(body?.detail)) {
    return body.detail.map((part) => `${part.loc?.slice(-1)?.[0] || "Input"}: ${part.msg || "Invalid value"}`).join(". ");
  }
  if (typeof body?.error === "string") return body.error;
  if (typeof body?.message === "string") return body.message;
  return fallback;
}
async function api(url, options = {}) {
  const { timeoutMs = options.method === "POST" ? 120000 : 15000, ...requestOptions } = options;
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  let response; let body;
  try {
    response = await fetch(url, { cache: "no-store", ...requestOptions, signal: controller.signal });
    try { body = await response.json(); }
    catch (error) { if (controller.signal.aborted) throw error; body = null; }
    setConnection(true);
  } catch {
    setConnection(false);
    const error = new Error(controller.signal.aborted
      ? "The local analyzer took too long to respond. Check the server; the analysis may still be running."
      : "Could not connect to the local analyzer. Check that the development server is running.");
    error.retryable = true;
    throw error;
  } finally { clearTimeout(timer); }
  if (!response.ok) {
    const error = new Error(errorMessage(body, `The analyzer returned an error (${response.status}). Please try again.`));
    error.status = response.status;
    error.retryable = response.status >= 500;
    throw error;
  }
  if (body === null) throw new Error("The analyzer returned an invalid response. Check the server log and try again.");
  return body;
}

function acceptFile(file) {
  if (!file) return;
  const error = !/\.(mp4|mov)$/i.test(file.name)
    ? "Choose an MP4 or MOV video. Other file types are not supported."
    : !file.size ? "This file is empty. Choose a video with content."
    : file.size > 200 * 1024 * 1024 ? "Video is too large. Maximum upload is 200 MB." : "";
  if (error) {
    showError(error);
    $("video-file").value = "";
    $("video-file").required = true;
    state.selectedFile = null;
    $("file-label").textContent = "Drop your kicking video here";
    $("file-detail").textContent = "MP4 or MOV · Up to 200 MB · 120 seconds";
    $("drop-zone").classList.remove("has-file");
    return;
  }
  showError("");
  state.selectedFile = file;
  $("file-label").textContent = file.name;
  $("file-detail").textContent = `${number(file.size / (1024 * 1024), 1)} MB · Ready to analyze`;
  $("drop-zone").classList.add("has-file");
  // Dropped files can be selected without modifying the browser's FileList.
  $("video-file").required = false;
  announce(`${file.name} selected`);
}
function formSettings(form) {
  const values = new FormData(form);
  const settings = {};
  for (const key of ["foot", "shot_type", "camera_view", "camera_motion", "direction"]) {
    if (values.has(key)) settings[key] = values.get(key);
  }
  for (const key of ["scale_m_per_px", "contact_frame"]) {
    if (values.has(key)) settings[key] = values.get(key)?.toString().trim() ? Number(values.get(key)) : null;
  }
  return settings;
}
function prepareProgress(name, message = "Preparing your video…") {
  if ($("settings-dialog").open) $("settings-dialog").close();
  clearPlayback();
  showPanel("processing");
  $("processing-heading").textContent = "Finding the movement";
  $("processing-name").textContent = name || "Your video";
  $("processing-message").textContent = message;
  $("analysis-progress").value = 0;
  $("progress-percent").textContent = "0%";
  $("processing-failure").hidden = true;
  $("back-to-upload").hidden = true;
  $("retry-analysis").hidden = true;
  $("reload-session").hidden = true;
  document.querySelector(".processing-symbol").hidden = false;
}
async function startUpload(event) {
  event.preventDefault();
  const file = state.selectedFile || $("video-file").files?.[0];
  if (!file) { showError("Choose a kicking video to begin."); return; }
  if (!$("upload-form").reportValidity()) return;
  const payload = new FormData();
  payload.append("file", file, file.name);
  for (const [key, value] of Object.entries(formSettings($("upload-form")))) {
    if (value !== null) payload.append(key, String(value));
  }
  showError("");
  stopPolling();
  const token = ++state.selectionVersion;
  state.restoreFrame = 0;
  $("analyze-button").disabled = true;
  prepareProgress(file.name, "Uploading your video to the local analyzer…");
  try {
    const job = await api("/api/analyses", { method: "POST", body: payload });
    if (token !== state.selectionVersion) { await refreshSessions(); return; }
    await selectJob(job);
  } catch (error) {
    if (token === state.selectionVersion) { showPanel("upload"); showError(error.message); }
  } finally { $("analyze-button").disabled = false; }
}
async function startExample() {
  showError("");
  $("load-example").disabled = true;
  stopPolling();
  const token = ++state.selectionVersion;
  state.restoreFrame = 0;
  prepareProgress("Sample soccer footage", "Preparing the sample soccer clip…");
  try {
    const job = await api("/api/analyses/example", { method: "POST" });
    if (token !== state.selectionVersion) { await refreshSessions(); return; }
    await selectJob(job);
  } catch (error) {
    if (token === state.selectionVersion) { showPanel("upload"); showError(error.message); }
  } finally { $("load-example").disabled = false; }
}
function updateProgress(job) {
  const percent = Math.round(clamp(Number(job.progress) || 0, 0, 1) * 100);
  $("processing-name").textContent = job.name || "Your video";
  $("processing-message").textContent = job.message || (job.status === "queued" ? "Waiting for the analyzer…" : "Analyzing your video…");
  $("progress-percent").textContent = `${percent}%`;
  $("analysis-progress").value = percent;
  if (job.status === "failed") {
    $("processing-heading").textContent = "This video needs another look";
    $("processing-failure").textContent = typeof job.error === "string" ? job.error : job.message || "Analysis could not be completed. Try a shorter, clear clip with the player fully visible.";
    $("processing-failure").hidden = false;
    $("back-to-upload").hidden = false;
    $("retry-analysis").hidden = false;
    document.querySelector(".processing-symbol").hidden = true;
    announce("Analysis could not be completed. Review the error message.");
  }
}
function handleLoadError(error, id, token) {
  if (token !== state.selectionVersion) return;
  showError(error.message);
  $("processing-heading").textContent = error.retryable ? "Reconnecting to the analyzer" : "Unable to load this analysis";
  $("processing-message").textContent = error.retryable
    ? "Your server may still be processing. Reconnecting automatically…" : "Reload the session to try again.";
  $("back-to-upload").hidden = false;
  $("reload-session").hidden = false;
  document.querySelector(".processing-symbol").hidden = !error.retryable;
  if (error.retryable) {
    const delay = Math.min(10000, 1100 * 2 ** Math.min(state.pollFailures++, 4));
    state.pollingTimer = setTimeout(() => pollJob(id, token), delay);
  }
}
async function pollJob(id, token) {
  if (token !== state.selectionVersion) return;
  try { await handleJob(await api(`/api/analyses/${encodeURIComponent(id)}`), token); }
  catch (error) { handleLoadError(error, id, token); }
}
async function selectJob(jobOrId, preserveFrame = false) {
  const id = typeof jobOrId === "string" ? jobOrId : jobOrId.id;
  if (!id) return;
  stopPolling();
  video.pause();
  const token = ++state.selectionVersion;
  state.jobId = id;
  if (!preserveFrame) state.restoreFrame = 0;
  showError("");
  prepareProgress(typeof jobOrId === "object" ? jobOrId.name : "Loading your session", "Loading analysis…");
  history.replaceState(null, "", `${location.pathname}#analysis=${encodeURIComponent(id)}`);
  renderSessions();
  try {
    const job = typeof jobOrId === "object" ? jobOrId : await api(`/api/analyses/${encodeURIComponent(id)}`);
    if (token !== state.selectionVersion) return;
    await handleJob(job, token);
  } catch (error) {
    handleLoadError(error, id, token);
  }
}
async function handleJob(job, token) {
  if (token !== state.selectionVersion) return;
  state.pollFailures = 0;
  showError("");
  $("reload-session").hidden = true;
  $("processing-heading").textContent = "Finding the movement";
  const index = state.jobs.findIndex((entry) => entry.id === job.id);
  if (index === -1) state.jobs.unshift(job); else state.jobs[index] = job;
  renderSessions();
  updateProgress(job);
  if (job.status === "complete") {
    $("processing-message").textContent = "Opening your analysis…";
    let report;
    try { report = await api(`/api/analyses/${encodeURIComponent(job.id)}/report`); }
    catch (error) {
      if (error.status === 409) {
        // A damaged saved report becomes a retryable failed job server-side;
        // another tab may also have started a recalculation since our last GET.
        const latest = await api(`/api/analyses/${encodeURIComponent(job.id)}`);
        if (latest.status !== "complete") { await handleJob(latest, token); return; }
      }
      throw error;
    }
    if (token !== state.selectionVersion) return;
    renderReport(report, job);
    refreshSessions().catch(() => {});
    return;
  }
  if (job.status === "failed") return;
  state.pollingTimer = setTimeout(() => pollJob(job.id, token), 1100);
}
async function refreshSessions() {
  const token = ++state.sessionRefreshVersion;
  const response = await api("/api/analyses");
  if (token !== state.sessionRefreshVersion) return;
  const jobs = new Map(state.jobs.map((job) => [job.id, job]));
  for (const job of response.analyses || []) {
    const current = jobs.get(job.id);
    if (!current || new Date(job.updated_at) >= new Date(current.updated_at)) jobs.set(job.id, job);
  }
  state.jobs = [...jobs.values()].sort((a, b) => new Date(b.created_at) - new Date(a.created_at));
  renderSessions();
}
async function retryAnalysis() {
  if (!state.jobId) return;
  const id = state.jobId;
  const token = state.selectionVersion;
  $("retry-analysis").disabled = true;
  showError("");
  try {
    const job = await api(`/api/analyses/${encodeURIComponent(id)}`, {
      method: "PATCH", headers: { "Content-Type": "application/json" }, body: "{}",
    });
    if (id === state.jobId && token === state.selectionVersion) await selectJob(job);
  } catch (error) { if (id === state.jobId && token === state.selectionVersion) showError(error.message); }
  finally { $("retry-analysis").disabled = false; }
}
function renderSessions() {
  const list = $("session-list");
  list.replaceChildren();
  if (!state.jobs.length) {
    list.append(element("p", "session-empty", "Your analyzed clips will appear here."));
    return;
  }
  for (const job of state.jobs) {
    const button = element("button", `session-item ${job.status || ""}${job.id === state.jobId ? " current" : ""}`);
    button.type = "button";
    button.title = `${job.name || "Video analysis"} · ${job.status}`;
    if (job.id === state.jobId) button.setAttribute("aria-current", "true");
    const icon = element("span", "session-icon", job.status === "complete" ? "▷" : job.status === "failed" ? "!" : "◷");
    icon.setAttribute("aria-hidden", "true");
    const text = element("span");
    text.append(element("strong", "", job.name || "Video analysis"));
    const date = job.created_at ? new Date(job.created_at) : null;
    const dateText = date && !Number.isNaN(date.getTime()) ? date.toLocaleDateString(undefined, { month: "short", day: "numeric" }) : "";
    const status = job.status === "complete" ? "Ready to review" : job.status === "failed" ? "Analysis failed" : "Processing";
    text.append(element("small", "", `${status}${dateText ? ` · ${dateText}` : ""}`));
    button.append(icon, text);
    button.addEventListener("click", () => selectJob(job.id));
    list.append(button);
  }
}
function newAnalysis() {
  stopPolling();
  state.selectionVersion++;
  state.jobId = null;
  clearPlayback();
  showPanel("upload");
  showError("");
  history.replaceState(null, "", location.pathname);
  renderSessions();
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function renderReport(report, job) {
  if (!Array.isArray(report.frames) || !report.frames.length) throw new Error("The report contains no decoded video frames. Please analyze the video again.");
  state.report = report;
  state.currentFrame = clamp(state.restoreFrame || 0, 0, maxFrame());
  state.pendingSeekFrame = state.currentFrame;
  const settings = report.settings || {};
  const metadata = report.metadata || {};
  $("result-name").textContent = report.name || job.name || "Kicking analysis";
  $("result-name").title = report.name || job.name || "Kicking analysis";
  $("result-context").textContent = [SHOT_NAMES[settings.shot_type], `${settings.foot === "left" ? "Left" : "Right"} foot`, VIEW_NAMES[settings.camera_view], `${number(metadata.fps, 2)} fps`].filter(Boolean).join(" · ");
  renderQuality(report.quality || {});
  renderContact(report.contact || {});
  renderSummary(report.summary || [], report.quality || {});
  renderObservations(report.observations || []);
  renderDefinitions(report.metric_definitions || []);
  renderMetricTabs(report.metric_definitions || []);
  renderTimeline(report.phases || []);
  $("frame-slider").max = maxFrame();
  $("settings-form").elements.namedItem("contact_frame").max = maxFrame();
  canvas.width = metadata.width || 1280;
  canvas.height = metadata.height || 720;
  for (const [id, key] of [["download-video", "overlay"], ["download-csv", "csv"], ["download-json", "json"]]) {
    const link = $(id);
    const url = report.urls?.[key];
    link.hidden = !url;
    if (url) link.href = url;
  }
  $("video-error").hidden = true;
  const source = report.urls?.video;
  if (!source) throw new Error("The report is missing its playable video URL. Check the server output.");
  const sourceChanged = video.getAttribute("src") !== source;
  if (sourceChanged) {
    video.src = source;
    video.load();
  }
  video.playbackRate = Number($("playback-speed").value);
  showPanel("results");
  renderGraph();
  renderFrame(state.currentFrame);
  if (!sourceChanged && video.readyState >= 1) seekFrame(state.currentFrame, false);
  startVideoFrameLoop();
  announce("Analysis ready. Review the contact frame and measurements.");
}
function renderQuality(quality) {
  const warnings = Array.isArray(quality.warnings) ? quality.warnings : [];
  $("quality-notice").replaceChildren();
  $("quality-notice").hidden = !warnings.length;
  if (warnings.length) {
    const context = quality.identity_suspect_frames?.length ? "Check leg identity" : "2D measurement limits";
    $("quality-notice").append(element("summary", "", `Measurement context · ${context} · ${warnings.length} notes`));
    for (const warning of warnings) $("quality-notice").append(element("p", "", warning));
  }
}
function renderContact(contact) {
  const available = Number.isInteger(contact.frame) && contact.frame >= 0;
  const confirmed = contact.source === "manual";
  $("contact-status").textContent = confirmed ? "Manually selected" : available ? "Review needed" : "Select a frame";
  $("contact-status").className = `pill pill-review${confirmed ? " pill-confirmed" : ""}`;
  $("contact-heading").textContent = confirmed ? "Your contact frame is set." : "Find the moment of impact.";
  $("contact-reason").textContent = contact.reason || (available ? "A motion-based candidate needs visual confirmation in the video." : "Automatic contact estimation was unavailable. Choose the contact frame from the video.");
  $("contact-frame-label").textContent = confirmed ? "Selected frame" : "Candidate frame";
  $("contact-frame").textContent = available ? `${contact.frame} · ${timeLabel(contact.frame / fps())}` : "Not identified";
  $("jump-contact").disabled = !available;
  $("set-contact").disabled = false;
  $("chart-contact-source").textContent = confirmed ? "selected" : "candidate";
}
function renderSummary(summary, quality) {
  const container = $("summary-metrics");
  container.replaceChildren();
  for (const item of summary) {
    const row = element("div", "summary-metric");
    const heading = element("div", "summary-metric-header");
    heading.append(element("span", "summary-metric-name", item.label || item.key));
    const value = element("div", "summary-metric-value", finite(item.value) ? number(item.value, 1) : "—");
    if (finite(item.value) && item.unit) value.append(element("span", "", item.unit));
    heading.append(value);
    row.append(heading);
    if (item.detail) row.append(element("p", "summary-metric-detail", item.detail));
    container.append(row);
  }
  if (!summary.length) container.append(element("p", "summary-metric-detail", "No summary measurements are available for this clip. Review landmark visibility and camera setup."));
  const coverage = finite(quality.pose_coverage) ? clamp(quality.pose_coverage, 0, 1) : null;
  $("pose-coverage").textContent = coverage === null ? "—" : `${number(coverage * 100, 1)}%`;
  $("pose-coverage-bar").style.width = `${(coverage || 0) * 100}%`;
}
function renderObservations(observations) {
  const container = $("observations");
  container.replaceChildren();
  for (const item of observations) {
    const card = element("div", `observation ${["measurement", "limitation", "review"].includes(item.kind) ? item.kind : ""}`);
    const heading = element("div", "observation-title");
    const icon = element("span", "", item.kind === "measurement" ? "↗" : "i");
    icon.setAttribute("aria-hidden", "true");
    heading.append(icon, document.createTextNode(item.title || "Observation"));
    card.append(heading, element("p", "", item.body || ""));
    container.append(card);
  }
  if (!observations.length) container.append(element("p", "metric-description", "No supported technique observations could be generated for this clip."));
}
function renderDefinitions(definitions) {
  const container = $("metric-definitions");
  container.replaceChildren();
  for (const metric of definitions) {
    const item = element("div", "metric-definition");
    item.append(element("strong", "", `${metric.label}${metric.unit ? ` (${metric.unit})` : ""}`), element("p", "", metric.description || ""));
    container.append(item);
  }
}
function renderMetricTabs(definitions) {
  const container = $("metric-tabs");
  container.replaceChildren();
  const metrics = definitions.length ? definitions : Object.keys(METRIC_SHORT_NAMES).map((key) => ({ key, label: METRIC_SHORT_NAMES[key] }));
  if (!metrics.some((item) => item.key === state.metric)) state.metric = metrics[0].key;
  for (const metric of metrics) {
    const button = element("button", `metric-tab${metric.key === state.metric ? " selected" : ""}`, METRIC_SHORT_NAMES[metric.key] || metric.label);
    button.type = "button";
    button.dataset.metric = metric.key;
    button.setAttribute("aria-pressed", String(metric.key === state.metric));
    button.title = metric.label;
    button.addEventListener("click", () => {
      state.metric = metric.key;
      for (const tab of container.querySelectorAll("button")) {
        tab.classList.toggle("selected", tab === button);
        tab.setAttribute("aria-pressed", String(tab === button));
      }
      renderGraph();
      renderGraphMarker();
    });
    container.append(button);
  }
}
function renderTimeline(phases) {
  $("phase-timeline").replaceChildren();
  $("phase-legend").replaceChildren();
  const total = maxFrame() + 1;
  for (const phase of phases) {
    const start = clamp(phase.start_frame, 0, maxFrame());
    const end = clamp(phase.end_frame, start, maxFrame());
    const label = phase.label || humanPhase(phase.id);
    const button = element("button", "phase-segment");
    button.type = "button";
    button.style.left = `${(start / total) * 100}%`;
    button.style.width = `${Math.max(1 / total, (end - start + 1) / total) * 100}%`;
    button.style.background = phaseColor(phase.id);
    button.dataset.start = start;
    button.dataset.end = end;
    button.title = `${label} · Frames ${start}–${end} · ${phase.source || "motion estimate"}`;
    button.setAttribute("aria-label", `Seek to ${label}, frame ${start}`);
    button.addEventListener("click", () => seekFrame(start));
    $("phase-timeline").append(button);
    const legend = element("span");
    const swatch = element("i");
    swatch.style.background = phaseColor(phase.id);
    swatch.setAttribute("aria-hidden", "true");
    legend.append(swatch, document.createTextNode(label));
    $("phase-legend").append(legend);
  }
  if (!phases.length) $("phase-legend").append(element("span", "", "No reliable phases identified. Review contact and landmark visibility."));
  $("phase-timeline").append(element("div", "timeline-marker"));
  $("phase-source").textContent = state.report.contact?.source === "manual" ? "Selected contact · Other boundaries estimated" : "Motion estimates · Review with video";
}

function seekFrame(frame, pause = true) {
  if (!state.report) return;
  const index = clamp(Math.round(Number(frame) || 0), 0, maxFrame());
  if (pause) video.pause();
  state.currentFrame = index;
  state.pendingSeekFrame = index;
  const frameTime = state.report.frames[index]?.time;
  const time = finite(frameTime) ? frameTime : index / fps();
  if (video.readyState >= 1) video.currentTime = Math.min(time + 0.00001, Math.max(0, (video.duration || time + 1) - 0.0001));
  renderFrame(index);
}
function renderFrame(index) {
  if (!state.report) return;
  state.currentFrame = clamp(Math.round(index), 0, maxFrame());
  const frame = currentData();
  const time = finite(frame?.time) ? frame.time : state.currentFrame / fps();
  $("frame-slider").value = state.currentFrame;
  $("frame-slider").setAttribute("aria-valuetext", `Frame ${state.currentFrame} of ${maxFrame()}, ${timeLabel(time)}`);
  $("frame-count").textContent = `${state.currentFrame} / ${maxFrame()}`;
  $("frame-badge").textContent = `FRAME ${String(state.currentFrame).padStart(3, "0")}`;
  $("playback-time").textContent = `${timeLabel(time)} / ${timeLabel(state.report.metadata.duration || (maxFrame() + 1) / fps())}`;
  const phase = state.report.phases?.find((item) => state.currentFrame >= item.start_frame && state.currentFrame <= item.end_frame);
  $("phase-badge").textContent = phase?.label || humanPhase(frame?.phase);
  for (const segment of $("phase-timeline").querySelectorAll(".phase-segment")) {
    segment.classList.toggle("current", state.currentFrame >= Number(segment.dataset.start) && state.currentFrame <= Number(segment.dataset.end));
  }
  const marker = $("phase-timeline").querySelector(".timeline-marker");
  if (marker) marker.style.left = `${(state.currentFrame / Math.max(1, maxFrame())) * 100}%`;
  drawPose(frame);
  renderGraphMarker();
}
function validLandmark(point) {
  return point && finite(point.x) && finite(point.y) && (point.visibility ?? 1) >= 0.5;
}
function drawPose(frame) {
  const ctx = canvasContext;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!state.overlay || !frame) return;
  const landmarks = frame.landmarks || {};
  const foot = state.report.settings?.foot || "right";
  const size = Math.max(1, canvas.width / 640);
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  for (const [a, b] of CONNECTIONS) {
    const p1 = landmarks[a]; const p2 = landmarks[b];
    if (!validLandmark(p1) || !validLandmark(p2)) continue;
    const selected = a.startsWith(foot) && b.startsWith(foot);
    ctx.beginPath();
    ctx.moveTo(p1.x, p1.y); ctx.lineTo(p2.x, p2.y);
    ctx.lineWidth = 5 * size;
    ctx.strokeStyle = "rgba(13, 35, 25, .6)";
    ctx.stroke();
    ctx.lineWidth = 2.2 * size;
    ctx.strokeStyle = selected ? "#d8f29c" : "#f2f5eb";
    ctx.stroke();
  }
  for (const [name, point] of Object.entries(landmarks)) {
    if (!validLandmark(point)) continue;
    ctx.beginPath();
    ctx.arc(point.x, point.y, 3 * size, 0, Math.PI * 2);
    ctx.fillStyle = name.startsWith(foot) ? "#d8f29c" : "#f2f5eb";
    ctx.fill();
    ctx.strokeStyle = "#1f3e2c"; ctx.lineWidth = 1.2 * size; ctx.stroke();
  }
  const metrics = frame.metrics || {};
  const tags = [
    [landmarks[`${foot}_knee`], metrics.knee_flexion, "Knee", 12, -9],
    [landmarks[`${foot}_hip`], metrics.hip_angle, "Hip", 12, -9],
    [landmarks[`${foot === "right" ? "left" : "right"}_knee`], metrics.plant_knee_flexion, "Plant", -85, 17],
  ];
  ctx.font = `500 ${10 * size}px "Segoe UI", sans-serif`;
  for (const [point, value, label, offsetX, offsetY] of tags) {
    if (!validLandmark(point) || !finite(value)) continue;
    const text = `${label} ${number(value, 0)}°`;
    const width = ctx.measureText(text).width + 12 * size;
    const x = clamp(point.x + offsetX * size, 2, canvas.width - width - 2);
    const y = clamp(point.y + offsetY * size, 17 * size, canvas.height - 2);
    ctx.fillStyle = "rgba(23, 46, 33, .88)";
    ctx.fillRect(x, y - 14 * size, width, 19 * size);
    ctx.fillStyle = "#edf6d8";
    ctx.fillText(text, x + 6 * size, y);
  }
}
function updateFromTime(time, presentedFrame = false) {
  if (!state.report || !finite(time) || $("results-panel").hidden) return;
  // Decoder timestamps identify frame starts; ordinary playback time lies
  // anywhere INSIDE a frame and must not round ahead to the next pose.
  const position = time * fps();
  const frame = clamp(presentedFrame ? Math.round(position) : Math.floor(position + 0.000001), 0, maxFrame());
  if (state.pendingSeekFrame !== null && video.paused) {
    if (frame !== state.pendingSeekFrame) return;
    state.pendingSeekFrame = null;
  }
  if (frame !== state.currentFrame) renderFrame(frame);
}
function stopVideoFrameLoop() {
  state.frameLoopVersion++;
  if (state.playbackCallback !== null && video.cancelVideoFrameCallback) video.cancelVideoFrameCallback(state.playbackCallback);
  if (state.animationFrame !== null) cancelAnimationFrame(state.animationFrame);
  state.playbackCallback = null;
  state.animationFrame = null;
}
function clearPlayback() {
  video.pause();
  stopVideoFrameLoop();
  state.report = null;
  state.graph = null;
  state.pendingSeekFrame = null;
  video.removeAttribute("src");
  video.load();
  canvasContext.clearRect(0, 0, canvas.width, canvas.height);
}
function startVideoFrameLoop() {
  stopVideoFrameLoop();
  if (!state.report || $("results-panel").hidden) return;
  const token = state.frameLoopVersion;
  if (typeof video.requestVideoFrameCallback === "function") {
    const update = (_now, metadata) => {
      if (token !== state.frameLoopVersion) return;
      updateFromTime(metadata.mediaTime, true);
      state.playbackCallback = video.requestVideoFrameCallback(update);
    };
    state.playbackCallback = video.requestVideoFrameCallback(update);
  } else {
    const update = () => {
      if (token !== state.frameLoopVersion || video.paused) { state.animationFrame = null; return; }
      updateFromTime(video.currentTime);
      state.animationFrame = requestAnimationFrame(update);
    };
    if (!video.paused) state.animationFrame = requestAnimationFrame(update);
  }
}
async function togglePlayback() {
  if (!state.report) return;
  if (!video.paused) { video.pause(); return; }
  if (video.ended || state.currentFrame >= maxFrame()) seekFrame(0);
  try { await video.play(); }
  catch (error) { showError(`The video could not start: ${error.message}. Download the annotated MP4 if browser playback is unavailable.`); }
}

function renderGraph() {
  if (!state.report) return;
  const definition = currentDefinition();
  const frames = state.report.frames;
  const values = frames.map((frame) => frame.metrics?.[state.metric]).filter(finite);
  const grid = $("chart-grid"); const series = $("chart-series"); const contactGroup = $("chart-contact");
  grid.replaceChildren(); series.replaceChildren(); contactGroup.replaceChildren();
  $("chart-title").textContent = `${definition.label} over video time`;
  $("chart-description").textContent = `${definition.description || definition.label} Gaps indicate missing measurements. Use the frame slider or click the chart to seek.`;
  $("chart-metric-label").textContent = definition.label;
  $("metric-description").textContent = definition.description || "";
  $("chart-current-unit").textContent = definition.unit || "";
  $("chart-empty").hidden = values.length > 0;
  $("chart-empty").textContent = "This measurement is unavailable for the clip. Missing or unreliable landmarks are left blank.";
  let min = values.length ? Math.min(...values) : 0;
  let max = values.length ? Math.max(...values) : 1;
  const padding = Math.max((max - min) * 0.12, max === min ? Math.max(Math.abs(max) * 0.1, 1) : 0.1);
  min = Math.max(0, min - padding);
  max += padding;
  if (max <= min) max = min + 1;
  const left = 58; const top = 18; const width = 842; const height = 184;
  const duration = Math.max(frames[frames.length - 1]?.time || maxFrame() / fps(), 1 / fps());
  const x = (time) => left + (time / duration) * width;
  const y = (value) => top + height - ((value - min) / (max - min)) * height;
  state.graph = { min, max, duration, left, top, width, height, x, y };
  for (let i = 0; i <= 4; i++) {
    const value = min + ((max - min) * i) / 4;
    const rowY = y(value);
    grid.append(svgElement("line", { x1: left, y1: rowY, x2: left + width, y2: rowY, class: "chart-grid-line" }));
    grid.append(svgElement("text", { x: left - 12, y: rowY + 4, class: "chart-tick", "text-anchor": "end" }, number(value, max > 100 ? 0 : 1)));
  }
  for (let i = 0; i <= 4; i++) {
    const time = (duration * i) / 4;
    grid.append(svgElement("text", { x: x(time), y: top + height + 25, class: "chart-tick", "text-anchor": i === 0 ? "start" : i === 4 ? "end" : "middle" }, `${number(time, 1)}s`));
  }
  // Each contiguous run is its own path so missing pose never becomes invented data.
  let points = [];
  function flush() {
    if (!points.length) return;
    if (points.length === 1) {
      series.append(svgElement("circle", { cx: points[0][0], cy: points[0][1], r: 2, fill: "#5e8050" }));
    } else {
      const d = points.map((point, index) => `${index === 0 ? "M" : "L"}${point[0].toFixed(2)},${point[1].toFixed(2)}`).join(" ");
      series.append(svgElement("path", { d, class: "metric-line" }));
    }
    points = [];
  }
  for (const frame of frames) {
    const value = frame.metrics?.[state.metric];
    if (!finite(value)) { flush(); continue; }
    points.push([x(frame.time), y(value)]);
  }
  flush();
  const contact = state.report.contact?.frame;
  if (Number.isInteger(contact) && contact >= 0 && contact <= maxFrame()) {
    const contactX = x(frames[contact]?.time ?? contact / fps());
    contactGroup.append(svgElement("line", { x1: contactX, y1: top, x2: contactX, y2: top + height, class: "contact-line" }));
  }
  renderGraphMarker();
}
function renderGraphMarker() {
  if (!state.graph || !state.report) return;
  const frame = currentData();
  const value = frame?.metrics?.[state.metric];
  $("chart-current-value").textContent = number(value, 1);
  const marker = $("chart-marker");
  marker.replaceChildren();
  const { x, y, top, height } = state.graph;
  const markerX = x(frame?.time ?? state.currentFrame / fps());
  marker.append(svgElement("line", { x1: markerX, y1: top, x2: markerX, y2: top + height, class: "current-line" }));
  if (finite(value)) marker.append(svgElement("circle", { cx: markerX, cy: y(value), r: 4.5, class: "current-dot" }));
}
function chartPoint(event) {
  if (!state.graph || !state.report) return null;
  const svg = $("metric-chart");
  const point = svg.createSVGPoint();
  point.x = event.clientX; point.y = event.clientY;
  const matrix = svg.getScreenCTM();
  if (!matrix) return null;
  const location = point.matrixTransform(matrix.inverse());
  const ratio = clamp((location.x - state.graph.left) / state.graph.width, 0, 1);
  const frame = clamp(Math.round((ratio * state.graph.duration) * fps()), 0, maxFrame());
  return { frame, x: event.clientX };
}
function graphHover(event) {
  const point = chartPoint(event);
  if (!point) return;
  const frame = state.report.frames[point.frame];
  const value = frame.metrics?.[state.metric];
  const definition = currentDefinition();
  const tooltip = $("chart-tooltip");
  tooltip.textContent = `${timeLabel(frame.time)} · Frame ${point.frame} · ${finite(value) ? `${number(value, 1)} ${definition.unit || ""}` : "No measurement"}`;
  tooltip.hidden = false;
  const bounds = $("chart-wrapper").getBoundingClientRect();
  tooltip.style.left = `${clamp(point.x - bounds.left - tooltip.offsetWidth / 2, 0, Math.max(0, bounds.width - tooltip.offsetWidth))}px`;
}

async function updateSettings(settings) {
  if (!state.jobId) return;
  state.restoreFrame = state.currentFrame;
  const id = state.jobId;
  const token = state.selectionVersion;
  const job = await api(`/api/analyses/${encodeURIComponent(id)}`, {
    method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify(settings),
  });
  if (state.jobId !== id || state.selectionVersion !== token) return;
  $("settings-dialog").close();
  await selectJob(job, true);
}
async function selectContact() {
  if (!state.report) return;
  $("set-contact").disabled = true;
  showError("");
  try { await updateSettings({ contact_frame: state.currentFrame }); }
  catch (error) { showError(error.message); }
  finally { $("set-contact").disabled = false; }
}
function openSettings() {
  if (!state.report) return;
  const settings = { ...state.report.settings };
  settings.contact_frame = state.report.contact?.source === "manual" ? state.report.contact.frame : null;
  const form = $("settings-form");
  for (const [key, value] of Object.entries(settings)) {
    const control = form.elements.namedItem(key);
    if (control) control.value = value ?? "";
  }
  $("settings-error").hidden = true;
  $("settings-dialog").showModal();
}

$("upload-form").addEventListener("submit", startUpload);
$("video-file").addEventListener("change", (event) => acceptFile(event.target.files?.[0]));
$("load-example").addEventListener("click", startExample);
for (const name of ["dragenter", "dragover"]) $("drop-zone").addEventListener(name, (event) => { event.preventDefault(); $("drop-zone").classList.add("dragging"); });
for (const name of ["dragleave", "drop"]) $("drop-zone").addEventListener(name, (event) => { event.preventDefault(); $("drop-zone").classList.remove("dragging"); });
$("drop-zone").addEventListener("drop", (event) => acceptFile(event.dataTransfer.files?.[0]));
for (const id of ["new-analysis", "new-analysis-nav", "back-to-upload"]) $(id).addEventListener("click", newAnalysis);
$("refresh-sessions").addEventListener("click", () => refreshSessions().catch((error) => showError(error.message)));
$("retry-analysis").addEventListener("click", retryAnalysis);
$("reload-session").addEventListener("click", () => { if (state.jobId) selectJob(state.jobId); });
$("play-pause").addEventListener("click", togglePlayback);
$("previous-frame").addEventListener("click", () => seekFrame(state.currentFrame - 1));
$("next-frame").addEventListener("click", () => seekFrame(state.currentFrame + 1));
$("frame-slider").addEventListener("input", (event) => seekFrame(event.target.value));
$("playback-speed").addEventListener("change", (event) => { video.playbackRate = Number(event.target.value); });
$("overlay-toggle").addEventListener("change", (event) => { state.overlay = event.target.checked; drawPose(currentData()); });
$("video-stage").addEventListener("keydown", (event) => {
  if (event.target !== $("video-stage")) return;
  if ([" ", "ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) event.preventDefault();
  if (event.key === " ") togglePlayback();
  if (event.key === "ArrowLeft") seekFrame(state.currentFrame - (event.shiftKey ? 10 : 1));
  if (event.key === "ArrowRight") seekFrame(state.currentFrame + (event.shiftKey ? 10 : 1));
  if (event.key === "Home") seekFrame(0);
  if (event.key === "End") seekFrame(maxFrame());
});
$("video-stage").addEventListener("click", (event) => { if (event.target === video || event.target === $("video-stage")) togglePlayback(); });
$("fullscreen").addEventListener("click", async () => {
  try {
    if (document.fullscreenElement) await document.exitFullscreen();
    else if ($("video-stage").requestFullscreen) await $("video-stage").requestFullscreen();
    else showError("Full-screen playback is unavailable in this browser. You can download the annotated MP4.");
  } catch (error) { showError(`Unable to open full screen: ${error.message}`); }
});
video.addEventListener("loadedmetadata", () => { if (state.report) seekFrame(state.currentFrame, false); });
video.addEventListener("seeked", () => { if (!video.requestVideoFrameCallback) updateFromTime(video.currentTime); });
video.addEventListener("timeupdate", () => { if (!video.requestVideoFrameCallback) updateFromTime(video.currentTime); });
video.addEventListener("play", () => {
  state.pendingSeekFrame = null;
  $("play-pause").textContent = "Ⅱ"; $("play-pause").setAttribute("aria-label", "Pause video");
  startVideoFrameLoop();
});
video.addEventListener("pause", () => {
  $("play-pause").textContent = "▶"; $("play-pause").setAttribute("aria-label", "Play video");
  if (!video.requestVideoFrameCallback) stopVideoFrameLoop();
});
video.addEventListener("ended", () => renderFrame(maxFrame()));
video.addEventListener("error", () => {
  if (!video.getAttribute("src") || !video.error) return;
  $("video-error").textContent = "The browser could not play this video. Try reopening the session or download the annotated MP4 below.";
  $("video-error").hidden = false;
});
$("metric-chart").addEventListener("click", (event) => { const point = chartPoint(event); if (point) seekFrame(point.frame); });
$("metric-chart").addEventListener("pointermove", graphHover);
$("metric-chart").addEventListener("pointerleave", () => { $("chart-tooltip").hidden = true; });
$("jump-contact").addEventListener("click", () => { if (Number.isInteger(state.report?.contact?.frame)) seekFrame(state.report.contact.frame); });
$("set-contact").addEventListener("click", selectContact);
$("edit-settings").addEventListener("click", openSettings);
for (const id of ["close-settings", "cancel-settings"]) $(id).addEventListener("click", () => $("settings-dialog").close());
$("settings-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!$("settings-form").reportValidity()) return;
  $("save-settings").disabled = true;
  $("settings-error").hidden = true;
  try { await updateSettings(formSettings($("settings-form"))); }
  catch (error) { $("settings-error").textContent = error.message; $("settings-error").hidden = false; }
  finally { $("save-settings").disabled = false; }
});
$("open-downloads").addEventListener("click", () => { $("downloads").scrollIntoView({ behavior: "smooth", block: "center" }); $("download-video").focus({ preventScroll: true }); });
window.addEventListener("hashchange", () => {
  const id = new URLSearchParams(location.hash.slice(1)).get("analysis");
  if (id && id !== state.jobId) selectJob(id);
  else if (!id) newAnalysis();
});

async function init() {
  const checks = await Promise.allSettled([
    refreshSessions(),
    api("/api/examples").then((response) => { $("example-area").hidden = !response.examples?.length; }),
    api("/api/health").then(() => { $("connection-status").classList.remove("offline"); }),
  ]);
  if (checks[2].status === "rejected") {
    $("connection-status").classList.add("offline");
    $("connection-status").lastChild.textContent = " Server unavailable";
    showError(checks[2].reason.message);
  }
  if (checks[0].status === "rejected" && checks[2].status === "fulfilled") showError(checks[0].reason.message);
  const id = new URLSearchParams(location.hash.slice(1)).get("analysis");
  if (id) await selectJob(id);
}
init();
