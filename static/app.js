const statusEl = document.querySelector("#status");
const placeholder = document.querySelector("#placeholder");
const annotated = document.querySelector("#annotated");
const background = document.querySelector("#background");
const diff = document.querySelector("#diff");
const mask = document.querySelector("#mask");
const overlay = document.querySelector("#overlay");
const source = document.querySelector("#source");
const objectsEl = document.querySelector("#objects");
const presenceEl = document.querySelector("#presence");
const zoneCountEl = document.querySelector("#zone-count");
const alertsEl = document.querySelector("#alerts");
const threshold = document.querySelector("#threshold");
const thresholdValue = document.querySelector("#threshold-value");
const otsu = document.querySelector("#otsu");
const median = document.querySelector("#median");
const stopButton = document.querySelector("#stop");
const flipButton = document.querySelector("#flip");
const closeZone = document.querySelector("#close-zone");

const polygons = [];
let draft = [];
let drawing = false;
let facingMode = "user";
let socket = null;
let streaming = false;
let sending = false;
let lastHistory = "";
let captureTimer = null;
let openWait = null;

function connect() {
  if (socket && socket.readyState <= WebSocket.OPEN) return socket;
  const protocol = location.protocol === "https:" ? "wss" : "ws";
  socket = new WebSocket(`${protocol}://${location.host}/ws`);
  socket.onmessage = (event) => showResult(JSON.parse(event.data));
  socket.onopen = () => sendControl();
  socket.onclose = () => {
    socket = null;
    if (streaming) {
      setStatus("Reconnexion...", "warn");
      setTimeout(() => {
        if (streaming) whenOpen(() => {});
      }, 1000);
      return;
    }
    setStatus("En attente", "idle");
  };
  return socket;
}

function sendControl(command, extra) {
  if (!socket || socket.readyState !== WebSocket.OPEN) return;
  const payload = {
    otsu: otsu.checked,
    threshold: Number(threshold.value),
    median: median.checked,
    polygons,
  };
  if (command) payload.cmd = command;
  if (extra) Object.assign(payload, extra);
  socket.send(JSON.stringify(payload));
}

function setStatus(text, kind) {
  statusEl.textContent = text;
  statusEl.className = `status ${kind}`;
}

function showResult(payload) {
  if (payload.history) renderHistory(payload.history);
  if (payload.error) {
    setStatus(payload.error, "alert");
    return;
  }
  if (!payload.annotated) return;
  setImage(annotated, payload.annotated);
  setImage(background, payload.background);
  setImage(diff, payload.diff);
  setImage(mask, payload.mask);
  placeholder.hidden = true;
  overlay.hidden = false;
  objectsEl.textContent = String(payload.objects);
  presenceEl.textContent = `${payload.persist}/${payload.persist_max}`;
  zoneCountEl.textContent = String(payload.zones);
  const kind = payload.learning ? "warn" : payload.status.startsWith("ALERTE") || payload.status.startsWith("SABOTAGE") ? "alert" : "ok";
  setStatus(payload.status, kind);
  redraw();
}

function renderHistory(history) {
  const signature = JSON.stringify(history);
  if (signature === lastHistory) return;
  lastHistory = signature;
  const empty = document.querySelector("#alerts-empty");
  alertsEl.replaceChildren();
  if (empty) empty.hidden = history.length > 0;
  history.slice().reverse().forEach((item) => {
    const line = document.createElement("li");
    const image = document.createElement("img");
    image.src = item.image;
    image.alt = item.kind;
    const text = document.createElement("div");
    text.textContent = label(item.kind);
    const time = document.createElement("div");
    time.textContent = item.time;
    text.append(time);
    line.append(image, text);
    alertsEl.append(line);
  });
}

function label(kind) {
  return {
    mouvement: "Mouvement",
    visage: "Visage",
    obscurcissement: "Objectif masqué",
    deplacement: "Caméra déplacée",
  }[kind] || kind;
}

function setImage(element, encoded) {
  if (!encoded) return;
  element.hidden = false;
  element.src = `data:image/jpeg;base64,${encoded}`;
}

function imagePoint(event) {
  const rect = overlay.getBoundingClientRect();
  if (!annotated.naturalWidth || rect.width === 0) return null;
  const x = (event.clientX - rect.left) * annotated.naturalWidth / rect.width;
  const y = (event.clientY - rect.top) * annotated.naturalHeight / rect.height;
  if (x < 0 || y < 0 || x >= annotated.naturalWidth || y >= annotated.naturalHeight) return null;
  return [Math.round(x), Math.round(y)];
}

function redraw() {
  if (overlay.hidden || !annotated.naturalWidth) return;
  const width = overlay.clientWidth;
  const height = overlay.clientHeight;
  overlay.width = width;
  overlay.height = height;
  const scaleX = width / annotated.naturalWidth;
  const scaleY = height / annotated.naturalHeight;
  const context = overlay.getContext("2d");
  context.clearRect(0, 0, width, height);
  if (draft.length === 0) return;
  context.strokeStyle = "#7ad1ff";
  context.fillStyle = "#7ad1ff";
  context.lineWidth = 2;
  context.beginPath();
  draft.forEach(([x, y], index) => {
    const px = x * scaleX;
    const py = y * scaleY;
    if (index === 0) context.moveTo(px, py);
    else context.lineTo(px, py);
    context.fillRect(px - 3, py - 3, 6, 6);
  });
  context.stroke();
}

async function startCamera(deviceId) {
  const video = {
    width: { ideal: 640 },
    height: { ideal: 480 },
  };
  if (deviceId) video.deviceId = { exact: deviceId };
  else video.facingMode = facingMode;
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video, audio: false });
  } catch (_error) {
    setStatus("Caméra refusée", "alert");
    return;
  }
  stopSource();
  source.srcObject = stream;
  await source.play();
  flipButton.hidden = false;
  await listCameras(stream);
  beginFrames();
}

async function listCameras(stream) {
  const devices = await navigator.mediaDevices.enumerateDevices();
  const videos = devices.filter((device) => device.kind === "videoinput");
  const select = document.querySelector("#devices");
  select.replaceChildren();
  videos.forEach((device, index) => {
    const option = document.createElement("option");
    option.value = device.deviceId;
    option.textContent = device.label || `Caméra ${index + 1}`;
    select.append(option);
  });
  const current = stream.getVideoTracks()[0]?.getSettings().deviceId;
  if (current) select.value = current;
  select.hidden = videos.length < 2;
}

function whenOpen(callback) {
  connect();
  if (openWait) clearInterval(openWait);
  openWait = setInterval(() => {
    if (socket && socket.readyState === WebSocket.OPEN) {
      clearInterval(openWait);
      openWait = null;
      callback();
    }
  }, 50);
}

function beginFrames() {
  streaming = true;
  stopButton.disabled = false;
  whenOpen(() => {
    sendControl("stop");
    captureTimer = setInterval(sendFrame, 120);
  });
}

function sendFrame() {
  if (!streaming || !socket || socket.readyState !== WebSocket.OPEN || sending) return;
  if (!source.videoWidth) return;
  const canvas = document.createElement("canvas");
  canvas.width = source.videoWidth;
  canvas.height = source.videoHeight;
  canvas.getContext("2d").drawImage(source, 0, 0);
  sending = true;
  canvas.toBlob((blob) => {
    if (blob && socket.readyState === WebSocket.OPEN) socket.send(blob);
    sending = false;
  }, "image/jpeg", 0.7);
}

function stopSource() {
  streaming = false;
  sending = false;
  if (captureTimer) clearInterval(captureTimer);
  captureTimer = null;
  if (openWait) clearInterval(openWait);
  openWait = null;
  const stream = source.srcObject;
  if (stream) stream.getTracks().forEach((track) => track.stop());
  source.srcObject = null;
  source.removeAttribute("src");
  source.load();
  flipButton.hidden = true;
  stopButton.disabled = true;
}

document.querySelector("#camera").addEventListener("click", () => {
  facingMode = "user";
  startCamera();
});

flipButton.addEventListener("click", () => {
  facingMode = facingMode === "user" ? "environment" : "user";
  flipButton.textContent = facingMode === "user" ? "Caméra arrière" : "Caméra avant";
  startCamera();
});

document.querySelector("#file").addEventListener("change", (event) => {
  const file = event.target.files[0];
  if (!file) return;
  stopSource();
  source.src = URL.createObjectURL(file);
  source.loop = true;
  source.play().then(beginFrames).catch(() => setStatus("Vidéo illisible", "alert"));
});

document.querySelector("#devices").addEventListener("change", (event) => {
  if (event.target.value) startCamera(event.target.value);
});

document.querySelector("#connect-stream").addEventListener("click", () => {
  const url = document.querySelector("#stream-url").value.trim();
  if (!url) {
    setStatus("Indiquez l'adresse du flux", "warn");
    return;
  }
  stopSource();
  stopButton.disabled = false;
  whenOpen(() => sendControl("stream", { url }));
});

document.querySelector("#demo").addEventListener("click", () => {
  stopSource();
  stopButton.disabled = false;
  whenOpen(() => sendControl("demo"));
});

stopButton.addEventListener("click", () => {
  stopSource();
  sendControl("stop");
  setStatus("En attente", "idle");
});

document.querySelector("#relearn").addEventListener("click", () => sendControl("relearn"));
document.querySelector("#clear").addEventListener("click", () => {
  polygons.length = 0;
  draft = [];
  drawing = false;
  closeZone.disabled = true;
  overlay.style.pointerEvents = "none";
  sendControl("clear");
  redraw();
});

otsu.addEventListener("change", () => sendControl());
median.addEventListener("change", () => sendControl());
threshold.addEventListener("input", () => {
  thresholdValue.textContent = threshold.value;
  sendControl();
});

document.querySelector("#draw").addEventListener("click", () => {
  if (!annotated.naturalWidth) {
    setStatus("Lancez une source d'abord", "warn");
    return;
  }
  drawing = true;
  draft = [];
  closeZone.disabled = true;
  overlay.hidden = false;
  overlay.style.pointerEvents = "auto";
  placeholder.hidden = true;
  redraw();
});

closeZone.addEventListener("click", () => finishZone());

overlay.addEventListener("pointerdown", (event) => {
  if (!drawing) return;
  const point = imagePoint(event);
  if (!point) return;
  if (draft.length >= 3) {
    const [x, y] = draft[0];
    if (Math.abs(x - point[0]) + Math.abs(y - point[1]) < 20) {
      finishZone();
      return;
    }
  }
  draft.push(point);
  closeZone.disabled = draft.length < 3;
  redraw();
});

window.addEventListener("resize", redraw);

function finishZone() {
  if (draft.length < 3) return;
  polygons.push(draft);
  draft = [];
  drawing = false;
  closeZone.disabled = true;
  overlay.style.pointerEvents = "none";
  sendControl();
  redraw();
}

fetch("/api/me")
  .then((response) => (response.ok ? response.json() : null))
  .then((user) => {
    if (!user) {
      window.location.href = "/login";
      return;
    }
    document.querySelector("#user-name").textContent = user.name;
  });

document.querySelector("#logout").addEventListener("click", async () => {
  stopSource();
  await fetch("/api/logout", { method: "POST" });
  window.location.href = "/login";
});

connect();
