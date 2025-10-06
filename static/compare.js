let current = null, imgL = new Image(), imgR = new Image();

const leftCanvas = document.getElementById("canvasLeft"),
      rightCanvas = document.getElementById("canvasRight");
const ctxL = leftCanvas.getContext("2d"),
      ctxR = rightCanvas.getContext("2d");

const timeSlider   = document.getElementById("timeSlider"),
      timestamp    = document.getElementById("timestamp");
const audioPlayer  = document.getElementById("audioPlayer"),
      uploadForm   = document.getElementById("uploadForm");
const fileInput    = document.getElementById("fileInput"),
      uploadBtn    = document.getElementById("uploadBtn"),
      resetBtn     = document.getElementById("resetBtn");
const layoutSelect = document.getElementById("layoutSelect");
const viewerContainer = document.getElementById("viewerContainer");
const loader       = document.getElementById("loaderOverlay");
const mainLayout   = document.getElementById("mainLayout");

let state = { scale: 1, tx: 0, ty: 0 };

function resize() {
  for (const [c, ctx] of [[leftCanvas, ctxL],[rightCanvas, ctxR]]) {
    const r = c.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
    c.width = r.width * dpr; c.height = r.height * dpr;
    ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.scale(dpr, dpr);
  }
}

function draw(ctx, canvas, img, type) {
  if (!img.complete || !current) return;
  const cw = canvas.clientWidth, ch = canvas.clientHeight;
  ctx.save(); ctx.clearRect(0, 0, cw, ch); ctx.translate(state.tx, state.ty); ctx.scale(state.scale, state.scale);

  let imgW, imgH;
  if (type === "clean") { imgH = ch; imgW = img.width * (imgH / img.height); }
  else { imgH = ch; const cleanW = imgR.width * (imgH / imgR.height); imgW = cleanW; }
  ctx.drawImage(img, 0, 0, imgW, imgH);

  const pct = timeSlider.value / 100, markerX = pct * imgW;
  ctx.beginPath(); ctx.moveTo(markerX, 0); ctx.lineTo(markerX, imgH);
  ctx.lineWidth = 1 / state.scale; ctx.strokeStyle = "lime"; ctx.stroke();
  ctx.restore();
}

function drawAll() { draw(ctxL, leftCanvas, imgL, "unclean"); draw(ctxR, rightCanvas, imgR, "clean"); updateTime(); }
function updateTime() {
  if (!current) return;
  const t = (timeSlider.value / 100) * current.duration;
  const mm = Math.floor(t / 60), ss = (t % 60).toFixed(2).padStart(5, "0");
  timestamp.textContent = `${mm}:${ss}`;
}

/* ===== Loader helpers (same behavior as main page) ===== */
function showLoader(label = "Analyzing…") {
  if (!loader) return;
  const textNode = loader.querySelector(".loader-text");
  if (textNode) textNode.textContent = label;
  loader.classList.remove("hidden");
  if (mainLayout) mainLayout.setAttribute("aria-busy", "true");
  uploadBtn?.setAttribute("disabled", "true");
  resetBtn?.setAttribute("disabled", "true");
}
function hideLoader() {
  if (!loader) return;
  loader.classList.add("hidden");
  if (mainLayout) mainLayout.setAttribute("aria-busy", "false");
  uploadBtn?.removeAttribute("disabled");
  resetBtn?.removeAttribute("disabled");
}

/* ===== Upload flow with loader shown until both images load ===== */
uploadForm.addEventListener("submit", (e) => {
  e.preventDefault();
  if (!fileInput.files.length) { alert("Choose a file"); return; }

  const fd = new FormData(); fd.append("file", fileInput.files[0]);

  showLoader("Analyzing…");
  fetch("/compare/upload", { method: "POST", body: fd })
    .then(r => r.json())
    .then(s => {
      if (s.error) throw new Error(s.error);

      current = s; state = { scale: 1, tx: 0, ty: 0 }; timeSlider.value = 0;
      audioPlayer.src = s.audio_url;

      // Only hide loader after BOTH images finish loading
      let pending = 2;
      function done() { pending -= 1; if (pending <= 0) { resize(); drawAll(); hideLoader(); } }

      imgL = new Image();
      imgL.onload = done;
      imgL.onerror = () => { alert("Failed to load original spectrogram image."); hideLoader(); };
      imgL.src = s.og_url;

      imgR = new Image();
      imgR.onload = done;
      imgR.onerror = () => { alert("Failed to load clean spectrogram image."); hideLoader(); };
      imgR.src = s.clean_url;
    })
    .catch(err => {
      alert(err.message || "Upload failed");
      hideLoader();
    });
});

timeSlider.addEventListener("input", () => {
  if (audioPlayer.duration) audioPlayer.currentTime = (timeSlider.value / 100) * audioPlayer.duration;
  drawAll();
});
audioPlayer.addEventListener("timeupdate", () => {
  if (!audioPlayer.duration) return;
  timeSlider.value = (audioPlayer.currentTime / audioPlayer.duration) * 100;
  drawAll();
});

layoutSelect.addEventListener("change", () => {
  viewerContainer.classList.toggle("row", layoutSelect.value === "side");
  viewerContainer.classList.toggle("column", layoutSelect.value !== "side");
  resize(); drawAll();
});

resetBtn.addEventListener("click", () => { state = { scale: 1, tx: 0, ty: 0 }; drawAll(); });
window.addEventListener("resize", () => { resize(); drawAll(); });

// drag/pan + zoom
let dragging = false, lastX = 0, lastY = 0;
[leftCanvas, rightCanvas].forEach((c) => {
  c.addEventListener("mousedown", (e) => { dragging = true; lastX = e.clientX; lastY = e.clientY; });
  c.addEventListener("wheel", (e) => {
    e.preventDefault();
    const rect = c.getBoundingClientRect(), mx = e.clientX - rect.left, my = e.clientY - rect.top;
    const zoom = Math.exp(-e.deltaY * 0.0012);
    const newScale = Math.min(Math.max(state.scale * zoom, 0.5), 20);
    state.tx = mx - (mx - state.tx) * (newScale / state.scale);
    state.ty = my - (my - state.ty) * (newScale / state.scale);
    state.scale = newScale; drawAll();
  }, { passive: false });
});
window.addEventListener("mousemove", (e) => {
  if (!dragging) return;
  const dx = e.clientX - lastX, dy = e.clientY - lastY;
  lastX = e.clientX; lastY = e.clientY;
  state.tx += dx; state.ty += dy; drawAll();
});
window.addEventListener("mouseup", () => { dragging = false; });

resize(); drawAll();

/* Optional: show loader during very first page load of UI assets
   If you want a splash, uncomment below:
   showLoader("Ready"); setTimeout(hideLoader, 400);
*/
