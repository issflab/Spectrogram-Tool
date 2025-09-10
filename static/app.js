// static/app.js

let gState = {
    fileId: null,
    lastResult: null,
    audioUrl: null,
    method: "raw",
    params: {},
    cmap: "Inferno",
    showF0: true,
    showFormants: true,
    showContours: true,
  };
  
  const el = (id) => document.getElementById(id);
  const loader = document.getElementById("loaderOverlay");
  const analyzeBtn = document.getElementById("analyzeBtn");
  const reAnalyzeBtn = document.getElementById("reAnalyze");
  const mainLayout = document.getElementById("mainLayout");
  
  /* ================= Loader helpers ================= */
  function showLoader() {
    loader.classList.remove("hidden");
    mainLayout.setAttribute("aria-busy", "true");
    analyzeBtn?.setAttribute("disabled", "true");
    reAnalyzeBtn?.setAttribute("disabled", "true");
    analyzeBtn && (analyzeBtn.dataset._label = analyzeBtn.textContent);
    analyzeBtn && (analyzeBtn.textContent = "Analyzing…");
  }
  function hideLoader() {
    loader.classList.add("hidden");
    mainLayout.setAttribute("aria-busy", "false");
    analyzeBtn?.removeAttribute("disabled");
    if (gState.fileId) reAnalyzeBtn?.removeAttribute("disabled");
    if (analyzeBtn?.dataset?._label) analyzeBtn.textContent = analyzeBtn.dataset._label;
  }
  
  /* ================= Tabs ================= */
  document.querySelectorAll(".tab").forEach(btn => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".tab").forEach(x => x.classList.remove("active"));
      document.querySelectorAll(".tabpane").forEach(x => x.classList.remove("active"));
      btn.classList.add("active");
      el(`tab-${btn.dataset.tab}`).classList.add("active");
    });
  });
  
  /* ================= Controls visibility ================= */
  el("method").addEventListener("change", (e) => {
    const m = e.target.value;
    gState.method = m;
    el("rawParams").style.display = (m === "raw") ? "" : "none";
    el("rawToggles").style.display = (m === "raw") ? "" : "none";
    el("imgParams").style.display = (m === "image") ? "" : "none";
    el("imgToggles").style.display = (m === "image") ? "" : "none";
  });
  
  /* ================= Upload + analyze ================= */
  el("uploadForm").addEventListener("submit", async (e) => {
    e.preventDefault();
    const file = el("fileInput").files[0];
    if (!file) return;
  
    const method = el("method").value;
    const denoise = document.getElementById("denoiseChk").checked;
  
    const params = (method === "raw") ? {
      max_formants: parseInt(el("maxFormants").value || 20),
      dur_limit_sec: parseFloat(el("durLimit").value || 5.0),
      denoise
    } : {
      percentile: parseFloat(el("percentile").value || 90),
      max_freq_range: parseFloat(el("maxFreqRange").value || 500),
      min_len: parseInt(el("minLen").value || 20),
      denoise
    };
  
    const fd = new FormData();
    fd.append("file", file);
    fd.append("method", method);
    fd.append("params", JSON.stringify(params));
  
    try {
      showLoader();
      const res = await fetch("/analyze", { method: "POST", body: fd });
      const data = await res.json();
      if (data.error) throw new Error(data.error);
  
      gState.fileId = data.file_id;
      gState.lastResult = data;
      gState.method = method;
      gState.params = params;
      el("reAnalyze").disabled = false;
  
      if (gState.audioUrl) URL.revokeObjectURL(gState.audioUrl);
      gState.audioUrl = URL.createObjectURL(file);
      el("player").src = gState.audioUrl;
  
      document.querySelector(".tag").textContent = data?.denoised ? "denoised" : "enhanced";
  
      renderAll();
    } catch (err) {
      alert(err.message || "Analyze failed");
    } finally {
      hideLoader();
    }
  });
  
  /* ================= Re-analyze ================= */
  el("reAnalyze").addEventListener("click", async () => {
    if (!gState.fileId) return;
    const method = el("method").value;
    const denoise = document.getElementById("denoiseChk").checked;
  
    const params = (method === "raw") ? {
      max_formants: parseInt(el("maxFormants").value || 20),
      dur_limit_sec: parseFloat(el("durLimit").value || 5.0),
      denoise
    } : {
      percentile: parseFloat(el("percentile").value || 90),
      max_freq_range: parseFloat(el("maxFreqRange").value || 500),
      min_len: parseInt(el("minLen").value || 20),
      denoise
    };
  
    try {
      showLoader();
      const res = await fetch("/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ file_id: gState.fileId, method, params })
      });
      const data = await res.json();
      if (data.error) throw new Error(data.error);
  
      gState.lastResult = data;
      gState.method = method;
      gState.params = params;
  
      document.querySelector(".tag").textContent = data?.denoised ? "denoised" : "enhanced";
  
      renderAll();
    } catch (err) {
      alert(err.message || "Re-analyze failed");
    } finally {
      hideLoader();
    }
  });
  
  /* ================= Options toggles ================= */
  el("cmap").addEventListener("change", e => { gState.cmap = e.target.value; renderSpectrogram(); });
  
  const toggleF0El = document.getElementById("toggleF0");
  const toggleFormantsEl = document.getElementById("toggleFormants");
  if (toggleF0El) toggleF0El.addEventListener("change", e => { gState.showF0 = e.target.checked; renderSpectrogram(); });
  if (toggleFormantsEl) toggleFormantsEl.addEventListener("change", e => { gState.showFormants = e.target.checked; renderSpectrogram(); });
  
  const toggleContoursEl = document.getElementById("toggleContours");
  if (toggleContoursEl) toggleContoursEl.addEventListener("change", e => { gState.showContours = e.target.checked; renderSpectrogram(); });
  
  /* ================= Downloads ================= */
  el("downloadPng").addEventListener("click", async () => {
    Plotly.toImage(el("specPlot"), { format: "png", height: 700, width: 1200 }).then((dataUrl) => {
      const a = document.createElement("a");
      a.href = dataUrl;
      a.download = "spectrogram.png";
      a.click();
    });
  });
  el("dlDist").addEventListener("click", () => downloadCsv(gState?.lastResult?.tables?.distance_csv, "distance_matrix.csv"));
  el("dlTime").addEventListener("click", () => downloadCsv(gState?.lastResult?.tables?.time_csv, "time_ranges.csv"));
  function downloadCsv(text, filename) {
    if (!text) return;
    fetch("/download/csv", {
      method: "POST",
      headers: {"Content-Type":"application/json"},
      body: JSON.stringify({ csv: text, filename })
    }).then(res => {
      if (!res.ok) throw new Error("Download failed");
      return res.blob();
    }).then(blob => {
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url; a.download = filename; a.click();
      URL.revokeObjectURL(url);
    }).catch(err => alert(err.message));
  }
  
  /* ================= CSV → Array helper ================= */
  function parseCSV(csvText) {
    if (!csvText) return [];
    const lines = csvText.trim().split(/\r?\n/);
    return lines.map(line => line.split(",").map(s => s.trim()));
  }
  
  /* ================= Build HTML table with partition lines ================= */
  function buildTable(containerId, rows, opts = {}) {
    const { headerRow = true, indexCol = true, partitionEvery = 5, numeric = true } = opts;
    const container = document.getElementById(containerId);
    container.innerHTML = "";
    if (!rows || !rows.length) { container.textContent = "—"; return; }
  
    const headers = headerRow ? rows[0] : Array.from({length: rows[0].length - (indexCol?1:0)}, (_,i)=>`C${i+1}`);
    const bodyRows = headerRow ? rows.slice(1) : rows;
  
    const table = document.createElement("table");
    table.className = "datatable";
  
    // THEAD
    const thead = document.createElement("thead");
    const trh = document.createElement("tr");
    if (indexCol) {
      const thBlank = document.createElement("th");
      thBlank.textContent = "";
      trh.appendChild(thBlank);
    }
    headers.forEach((h, j) => {
      const th = document.createElement("th");
      th.textContent = h;
      if (partitionEvery > 0 && (j+1) % partitionEvery === 0) th.classList.add("vcut");
      trh.appendChild(th);
    });
    thead.appendChild(trh);
    table.appendChild(thead);
  
    // TBODY
    const tbody = document.createElement("tbody");
    bodyRows.forEach((row, i) => {
      const tr = document.createElement("tr");
      if (partitionEvery > 0 && (i+1) % partitionEvery === 0) tr.classList.add("hcut");
  
      if (indexCol) {
        const th = document.createElement("th");
        th.textContent = row[0];
        tr.appendChild(th);
      }
  
      const start = indexCol ? 1 : 0;
      for (let j = start; j < row.length; j++) {
        const td = document.createElement("td");
        const val = row[j];
        if (numeric && val !== "" && !isNaN(Number(val))) {
          td.textContent = Number(val).toLocaleString(undefined, {maximumFractionDigits: 2});
        } else {
          td.textContent = val;
        }
        const colIndex = indexCol ? (j) : (j + 1);
        if (partitionEvery > 0 && ((colIndex) % partitionEvery === 0)) {
          td.classList.add("vcut");
        }
        tr.appendChild(td);
      }
      tbody.appendChild(tr);
    });
    table.appendChild(tbody);
  
    container.appendChild(table);
  }
  
  /* ================= Renderers ================= */
  function renderAll() {
    renderSpectrogram();
    renderBark();
    renderTables();
  }
  
  function renderTables() {
    // Distance Matrix table (with partition cuts)
    const distCsv = gState?.lastResult?.tables?.distance_csv || "";
    const distRows = parseCSV(distCsv);
    if (!distRows.length) {
      document.getElementById("distTable").textContent = "—";
    } else {
      buildTable("distTable", distRows, {
        headerRow: true,
        indexCol: true,
        partitionEvery: 5,   // change to 4/6/etc if you prefer
        numeric: true
      });
    }
  
    // Time Ranges table (simple table)
    const timeCsv = gState?.lastResult?.tables?.time_csv || "";
    const timeRows = parseCSV(timeCsv);
    if (!timeRows.length) {
      document.getElementById("timeTable").textContent = "—";
    } else {
      buildTable("timeTable", timeRows, {
        headerRow: true,
        indexCol: false,
        partitionEvery: 0,
        numeric: false
      });
    }
  }
  
  function renderBark() {
    const bark = gState?.lastResult?.bark;
    if (!bark) return;
    const x = [];
    for (let i=0;i<bark.edges.length-1;i++) {
      const left = bark.edges[i], right = bark.edges[i+1];
      x.push(`${left}–${right} Hz`);
    }
    const y = bark.energy;
  
    Plotly.newPlot("barkPlot", [{
      x, y, type: "bar", hovertemplate: "%{x}<br>Avg dB: %{y:.2f}<extra></extra>"
    }], {
      margin: {t: 20, r: 10, b: 60, l: 50},
      xaxis: { tickangle: -30 },
      yaxis: { title: "Avg dB" },
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
      font: { color: "#e6edf3" }
    }, {displaylogo:false, modeBarButtonsToRemove:["select2d","lasso2d"]});
  }
  
  function renderSpectrogram() {
    const res = gState.lastResult;
    if (!res) return;
  
    const z = res.S_db;
    const x = res.time;
    const y = res.freq;
  
    const heat = {
      z: z, x: x, y: y,
      type: "heatmap",
      colorscale: gState.cmap,
      zmin: -80, zmax: 0,
      hovertemplate: "t=%{x:.3f}s<br>f=%{y:.0f}Hz<br>dB=%{z:.1f}<extra></extra>"
    };
  
    const traces = [heat];
  
    if (res.mode === "Raw Audio") {
      if (gState.showF0 && res.f0) {
        traces.push({
          x: res.f0.time, y: res.f0.freq, mode: "lines", name: "F0",
          line: { width: 1.5 },
          hovertemplate: "F0<br>t=%{x:.3f}s<br>f=%{y:.0f}Hz<extra></extra>", yaxis: "y"
        });
      }
      if (gState.showFormants && res.formants) {
        res.formants.forEach((fr, idx) => {
          traces.push({
            x: fr.time, y: fr.freq, mode: "lines", name: `F${idx+1}`,
            line: { width: 1 },
            hovertemplate: `F${idx+1}<br>t=%{x:.3f}s<br>f=%{y:.0f}Hz<extra></extra>`, yaxis: "y"
          });
        });
      }
    } else {
      if (gState.showContours && res.contours) {
        res.contours.forEach(c => {
          traces.push({
            x: c.time, y: c.freq, mode: "lines", name: c.name,
            line: { width: 1 },
            hovertemplate: `${c.name}<br>t=%{x:.3f}s<br>f=%{y:.0f}Hz<extra></extra>`, yaxis: "y"
          });
        });
      }
    }
  
    const playheadX = getPlayheadTime();
    const shapes = [];
    if (playheadX != null && y.length) {
      shapes.push({
        type: "line", x0: playheadX, x1: playheadX, y0: y[0], y1: y[y.length-1],
        line: { color: "#6ae3ff", width: 1.5 }
      });
    }
  
    Plotly.newPlot("specPlot", traces, {
      margin: { t: 20, r: 15, b: 40, l: 50 },
      xaxis: { title: "Time (s)" },
      yaxis: { title: "Frequency (Hz)" },
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
      font: { color: "#e6edf3" },
      legend: { bgcolor: "rgba(0,0,0,0)" },
      shapes
    }, { responsive: true, displaylogo:false });
  
    el("specPlot").on('plotly_hover', (ev) => {
      const p = ev.points?.[0];
      if (!p) return;
      updateReadout(p.x, p.y, p.z);
    });
  
    el("specPlot").on('plotly_click', (ev) => {
      const p = ev.points?.[0];
      if (!p) return;
      const player = el("player");
      player.currentTime = p.x;
      updatePlayhead();
    });
  }
  
  /* ================= Readout & playhead ================= */
  function updateReadout(t, f, db) {
    el("ro-time").textContent = (t == null ? "–" : t.toFixed(3));
    el("ro-freq").textContent = (f == null ? "–" : f.toFixed(0));
    el("ro-level").textContent = (db == null ? "–" : db.toFixed(1));
  }
  function getPlayheadTime() {
    const player = el("player");
    if (!player || isNaN(player.currentTime)) return null;
    return player.currentTime;
  }
  function updatePlayhead() {
    const t = getPlayheadTime();
    const gd = el("specPlot");
    if (!gd || t == null) return;
  
    const y = gState?.lastResult?.freq || [];
    const shapes = (gd.layout.shapes || []).filter(s => s._id !== "playhead");
    if (y.length) {
      shapes.push({
        type: "line", x0: t, x1: t, y0: y[0], y1: y[y.length-1],
        line: { color: "#6ae3ff", width: 1.5 }, _id: "playhead"
      });
      Plotly.relayout(gd, { shapes });
    }
  }
  el("player").addEventListener("timeupdate", updatePlayhead);
  