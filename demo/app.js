/*
 * Interactive part of the gait demo (plain JavaScript, no libraries).
 *
 * How it fits together:
 *   - GaitEngine (engine.js) does all the gait analysis.
 *   - window.TEMPLATE_JSON and window.DEMO_DATA hold the healthy template, group statistics,
 *     the three demo pairs and their 10-second recordings.
 *   - A "panel" is one person on screen: their recording, its analysis, and its drawings.
 *   - One animation loop moves a playback clock and redraws both panels about 60 times a second.
 *
 * The first part (DemoCore) is pure calculation, so it can be tested in Node.js without a browser.
 */
(function () {
  "use strict";

  const WINDOW_S = 10; // seconds animated per recording
  const TRACE_S = 4; // seconds visible in the scrolling trace

  // ===================== Pure calculations (no drawing) =====================

  /** Build a panel's data from a recording. rule "all" for the short demo clips, "steady" for full files. */
  function makePanel(trial, options) {
    const engine = options.engine;
    let windowTrial = trial;
    let fullSummary = null;
    if (options.rule === "steady") {
      // A full recording: analyse it all, then animate 10 s starting at the first steady left stride.
      const full = engine.analyzeTrial(trial, { bodyWeightKg: options.weightKg, templates: options.templates });
      fullSummary = full.summary;
      const first = full.strides.find((s) => s.version === "16" && s.foot === "L");
      const start = first ? first.heel_strike : trial.time[0];
      windowTrial = sliceTrial(trial, start, start + WINDOW_S, engine.COLUMNS);
    }
    const analysis = engine.analyzeTrial(windowTrial, {
      bodyWeightKg: options.weightKg, templates: options.templates, strideRule: "all",
    });
    const fs = engine.samplingRate(windowTrial.time);
    const duration = windowTrial.time[windowTrial.time.length - 1] - windowTrial.time[0];
    // Body weight in newtons: from the weight box if given, otherwise estimated from the insole
    // (mean force while exactly one foot is on the ground, the same rule as the deviation score).
    const bodyWeightN = options.weightKg ? options.weightKg * 9.81 : singleSupportEstimate(trial, engine);
    return {
      label: options.label, meta: options.meta, group: options.group, weightKg: options.weightKg,
      bodyWeightN, weightEstimated: !options.weightKg,
      trial: windowTrial, fs, duration, analysis, fullSummary,
      forceMax: percentile(allSensorValues(windowTrial, engine), 0.97),
    };
  }

  /** Mean summed force (all 16 sensors) while exactly one foot carries more than 20 N. */
  function singleSupportEstimate(trial, engine) {
    const left = engine.sumChannels(trial, engine.VERSIONS["16"].L);
    const right = engine.sumChannels(trial, engine.VERSIONS["16"].R);
    let total = 0;
    let count = 0;
    for (let i = 0; i < left.length; i++) {
      if ((left[i] > 20) !== (right[i] > 20)) { total += left[i] + right[i]; count++; }
    }
    return count ? total / count : 1;
  }

  /** Largest heel or forefoot force (in body weights) in a panel's clip, for a shared y-axis. */
  function traceMax(panel, version) {
    const fore = version === "16" ? ["L6", "L7", "L8"] : ["L7"];
    let max = 0;
    for (let i = 0; i < panel.trial.time.length; i++) {
      max = Math.max(max, panel.trial.L1[i], fore.reduce((a, s) => a + panel.trial[s][i], 0));
    }
    return max / panel.bodyWeightN;
  }

  /** Cut [start, end) seconds out of a recording and restart its clock at 0. */
  function sliceTrial(trial, start, end, columns) {
    const first = trial.time.findIndex((t) => t >= start);
    let last = trial.time.findIndex((t) => t >= end);
    if (last < 0) last = trial.time.length;
    const out = {};
    for (const c of columns) out[c] = trial[c].slice(first, last);
    const t0 = out.time[0];
    out.time = out.time.map((t) => t - t0);
    return out;
  }

  function allSensorValues(trial, engine) {
    const values = [];
    for (const name of Object.keys(engine.SENSOR_POSITIONS)) for (const v of trial[name]) values.push(v);
    return values;
  }

  function percentile(values, p) {
    const sorted = [...values].sort((a, b) => a - b);
    return sorted[Math.min(sorted.length - 1, Math.floor(p * sorted.length))] || 1;
  }

  /** Index of the sample at playback time t (seconds). */
  function sampleAt(panel, t) {
    return Math.min(panel.trial.time.length - 1, Math.max(0, Math.round(t * panel.fs)));
  }

  /** Most recent completed left-foot stride (end <= t) for a version, or null. */
  function latestStride(panel, version, t) {
    let latest = null;
    for (const s of panel.analysis.strides) {
      if (s.version === version && s.foot === "L" && s.heel_strike_end <= t) latest = s;
    }
    return latest;
  }

  /** Text for the four cards: median of all strides so far (main), latest stride (small), group medians. */
  function cardValues(panel, version, t, medians) {
    const done = panel.analysis.strides.filter((s) => s.version === version && s.foot === "L" && s.heel_strike_end <= t);
    const stride = done.length ? done[done.length - 1] : null;
    // Heel load in body weights: the engine already divided by the entered weight; otherwise divide
    // its newtons by the single-support estimate.
    const toBw = (metric, v) => (metric === "heel_load_early" && !panel.weightKg ? v / panel.bodyWeightN : v);
    const bwLabel = panel.weightKg ? "× BW" : "× BW (est.)";
    const fmt = {
      heel_load_early: (v) => `${v.toFixed(2)} ${bwLabel}`,
      cop_05_15: (v) => v.toFixed(2),
      cadence: (v) => `${Math.round(v)}`,
      template_deviation: (v) => (v * 1000).toFixed(2),
    };
    const names = {
      heel_load_early: "Heel load, early stance",
      cop_05_15: "Centre of pressure, early stance (0 heel – 1 toe)",
      cadence: "Cadence (steps / min)",
      template_deviation: "Deviation from healthy template (×10⁻³)",
    };
    return Object.keys(fmt).map((metric) => {
      const values = done.map((s) => s[metric]).filter((v) => v !== null && v !== undefined && !Number.isNaN(v))
        .map((v) => toBw(metric, v));
      const latest = stride ? stride[metric] : null;
      const hasLatest = latest !== null && latest !== undefined && !Number.isNaN(latest);
      return {
        metric, name: names[metric],
        value: values.length ? fmt[metric](median(values)) : "waiting for first step…",
        latest: hasLatest ? `latest stride: ${fmt[metric](toBw(metric, latest))} · ${values.length} strides so far` : "",
        reference: `group medians: healthy <b>${fmtRef(metric, medians[version].Control[metric])}</b> · ` +
          `PD <b>${fmtRef(metric, medians[version].PD[metric])}</b>`,
      };
    });

    function fmtRef(metric, v) {
      if (v === null || v === undefined || Number.isNaN(v)) return "—";
      if (metric === "heel_load_early") return v.toFixed(2);
      if (metric === "cadence") return `${Math.round(v)}`;
      if (metric === "template_deviation") return (v * 1000).toFixed(2);
      return v.toFixed(2);
    }
  }

  function median(values) {
    const sorted = [...values].sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
  }

  /** Mix two hex colours: f = 0 gives a, f = 1 gives b. */
  function mixColor(a, b, f) {
    const pa = parseInt(a.slice(1), 16);
    const pb = parseInt(b.slice(1), 16);
    const ch = (shift) => Math.round(((pa >> shift) & 255) * (1 - f) + ((pb >> shift) & 255) * f);
    return `rgb(${ch(16)}, ${ch(8)}, ${ch(0)})`;
  }

  /** Centre of pressure (x, y in insole units) of the given sensors at sample i, or null if unloaded. */
  function centreOfPressure(trial, sensors, positions, i) {
    let total = 0;
    let x = 0;
    let y = 0;
    for (const s of sensors) {
      const f = Math.max(trial[s][i], 0);
      total += f;
      x += f * positions[s][0];
      y += f * positions[s][1];
    }
    return total > 20 ? { x: x / total, y: y / total } : null;
  }

  const DemoCore = { makePanel, sliceTrial, sampleAt, latestStride, cardValues, mixColor, centreOfPressure, percentile,
    singleSupportEstimate, traceMax };
  if (typeof module !== "undefined" && module.exports) { module.exports = DemoCore; return; }

  // ===================== Page code (runs in the browser) =====================

  const engine = window.GaitEngine;
  const DATA = window.DEMO_DATA;
  const TEMPLATES = engine.loadTemplates(window.TEMPLATE_JSON);
  const POS = engine.SENSOR_POSITIONS;
  const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

  const state = { pair: 0, version: "16", playing: true, speed: 1, t: 0, panels: [null, null], loaded: null };
  const $ = (id) => document.getElementById(id);

  // ---- Building the two panels ----

  function demoPanel(subject, group) {
    const w = DATA.windows[subject];
    const meta = demographicsLine(subject);
    return makePanel(w.trial, { engine, templates: TEMPLATES, weightKg: w.weight_kg, rule: "all",
      label: subject, group, meta });
  }

  function demographicsLine(subject) {
    const pair = DATA.pairs.find((p) => p.control === subject || p.pd === subject);
    if (!pair) return "";
    const isPd = pair.pd === subject;
    const age = isPd ? pair.pd_age : pair.control_age;
    const speed = isPd ? pair.pd_speed : pair.control_speed;
    const hy = isPd ? ` · Hoehn & Yahr ${pair.pd_hoehn_yahr}` : "";
    return `age ${age} · walking speed ${speed.toFixed(2)} m/s${hy} · Ga study`;
  }

  function showPair(index) {
    state.pair = index;
    const pair = DATA.pairs[index];
    state.panels[0] = demoPanel(pair.control, "Control");
    state.panels[1] = state.loaded || demoPanel(pair.pd, "PD");
    state.t = 0;
    buildPanelDom(0);
    buildPanelDom(1);
  }

  // ---- DOM for one panel: title, footprint SVG, trace canvas, cards ----

  function buildPanelDom(k) {
    const panel = state.panels[k];
    const root = $(k === 0 ? "panel-left" : "panel-right");
    const colour = panel.group === "PD" ? "var(--pd)" : panel.group === "Control" ? "var(--control)" : "var(--ink-2)";
    const tag = panel.group === "PD" ? "Parkinson’s" : panel.group === "Control" ? "Healthy" : "Loaded file";
    root.innerHTML = `
      <div class="person-title"><span class="tag" style="background:${colour}">${tag}</span>
        <h2 style="margin:0">${panel.label}</h2><span class="meta">${panel.meta || ""}</span></div>
      ${footprintSvg(k)}
      <canvas class="trace" id="trace-${k}"></canvas>
      <div class="cards" id="cards-${k}"></div>`;
  }

  function footprintSvg(k) {
    // Insole coordinates: x -1000..1000, y -1000 (heel) .. 1100 (toe); SVG y runs downward, so y is flipped.
    const outline = (cx) => `<path d="M ${cx - 230} 900 C ${cx - 330} 500, ${cx - 300} -350, ${cx - 210} -760
      C ${cx - 150} -1000, ${cx + 150} -1000, ${cx + 210} -760 C ${cx + 300} -350, ${cx + 330} 500, ${cx + 230} 900
      C ${cx + 140} 1080, ${cx - 140} 1080, ${cx - 230} 900 Z" transform="scale(1,-1)" fill="var(--insole)"
      stroke="var(--line)" stroke-width="8"/>`;
    const circles = Object.entries(POS).map(([name, [x, y]]) =>
      `<circle id="s-${k}-${name}" cx="${x}" cy="${-y}" r="95" stroke="var(--line)" stroke-width="6"/>
       <text x="${x}" y="${-y + 30}" font-size="70" text-anchor="middle" fill="var(--ink-3)"
       pointer-events="none">${name.slice(1)}</text>`).join("");
    const dots = ["L", "R"].map((f) => `<circle id="cop-${k}-${f}" r="55" fill="var(--pd)" stroke="#fff"
      stroke-width="14" visibility="hidden"/>`).join("");
    return `<svg class="feet" viewBox="-1000 -1150 2000 2300" role="img"
      aria-label="Footprints with 16 pressure sensors">
      ${outline(-500)}${outline(500)}${circles}${dots}
      <text x="-500" y="1130" font-size="80" text-anchor="middle" fill="var(--ink-2)">left foot</text>
      <text x="500" y="1130" font-size="80" text-anchor="middle" fill="var(--ink-2)">right foot</text></svg>`;
  }

  // ---- Drawing every frame ----

  function drawFootprint(k, panel, i) {
    const sensors = engine.VERSIONS[state.version];
    const active = new Set([...sensors.L, ...sensors.R]);
    const low = css("--ramp-low");
    const high = css("--ramp-high");
    for (const name of Object.keys(POS)) {
      const el = document.getElementById(`s-${k}-${name}`);
      const f = Math.max(0, Math.min(1, panel.trial[name][i] / panel.forceMax));
      el.setAttribute("fill", mixColor(low, high, f));
      el.setAttribute("opacity", active.has(name) ? "1" : "0.15");
    }
    for (const foot of ["L", "R"]) {
      const dot = document.getElementById(`cop-${k}-${foot}`);
      const cop = centreOfPressure(panel.trial, sensors[foot], POS, i);
      dot.setAttribute("visibility", cop ? "visible" : "hidden");
      if (cop) { dot.setAttribute("cx", cop.x); dot.setAttribute("cy", -cop.y); }
      dot.setAttribute("fill", panel.group === "Control" ? css("--control") : css("--pd"));
    }
  }

  function prepareCanvas(canvas) {
    const ratio = window.devicePixelRatio || 1;
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    if (canvas.width !== Math.round(w * ratio)) { canvas.width = Math.round(w * ratio); canvas.height = Math.round(h * ratio); }
    const ctx = canvas.getContext("2d");
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    ctx.clearRect(0, 0, w, h);
    return { ctx, w, h };
  }

  function drawTrace(k, panel, t) {
    const canvas = document.getElementById(`trace-${k}`);
    if (!canvas) return;
    const { ctx, w, h } = prepareCanvas(canvas);
    const fore = state.version === "16" ? ["L6", "L7", "L8"] : ["L7"];
    const trial = panel.trial;
    const t0 = t - TRACE_S;
    const pad = { l: 40, r: 8, t: 18, b: 18 };
    const x = (time) => pad.l + ((time - t0) / TRACE_S) * (w - pad.l - pad.r);
    // Same y-axis for both people, in body weights (forces divided by each person's weight)
    const yMax = Math.max(...state.panels.map((p) => traceMax(p, state.version))) * 1.05;
    const bw = panel.bodyWeightN;
    const y = (v) => h - pad.b - (v / bw / yMax) * (h - pad.t - pad.b);

    ctx.strokeStyle = css("--line");
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(pad.l, h - pad.b); ctx.lineTo(w - pad.r, h - pad.b); ctx.stroke();
    ctx.fillStyle = css("--ink-3");
    ctx.font = "11px system-ui, sans-serif";
    ctx.fillText(`${yMax.toFixed(2)}`, 2, pad.t + 4);
    ctx.fillText("0", 26, h - pad.b);
    ctx.fillText(panel.weightEstimated ? "× body weight (estimated)" : "× body weight", pad.l + 4, pad.t - 5);

    // heel strikes of the left foot (dashed)
    ctx.setLineDash([3, 3]);
    ctx.strokeStyle = css("--ink-3");
    for (const s of panel.analysis.strides) {
      if (s.version !== state.version || s.foot !== "L") continue;
      for (const hs of [s.heel_strike, s.heel_strike_end]) {
        if (hs > t0 && hs <= t) { ctx.beginPath(); ctx.moveTo(x(hs), pad.t); ctx.lineTo(x(hs), h - pad.b); ctx.stroke(); }
      }
    }
    ctx.setLineDash([]);

    const series = [["--heel", (i) => trial.L1[i]], ["--fore", (i) => fore.reduce((a, s) => a + trial[s][i], 0)]];
    for (const [colour, value] of series) {
      ctx.strokeStyle = css(colour);
      ctx.lineWidth = 2;
      ctx.beginPath();
      let started = false;
      for (let i = 0; i < trial.time.length; i++) {
        const time = trial.time[i];
        if (time < t0 || time > t) continue;
        if (!started) { ctx.moveTo(x(time), y(value(i))); started = true; } else ctx.lineTo(x(time), y(value(i)));
      }
      ctx.stroke();
    }
    // Legend, right-aligned using measured text widths so nothing is cut off
    const foreLabel = state.version === "16" ? "forefoot (sensors 6–8)" : "forefoot (sensor 7)";
    const heelLabel = "heel (sensor 1)";
    const gap = 14;
    const foreWidth = ctx.measureText(foreLabel).width;
    const heelWidth = ctx.measureText(heelLabel).width;
    ctx.fillStyle = css("--fore");
    ctx.fillText(foreLabel, w - pad.r - foreWidth, pad.t - 5);
    ctx.fillStyle = css("--heel");
    ctx.fillText(heelLabel, w - pad.r - foreWidth - gap - heelWidth, pad.t - 5);
  }

  function drawCards(k, panel, t) {
    const box = document.getElementById(`cards-${k}`);
    if (!box) return;
    const html = cardValues(panel, state.version, t, DATA.medians).map((c) =>
      `<div class="metric"><div class="name">${c.name}</div>
       <div class="value${c.latest ? "" : " waiting"}">${c.value}</div>
       <div class="latest">${c.latest}</div><div class="ref">${c.reference}</div></div>`).join("");
    if (box.dataset.last !== html) { box.innerHTML = html; box.dataset.last = html; } // only redraw when a stride changes
  }

  function drawGroupCurves() {
    const curves = DATA.curves[state.version];
    const specs = [["curve-heel", "heel", "heel load (× body weight)", [5, 15]],
                   ["curve-cop", "cop", "centre of pressure (0 heel – 1 toe)", [5, 15]]];
    for (const [id, key, label, shade] of specs) {
      const { ctx, w, h } = prepareCanvas($(id));
      const pad = { l: 40, r: 10, t: 12, b: 30 };
      let yMax = 0;
      for (const g of ["Control", "PD"]) {
        curves[g][`${key}_mean`].forEach((m, i) => { yMax = Math.max(yMax, m + curves[g][`${key}_sd`][i]); });
      }
      yMax = key === "cop" ? 1 : Math.ceil(yMax * 10) / 10;
      const x = (p) => pad.l + (p / 100) * (w - pad.l - pad.r);
      const y = (v) => h - pad.b - (v / yMax) * (h - pad.t - pad.b);
      ctx.fillStyle = css("--line");
      ctx.globalAlpha = 0.5;
      ctx.fillRect(x(shade[0]), pad.t, x(shade[1]) - x(shade[0]), h - pad.t - pad.b);
      ctx.globalAlpha = 1;
      ctx.strokeStyle = css("--line");
      ctx.beginPath(); ctx.moveTo(pad.l, h - pad.b); ctx.lineTo(w - pad.r, h - pad.b); ctx.moveTo(pad.l, pad.t);
      ctx.lineTo(pad.l, h - pad.b); ctx.stroke();
      ctx.fillStyle = css("--ink-3");
      ctx.font = "11px system-ui, sans-serif";
      [0, 25, 50, 75, 100].forEach((p) => ctx.fillText(`${p}%`, x(p) - 10, h - pad.b + 14));
      ctx.fillText(yMax.toFixed(1), 6, pad.t + 8);
      ctx.fillText("0", 26, h - pad.b);
      ctx.fillStyle = css("--ink-2");
      ctx.fillText(`${label}  ·  % of stance`, pad.l + 4, h - 2);
      for (const g of ["Control", "PD"]) {
        const mean = curves[g][`${key}_mean`];
        const sd = curves[g][`${key}_sd`];
        const colour = css(g === "Control" ? "--control" : "--pd");
        ctx.fillStyle = colour;
        ctx.globalAlpha = 0.15;
        ctx.beginPath();
        mean.forEach((m, i) => { const px = x(i); const py = y(m + sd[i]); if (i) ctx.lineTo(px, py); else ctx.moveTo(px, py); });
        for (let i = mean.length - 1; i >= 0; i--) ctx.lineTo(x(i), y(Math.max(0, mean[i] - sd[i])));
        ctx.closePath(); ctx.fill();
        ctx.globalAlpha = 1;
        ctx.strokeStyle = colour;
        ctx.lineWidth = 2;
        ctx.beginPath();
        mean.forEach((m, i) => (i ? ctx.lineTo(x(i), y(m)) : ctx.moveTo(x(i), y(m))));
        ctx.stroke();
      }
    }
    const n = DATA.curves[state.version];
    $("group-legend").innerHTML =
      `<span style="--c:var(--control)">Healthy (n = ${n.Control.n_subjects})</span>` +
      `<span style="--c:var(--pd)">Parkinson’s (n = ${n.PD.n_subjects})</span>` +
      `<span>${state.version} sensors · mean ± 1 SD</span>`;
  }

  // ---- Animation loop ----

  let lastFrame = null;
  function frame(timestamp) {
    if (lastFrame !== null && state.playing) {
      const duration = Math.min(...state.panels.map((p) => p.duration));
      state.t = (state.t + ((timestamp - lastFrame) / 1000) * state.speed) % duration;
    }
    lastFrame = timestamp;
    state.panels.forEach((panel, k) => {
      drawFootprint(k, panel, sampleAt(panel, state.t));
      drawTrace(k, panel, state.t);
      drawCards(k, panel, state.t);
    });
    requestAnimationFrame(frame);
  }

  // ---- Controls ----

  function setVersion(version) {
    state.version = version;
    $("v16").classList.toggle("active", version === "16");
    $("v6").classList.toggle("active", version === "6");
    state.panels.forEach((_, k) => { const box = $(`cards-${k}`); if (box) box.dataset.last = ""; });
    drawGroupCurves();
  }

  function loadFile(file) {
    const reader = new FileReader();
    reader.onload = () => {
      try {
        const trial = engine.parseTrialText(reader.result);
        if (trial.time.length < 500) throw new Error("too few samples (is this a PhysioNet trial file?)");
        const weight = parseFloat($("weight").value) || null;
        state.loaded = makePanel(trial, { engine, templates: TEMPLATES, weightKg: weight, rule: "steady",
          label: file.name, group: "Loaded",
          meta: weight ? `body weight ${weight} kg` : "no body weight: heel load in newtons" });
        const n = state.loaded.analysis.strides.filter((s) => s.version === "16" && s.foot === "L").length;
        $("file-status").textContent = `${file.name}: ${trial.time.length} samples, ${n} left strides in the 10-s window`;
        $("clear-file").hidden = false;
        showPair(state.pair);
      } catch (err) {
        $("file-status").textContent = `Could not analyse ${file.name}: ${err.message}`;
      }
    };
    reader.readAsText(file);
  }

  function init() {
    DATA.pairs.forEach((p, i) => {
      const option = document.createElement("option");
      option.value = i;
      option.textContent = `Pair ${i + 1}: ${p.control} (healthy) vs ${p.pd} (PD)`;
      $("pair").appendChild(option);
    });
    $("pair").addEventListener("change", (e) => showPair(Number(e.target.value)));
    $("play").addEventListener("click", () => {
      state.playing = !state.playing;
      $("play").textContent = state.playing ? "Pause" : "Play";
    });
    $("speed").addEventListener("input", (e) => {
      state.speed = Number(e.target.value);
      $("speed-value").textContent = `${state.speed}×`;
    });
    $("v16").addEventListener("click", () => setVersion("16"));
    $("v6").addEventListener("click", () => setVersion("6"));
    $("file").addEventListener("change", (e) => { if (e.target.files[0]) loadFile(e.target.files[0]); });
    $("weight").addEventListener("change", () => { if ($("file").files[0]) loadFile($("file").files[0]); });
    $("clear-file").addEventListener("click", () => {
      state.loaded = null;
      $("file").value = "";
      $("file-status").textContent = "";
      $("clear-file").hidden = true;
      showPair(state.pair);
    });
    window.addEventListener("resize", drawGroupCurves);
    showPair(0);
    drawGroupCurves();
    requestAnimationFrame(frame);
  }

  init();
})();
