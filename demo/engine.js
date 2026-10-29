/*
 * Gait analysis engine for the PD insole demo (plain JavaScript, no libraries).
 *
 * It reproduces the Python analysis (src/gait.py, src/cop.py, src/demo_metrics.py):
 *   1. find heel strikes and steady strides from the summed foot force,
 *   2. find each stance (heel strike -> foot off the ground) and stretch it to 0-100 %,
 *   3. compute per-stride metrics, in a 16-sensor and a 6-sensor version.
 * Centre-of-pressure traces are smoothed (Savitzky-Golay, 11 points, order 2, as in Step 11).
 * Template deviation divides each stride by an insole weight estimate (mean of left + right
 * total force over complete steady strides, times a calibration factor stored with the
 * template), so it needs no body weight; there is one template per foot.
 *
 * The same file works in a web page (it creates window.GaitEngine) and in Node.js
 * (require("./engine.js")), so it can be tested against Python before it goes in a browser.
 */
(function (root) {
  "use strict";

  // ---- Settings: identical to the Python code ----------------------------------
  // The sampling rate is read from the time column (100 Hz for PhysioNet files).
  const CONTACT_THRESHOLD_N = 20; // foot "on the ground" above this force
  const MIN_SWING_S = 0.1; // foot must be off the ground this long before a heel strike
  const STEADY_WINDOW = 7; // strides in the rolling median
  const STEADY_TOLERANCE = 0.1; // a stride is steady if within 10 % of that median
  const STEADY_MIN_RUN = 10; // at least 10 steady strides in a row form a segment
  const STEADY_TRIM = 2; // drop 2 strides at each end of a segment
  const STANCE_POINTS = 101; // stance stretched to 0, 1, ..., 100 %
  const STRIDE_POINTS = 100; // template strides
  const MIN_STANCE_S = 0.3;
  const MAX_STANCE_S = 2.0;
  const GRAVITY = 9.81;

  // Sensor positions inside the insole, from format.txt (x = side to side, y = heel -800 to toe +800)
  const SENSOR_POSITIONS = {
    L1: [-500, -800], L2: [-700, -400], L3: [-300, -400], L4: [-700, 0],
    L5: [-300, 0], L6: [-700, 400], L7: [-300, 400], L8: [-500, 800],
    R1: [500, -800], R2: [700, -400], R3: [300, -400], R4: [700, 0],
    R5: [300, 0], R6: [700, 400], R7: [300, 400], R8: [500, 800],
  };
  const COLUMNS = ["time",
    "L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8",
    "R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8",
    "L_total", "R_total"];
  const VERSIONS = {
    "16": { L: ["L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8"], R: ["R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8"] },
    "6": { L: ["L1", "L4", "L7"], R: ["R1", "R4", "R7"] },
  };
  const METRICS = ["stride_time", "cadence", "swing_time", "heel_load_early", "cop_05_15", "cop_25_35",
    "template_deviation"];

  // ---- Reading a PhysioNet trial file ------------------------------------------

  /** Turn the text of a trial file (19 numbers per line) into {time: [...], L1: [...], ...}. */
  function parseTrialText(text) {
    const trial = {};
    COLUMNS.forEach((name) => { trial[name] = []; });
    for (const line of text.split("\n")) {
      const parts = line.trim().split(/\s+/);
      if (parts.length < COLUMNS.length) continue; // skip blank or broken lines
      COLUMNS.forEach((name, k) => trial[name].push(parseFloat(parts[k])));
    }
    return trial;
  }

  // ---- Small numeric helpers (what numpy does for us in Python) ----------------

  /** Element-by-element sum of several equally long arrays. */
  function sumChannels(trial, names) {
    const n = trial[names[0]].length;
    const total = new Array(n).fill(0);
    for (const name of names) {
      const values = trial[name];
      for (let i = 0; i < n; i++) total[i] += values[i];
    }
    return total;
  }

  /** Median of an array (average of the two middle values for an even count). */
  function median(values) {
    const sorted = [...values].sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
  }

  function mean(values) {
    let total = 0;
    for (const v of values) total += v;
    return total / values.length;
  }

  /** Samples per second, from the time column (rounded: 100 for PhysioNet files). */
  function samplingRate(time) {
    return Math.round((time.length - 1) / (time[time.length - 1] - time[0]));
  }

  /** Savitzky-Golay smoothing, 11-point window, quadratic (scipy savgol_filter(x, 11, 2), mode "interp"). */
  function savgolSmooth(values) {
    const n = values.length;
    const half = 5;
    const out = new Array(n);
    // Inside: weighted average with the fixed weights (267 - 15 k^2) / 1287, k = -5..5
    for (let i = half; i < n - half; i++) {
      let total = 0;
      for (let k = -half; k <= half; k++) total += ((267 - 15 * k * k) / 1287) * values[i + k];
      out[i] = total;
    }
    // Ends: fit one parabola to the first (last) 11 points and read off the edge points
    const fitEdge = (offset, positions) => {
      const window = values.slice(offset, offset + 2 * half + 1);
      const coeffs = fitParabola(window);
      positions.forEach((t) => { out[offset + t] = coeffs[0] + coeffs[1] * t + coeffs[2] * t * t; });
    };
    fitEdge(0, [0, 1, 2, 3, 4]);
    fitEdge(n - 2 * half - 1, [6, 7, 8, 9, 10]);
    return out;
  }

  /** Least-squares parabola a + b t + c t^2 through points (t = 0, 1, ..., y.length - 1). */
  function fitParabola(y) {
    const s = [0, 0, 0, 0, 0]; // sums of t^0 .. t^4
    const r = [0, 0, 0]; // sums of y, t y, t^2 y
    y.forEach((v, t) => {
      for (let p = 0; p < 5; p++) s[p] += t ** p;
      for (let p = 0; p < 3; p++) r[p] += v * t ** p;
    });
    // Solve the 3 x 3 normal equations [[s0 s1 s2][s1 s2 s3][s2 s3 s4]] x = r (Cramer's rule)
    const m = [[s[0], s[1], s[2]], [s[1], s[2], s[3]], [s[2], s[3], s[4]]];
    const det = (a) => a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
      - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0]) + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
    const d = det(m);
    return [0, 1, 2].map((col) => det(m.map((row, i) => row.map((v, j) => (j === col ? r[i] : v)))) / d);
  }

  /** First index where sorted[index] >= value (numpy searchsorted, side="left"). */
  function searchSortedLeft(sorted, value) {
    let low = 0;
    let high = sorted.length;
    while (low < high) {
      const mid = (low + high) >> 1;
      if (sorted[mid] < value) low = mid + 1; else high = mid;
    }
    return low;
  }

  /** num evenly spaced values from start to stop (numpy linspace). */
  function linspace(start, stop, num, endpoint = true) {
    const step = (stop - start) / (endpoint ? num - 1 : num);
    const out = new Array(num);
    for (let k = 0; k < num; k++) out[k] = k * step + start;
    if (endpoint) out[num - 1] = stop;
    return out;
  }

  /** Straight-line interpolation of (xp, fp) at the points x (numpy interp; xp increasing). */
  function interp(x, xp, fp) {
    const out = new Array(x.length);
    const last = xp.length - 1;
    for (let k = 0; k < x.length; k++) {
      const v = x[k];
      if (v <= xp[0]) { out[k] = fp[0]; continue; }
      if (v >= xp[last]) { out[k] = fp[last]; continue; }
      let j = searchSortedLeft(xp, v);
      if (xp[j] === v) { out[k] = fp[j]; continue; }
      j -= 1; // now xp[j] < v < xp[j + 1]
      const slope = (fp[j + 1] - fp[j]) / (xp[j + 1] - xp[j]);
      out[k] = slope * (v - xp[j]) + fp[j];
    }
    return out;
  }

  // ---- Gait events (same rules as src/gait.py) ---------------------------------

  /** Heel strikes: force rises above 20 N after at least 0.1 s off the ground. Returns sample indices. */
  function detectHeelStrikes(force, fs) {
    const offGround = force.map((f) => f <= CONTACT_THRESHOLD_N);
    const nSwing = Math.trunc(MIN_SWING_S * fs);
    const strikes = [];
    for (let i = 1; i < force.length; i++) {
      if (!(offGround[i - 1] && !offGround[i])) continue; // not a rise
      if (i < nSwing) continue;
      let wasOff = true;
      for (let k = i - nSwing; k < i; k++) if (!offGround[k]) { wasOff = false; break; }
      if (wasOff) strikes.push(i);
    }
    return strikes;
  }

  /** Steady strides: within 10 % of the rolling median of 7, in runs of >= 10, minus 2 at each end. */
  function steadyStrideMask(strides) {
    const n = strides.length;
    const half = Math.floor(STEADY_WINDOW / 2);
    const steady = strides.map((s, i) => {
      const window = strides.slice(Math.max(0, i - half), Math.min(n, i + half + 1));
      const localMedian = median(window);
      return Math.abs(s - localMedian) <= STEADY_TOLERANCE * localMedian;
    });
    const mask = new Array(n).fill(false);
    let start = 0;
    while (start < n) {
      if (!steady[start]) { start++; continue; }
      let end = start;
      while (end < n && steady[end]) end++;
      if (end - start >= STEADY_MIN_RUN) {
        for (let k = start + STEADY_TRIM; k < end - STEADY_TRIM; k++) mask[k] = true;
      }
      start = end;
    }
    return mask;
  }

  /** [start time, end time] of every steady stride of one foot.
   *  rule "steady" (default): the steady-segment rule; "all": every complete stride
   *  (for short clips already cut from steady walking, too short for runs of 10). */
  function steadyStrideBounds(force, time, rule = "steady") {
    const strikes = detectHeelStrikes(force, samplingRate(time)).map((i) => time[i]);
    const strides = strikes.slice(1).map((t, k) => t - strikes[k]);
    const mask = rule === "all" ? strides.map(() => true) : steadyStrideMask(strides);
    const bounds = [];
    mask.forEach((keep, k) => { if (keep) bounds.push([strikes[k], strikes[k + 1]]); });
    return bounds;
  }

  /** [first, last) sample indices of the stance in a stride, or null (as in src/cop.py). */
  function stanceEnd(force, time, start, end, fs) {
    const first = searchSortedLeft(time, start);
    const strideEnd = searchSortedLeft(time, end);
    for (let i = first; i < strideEnd; i++) {
      if (force[i] <= CONTACT_THRESHOLD_N) {
        const duration = (i - first) / fs;
        return duration >= MIN_STANCE_S && duration <= MAX_STANCE_S ? [first, i] : null;
      }
    }
    return null;
  }

  /** Stretch samples first..last-1 onto 101 points (0-100 % of stance). */
  function resampleStance(values, first, last) {
    const xp = [];
    for (let i = first; i < last; i++) xp.push(i);
    return interp(linspace(first, last - 1, STANCE_POINTS), xp, values.slice(first, last));
  }

  /** Front-to-back centre of pressure per sample: 0 = heel row, 1 = toe (uses sensor y). */
  function apPosition(trial, sensors) {
    const n = trial.time.length;
    const q = new Array(n);
    for (let i = 0; i < n; i++) {
      let total = 0;
      let weighted = 0;
      for (const s of sensors) {
        const f = Math.max(trial[s][i], 0);
        total += f;
        weighted += f * SENSOR_POSITIONS[s][1];
      }
      q[i] = total > 1e-8 ? (weighted / total + 800) / 1600 : NaN;
    }
    return q;
  }

  /** One stride (start to end time) of each channel, resampled to 100 points (src/gait.resample_strides). */
  function resampleStride(trial, channels, start, end) {
    const time = trial.time;
    const first = searchSortedLeft(time, start);
    const last = searchSortedLeft(time, end);
    const grid = linspace(start, end, STRIDE_POINTS, false);
    const segmentTime = time.slice(first, last + 1);
    return channels.map((c) => interp(grid, segmentTime, trial[c].slice(first, last + 1)));
  }

  /** Insole weight estimate (newtons, uncalibrated): mean of left + right total force over all
   *  samples of the complete steady left strides. Over whole gait cycles the feet carry body weight. */
  function weightEstimateRaw(trial, version, rule = "steady") {
    const time = trial.time;
    const left = sumChannels(trial, VERSIONS[version].L);
    const right = sumChannels(trial, VERSIONS[version].R);
    let total = 0;
    let count = 0;
    for (const [start, end] of steadyStrideBounds(left, time, rule)) {
      const first = searchSortedLeft(time, start);
      const last = searchSortedLeft(time, end);
      for (let i = first; i < last; i++) { total += left[i] + right[i]; count++; }
    }
    return count ? total / count : NaN;
  }

  // ---- The analysis --------------------------------------------------------------

  /** Template JSON (from demo/build_template.py) -> {version: {L, R}, calibration: {version: factor}}. */
  function loadTemplates(json) {
    const out = { calibration: {} };
    for (const version of Object.keys(VERSIONS)) {
      out[version] = { L: json[version].L.values, R: json[version].R.values };
      out.calibration[version] = json.weight_estimate ? json.weight_estimate.calibration[version] : 1;
    }
    return out;
  }

  /**
   * Analyse one trial.
   * options.bodyWeightKg: divide forces by body weight (kg x 9.81) if given, else keep newtons.
   * options.templates: from loadTemplates(); needed for template deviation (no body weight needed).
   * options.strideRule: "steady" (default, full steady-segment rule) or "all" (every complete stride).
   * Returns {strides: [one object per stride], summary: {version: {metric: median}},
   *          weightEstimateN: {version: calibrated insole weight estimate in newtons}}.
   */
  function analyzeTrial(trial, options = {}) {
    const bodyWeightKg = options.bodyWeightKg || null;
    const scale = bodyWeightKg ? bodyWeightKg * GRAVITY : 1;
    const templates = options.templates || null;
    const rule = options.strideRule || "steady";
    const time = trial.time;
    const fs = samplingRate(time);
    const strides = [];
    const weightEstimateN = {};

    for (const version of Object.keys(VERSIONS)) {
      const channels = VERSIONS[version].L.concat(VERSIONS[version].R);
      const leftForce = sumChannels(trial, VERSIONS[version].L);
      const rightForce = sumChannels(trial, VERSIONS[version].R);
      const calibration = templates ? templates.calibration[version] : 1;
      const weightN = weightEstimateRaw(trial, version, rule) * calibration;
      weightEstimateN[version] = weightN;
      for (const foot of ["L", "R"]) {
        const sensors = VERSIONS[version][foot];
        const force = foot === "L" ? leftForce : rightForce;
        const q = apPosition(trial, sensors);
        const heel = trial[sensors[0]];
        for (const [start, end] of steadyStrideBounds(force, time, rule)) {
          const stance = stanceEnd(force, time, start, end, fs);
          if (stance === null) continue;
          const [first, last] = stance;
          const qStance = savgolSmooth(resampleStance(q, first, last));
          const heelStance = resampleStance(heel, first, last);
          let deviation = null;
          if (templates && Number.isFinite(weightN) && weightN > 0) {
            const stride = resampleStride(trial, channels, start, end);
            const template = templates[version][foot];
            let total = 0;
            for (let c = 0; c < stride.length; c++) {
              for (let t = 0; t < STRIDE_POINTS; t++) {
                const d = stride[c][t] / weightN - template[c][t];
                total += d * d;
              }
            }
            deviation = total / (stride.length * STRIDE_POINTS);
          }
          strides.push({
            version, foot, heel_strike: start, heel_strike_end: end, toe_off: time[last],
            stride_time: end - start,
            cadence: 120 / (end - start),
            swing_time: end - time[last],
            heel_load_early: mean(heelStance.slice(5, 16)) / scale,
            cop_05_15: mean(qStance.slice(5, 16)),
            cop_25_35: mean(qStance.slice(25, 36)),
            template_deviation: deviation,
          });
        }
      }
    }

    const summary = {};
    for (const version of Object.keys(VERSIONS)) {
      summary[version] = {};
      const rows = strides.filter((r) => r.version === version);
      for (const metric of METRICS) {
        const values = rows.map((r) => r[metric]).filter((v) => v !== null && !Number.isNaN(v));
        summary[version][metric] = values.length ? median(values) : null;
      }
    }
    return { strides, summary, weightEstimateN };
  }

  const GaitEngine = { parseTrialText, analyzeTrial, loadTemplates, steadyStrideBounds, detectHeelStrikes,
    weightEstimateRaw,
    sumChannels, apPosition, samplingRate, savgolSmooth, VERSIONS, METRICS, SENSOR_POSITIONS, COLUMNS };
  if (typeof module !== "undefined" && module.exports) module.exports = GaitEngine; // Node.js
  else root.GaitEngine = GaitEngine; // web page
})(typeof window !== "undefined" ? window : this);
