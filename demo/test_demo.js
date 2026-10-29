// Smoke test for the demo page's calculations (no browser needed):
//   module load node.js/22.14.0 && node demo/test_demo.js
// Checks that every embedded clip yields strides in both versions, that the cards fill in,
// and that a full PhysioNet file loads through the "Load recording" path.
const fs = require("fs");
const path = require("path");
const engine = require("./engine.js");
const core = require("./app.js");

const demo = JSON.parse(fs.readFileSync(path.join(__dirname, "demo_data.json"), "utf8"));
const templates = engine.loadTemplates(JSON.parse(fs.readFileSync(path.join(__dirname, "template.json"), "utf8")));
let failures = 0;
const check = (ok, message) => { console.log(`${ok ? "ok  " : "FAIL"} ${message}`); if (!ok) failures++; };

for (const [subject, w] of Object.entries(demo.windows)) {
  const panel = core.makePanel(w.trial, { engine, templates, weightKg: w.weight_kg, rule: "all", label: subject });
  for (const version of ["16", "6"]) {
    const left = panel.analysis.strides.filter((s) => s.version === version && s.foot === "L");
    const cards = core.cardValues(panel, version, panel.duration, demo.medians);
    check(left.length >= 5 && cards.every((c) => c.value !== "—"),
      `${subject} ${version}-sensor: ${panel.fs} Hz, ${panel.duration.toFixed(1)} s, ${left.length} left strides, ` +
      `cards: ${cards.map((c) => c.value).join(" | ")}`);
  }
}

const file = path.join(__dirname, "..", "data", "raw", "GaPt24_01.txt");
const trial = engine.parseTrialText(fs.readFileSync(file, "utf8"));
for (const weightKg of [null, 70]) {
  const panel = core.makePanel(trial, { engine, templates, weightKg, rule: "steady", label: "GaPt24_01.txt" });
  const left = panel.analysis.strides.filter((s) => s.version === "16" && s.foot === "L");
  const cards = core.cardValues(panel, "16", panel.duration, demo.medians);
  check(left.length >= 5 && Math.abs(panel.duration - 10) < 0.5,
    `loaded file, weight ${weightKg}: ${panel.fs} Hz, ${panel.duration.toFixed(2)} s window, ${left.length} left strides, ` +
    `cards: ${cards.map((c) => c.value).join(" | ")}`);
}

check(core.centreOfPressure({ L1: [100], L8: [100] }, ["L1", "L8"], engine.SENSOR_POSITIONS, 0).y === 0,
  "centre of pressure halfway between heel (y -800) and toe (y +800) sensors");
console.log(failures ? `${failures} FAILED` : "ALL PASSED");
process.exit(failures ? 1 : 0);
