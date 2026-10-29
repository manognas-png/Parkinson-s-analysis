// Command-line wrapper so Python can call the engine through Node.js:
//   node demo/run_engine.js <trial file> <body weight in kg, or "none"> [template.json]
// Prints the engine's result as JSON.
const fs = require("fs");
const engine = require("./engine.js");

const [file, weight, templatePath] = process.argv.slice(2);
const trial = engine.parseTrialText(fs.readFileSync(file, "utf8"));
const templates = templatePath ? engine.loadTemplates(JSON.parse(fs.readFileSync(templatePath, "utf8"))) : null;
const result = engine.analyzeTrial(trial, {
  bodyWeightKg: weight === "none" ? null : Number(weight),
  templates,
});
process.stdout.write(JSON.stringify(result));
