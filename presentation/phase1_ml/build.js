// Phase 1 (ML) deck: Model Application + Calibration & Compute. Data from data.json (results/).
const pptxgen = require("pptxgenjs");
const path = require("path");
const { applyTheme } = require(process.env.SKILL + "/scripts/apply_theme.js");
const D = require("./data.json");

const THEME = {
  name: "HIGGS ML",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "1B2333", lt1: "FFFFFF", dk2: "16324F", lt2: "EEF2F7",
    accent1: "EB6834", accent2: "2A78D6", accent3: "008300", accent4: "6250D6",
    accent5: "E34948", accent6: "8A94A6", hlink: "2A78D6", folHlink: "6250D6",
  },
};
const HEX = THEME.colors;
const OUT = process.argv[2] || "deck.pptx";

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.333 x 7.5
pres.title = "Model Application and Calibration - HIGGS Phase 1";
pres.author = "Ananya Soni";
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
const C = pres.SchemeColor;
const W = 13.333, M = 0.6;

// ---------- layouts ----------
pres.defineSlideMaster({
  title: "TITLE_DARK",
  background: { color: C.text2 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: M, y: 2.0, w: W - 2 * M, h: 1.4,
        fontSize: 44, bold: true, color: C.background1, fontFace: THEME.headFontFace, valign: "bottom", margin: 0 },
        text: "Title" } },
    { placeholder: { options: { name: "body", type: "body", x: M, y: 3.55, w: W - 2 * M, h: 1.0,
        fontSize: 20, color: C.background2, valign: "top", margin: 0 }, text: "Subtitle" } },
  ],
});
pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: C.background1 },
  margin: [0.5, M, 0.6, M],
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: M, y: 0.35, w: W - 2 * M, h: 0.85,
        fontSize: 36, bold: true, color: C.text2, fontFace: THEME.headFontFace, valign: "middle", margin: 0 },
        text: "Title" } },
    { text: { text: "HIGGS signal detection  ·  Phase 1: Machine Learning",
        options: { x: M, y: 7.0, w: 8, h: 0.3, fontSize: 10, color: C.accent6, margin: 0 } } },
  ],
  slideNumber: { x: W - M - 0.6, y: 7.0, w: 0.6, h: 0.3, fontSize: 10, color: C.accent6, align: "right" },
});

// ---------- helpers ----------
const fmt3 = (v) => v.toFixed(3);
const card = (slide, name, x, y, w, h, big, bigColor, label, sub) => {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.12,
    fill: { color: C.background2 }, line: { color: C.background2 }, objectName: name + " card" });
  slide.addText(big, { x: x + 0.25, y: y + 0.12, w: w - 0.5, h: h * 0.48, fontSize: 32, bold: true,
    color: bigColor, fontFace: THEME.headFontFace, valign: "bottom", margin: 0, isTextBox: true,
    objectName: name + " value" });
  slide.addText([
    { text: label, options: { bold: true, color: C.text1, breakLine: !!sub } },
    ...(sub ? [{ text: sub, options: { color: C.text1 } }] : []),
  ], { x: x + 0.25, y: y + h * 0.5 + 0.08, w: w - 0.5, h: h * 0.5 - 0.2, fontSize: 14, valign: "top",
    margin: 0, isTextBox: true, objectName: name + " label" });
};
const axisText = { catAxisLabelColor: HEX.dk1, valAxisLabelColor: HEX.dk1, catAxisLabelFontSize: 12,
  valAxisLabelFontSize: 11, catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt",
  dataLabelFontFace: "+mn-lt", legendFontFace: "+mn-lt", titleFontFace: "+mn-lt" };
const caption = (slide, text, x, y, w, name) => slide.addText(text, { x, y, w, h: 0.35, fontSize: 11,
  italic: true, color: C.accent6, margin: 0, isTextBox: true, objectName: name });

// ---------- 1. title ----------
pres.addSection({ title: "Introduction" });
let s = pres.addSlide({ masterName: "TITLE_DARK", sectionTitle: "Introduction" });
s.addText("Model Application & Calibration", { placeholder: "title" });
s.addText([
  { text: "HIGGS signal detection  ·  Phase 1: Machine Learning", options: { breakLine: true } },
  { text: "Ananya Soni", options: { bold: true, color: C.accent1 } },
], { placeholder: "body" });
const facts = [["2", "models"], ["5", "feature sets"], ["5", "seeds"], ["500k", "fixed test events"]];
facts.forEach(([n, l], i) => {
  const x = M + i * 3.05;
  s.addText([{ text: n, options: { fontSize: 30, bold: true, color: C.accent1, breakLine: true } },
             { text: l, options: { fontSize: 14, color: C.background2 } }],
    { x, y: 5.3, w: 2.8, h: 1.1, fontFace: THEME.bodyFontFace, margin: 0, valign: "top", isTextBox: true,
      objectName: `fact ${i + 1}` });
});
s.addNotes("I'll cover how we applied the two machine-learning models, how we know the differences are real, and whether their probabilities can be trusted, and at what compute cost. Everything uses the same 250k-event budget, five random seeds, and the same 500k held-out test events.");

// ---------- 2. results ----------
pres.addSection({ title: "Model application" });
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Model application" });
s.addText("XGBoost wins on every feature set", { placeholder: "title" });
s.addChart(pres.charts.BAR, [
  { name: "Logistic regression", labels: D.fs_label, values: D.logreg.roc_auc },
  { name: "XGBoost (tuned)", labels: D.fs_label, values: D.xgboost.roc_auc },
], {
  x: M, y: 1.4, w: 7.9, h: 5.0, barDir: "bar", barGrouping: "clustered", barGapWidthPct: 60,
  chartColors: [HEX.accent2, HEX.accent1], catAxisOrientation: "maxMin",
  valAxisMinVal: 0.5, valAxisMaxVal: 0.9, valAxisMajorUnit: 0.1, valAxisLabelFormatCode: "0.0",
  showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.000", dataLabelFontSize: 11,
  dataLabelColor: HEX.dk1, showLegend: true, legendPos: "b", legendFontSize: 12,
  valGridLine: { color: "D9DEE6", size: 0.75 }, catGridLine: { style: "none" },
  showTitle: true, title: "Test ROC AUC, mean of 5 seeds (axis starts at 0.5 = random guessing)",
  titleFontSize: 12, titleColor: HEX.dk1, ...axisText, objectName: "AUC by feature set chart",
});
card(s, "best", 8.9, 1.4, 3.83, 1.55, "0.838", C.accent1, "Best model", "XGBoost on geo + high");
card(s, "geo", 8.9, 3.12, 3.83, 1.55, "0.801 > 0.787", C.accent1, "Our geometry beats",
  "the 7 expert masses");
card(s, "gap", 8.9, 4.84, 3.83, 1.55, "+0.14 to +0.17", C.accent1, "XGBoost over LogReg", "on every feature set");
s.addNotes(`We applied two models: logistic regression, our linear, interpretable baseline, and XGBoost, a tuned gradient-boosted tree ensemble, the standard in particle physics. Both were trained on five feature sets at the same 250k-event budget, scored on the same 500k test events, and repeated over five seeds.

Three things stand out. First, XGBoost beats logistic regression on every feature set, by 0.14 to 0.17 AUC: a straight-line model simply can't represent this problem. Second, our geometric features reach 0.801, higher than the physicists' 7 expert masses at 0.787, without using any masses at all. Third, combining our geometry with the masses gives the best model, 0.838, beating the original paper's feature set at 0.823.

If asked why logistic regression fails: its log-odds are linear in the inputs, but the signal lives in angle differences and masses, which are non-linear functions of the raw measurements.`);

// ---------- 3. rigour ----------
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Model application" });
s.addText("Every gain holds in all 5 seeds", { placeholder: "title" });
const compColors = D.comps.map((c) => (c.label.includes("LogReg") ? HEX.accent2 : HEX.accent1));
s.addChart(pres.charts.BAR, [{ name: "AUC gain", labels: D.comps.map((c) => c.label), values: D.comps.map((c) => c.diff) }], {
  x: M, y: 1.4, w: 6.9, h: 4.6, barDir: "bar", catAxisOrientation: "maxMin", barGapWidthPct: 55,
  chartColors: compColors, valAxisMinVal: 0, valAxisMaxVal: 0.08, valAxisMajorUnit: 0.02,
  valAxisLabelFormatCode: "0.00", showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "+0.000",
  dataLabelFontSize: 12, dataLabelColor: HEX.dk1, showLegend: false,
  valGridLine: { color: "D9DEE6", size: 0.75 }, catGridLine: { style: "none" },
  showTitle: true, title: "Test AUC gain (A − B), mean of 5 seeds", titleFontSize: 12, titleColor: HEX.dk1,
  ...axisText, objectName: "AUC gains chart",
});
const maxCi = Math.max(...D.comps.map((c) => c.ci)).toFixed(4);
caption(s, `Paired DeLong test on the same 500k test events: every row significant (p < 0.05) in 5/5 seeds; 95% CI ± ≤ ${maxCi}.`,
  M, 6.1, 6.9, "DeLong caption");
s.addText("How the hyperparameters were chosen", { x: 7.9, y: 1.4, w: 4.83, h: 0.4, fontSize: 18, bold: true,
  color: C.text2, fontFace: THEME.headFontFace, margin: 0, isTextBox: true, objectName: "tuning heading" });
const hdr = (t) => ({ text: t, options: { bold: true, color: C.background1, fill: { color: C.text2 } } });
const rows = [[hdr("Features"), hdr("Depth"), hdr("η"), hdr("mcw"), hdr("CV AUC")]];
D.tuning.forEach((t, i) => rows.push([t.fs.replace("_", " + "), String(t.max_depth), String(t.learning_rate),
  String(t.min_child_weight), t.cv.toFixed(3)].map((v) => ({ text: v,
  options: { fill: { color: i % 2 ? HEX.lt1 : HEX.lt2 }, color: C.text1 } }))));
s.addTable(rows, { x: 7.9, y: 1.95, w: 4.83, colW: [1.45, 0.75, 0.75, 0.75, 1.13], fontSize: 14,
  align: "center", valign: "middle", rowH: 0.42, border: { type: "solid", pt: 0.5, color: "D9DEE6" },
  objectName: "tuning table" });
s.addText([
  { text: "Stratified 3-fold CV on the training split only", options: { bullet: true, breakLine: true } },
  { text: "12 random configurations per feature set", options: { bullet: true, breakLine: true } },
  { text: "Deep trees win: the signal is a high-order interaction", options: { bullet: true } },
], { x: 7.9, y: 4.65, w: 4.83, h: 1.4, fontSize: 14, color: C.text1, paraSpaceAfter: 6, margin: 0,
  valign: "top", isTextBox: true, objectName: "tuning bullets" });
s.addNotes(`How do we know these differences are real? All models are scored on the same test events, so their AUCs are correlated, and a naive comparison would be wrong. We use DeLong's paired test, which accounts for that correlation. Every improvement here, for example +0.064 from our geometric features, is significant in all five seeds, and seed-to-seed variation is under 0.001.

Hyperparameters were chosen fairly: stratified 3-fold cross-validation on training data only, so the test set was never touched. A random search over 12 configurations picked deep trees, depth 8 to 10, learning rate 0.05 and min_child_weight 20. Deep trees make sense because the signal is a high-order interaction of the raw inputs.`);

// ---------- 4. calibration ----------
pres.addSection({ title: "Calibration & compute" });
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Calibration & compute" });
s.addText("XGBoost's probabilities can be trusted", { placeholder: "title" });
const cats = D.rel.centers.map((c) => c.toFixed(2));
s.addChart(pres.charts.LINE, [
  { name: "Perfect calibration", labels: cats, values: D.rel.centers },
  { name: "Logistic regression", labels: cats, values: D.rel.logreg },
  { name: "XGBoost (tuned)", labels: cats, values: D.rel.xgboost },
], {
  x: M, y: 1.4, w: 7.4, h: 5.0, chartColors: [HEX.accent6, HEX.accent2, HEX.accent1], lineSize: 3,
  lineDataSymbol: "circle", lineDataSymbolSize: 8, valAxisMinVal: 0, valAxisMaxVal: 1, valAxisMajorUnit: 0.2,
  valAxisLabelFormatCode: "0.0", showValAxisTitle: true, valAxisTitle: "Observed signal fraction",
  valAxisTitleFontSize: 12, valAxisTitleColor: HEX.dk1, showCatAxisTitle: true,
  catAxisTitle: "Predicted probability (bin centre)", catAxisTitleFontSize: 12, catAxisTitleColor: HEX.dk1,
  showLegend: true, legendPos: "b", legendFontSize: 12, valGridLine: { color: "D9DEE6", size: 0.75 },
  catGridLine: { style: "none" }, showTitle: true,
  title: "Reliability on the 7 expert masses (seed 0, 500k test events, before correction)",
  titleFontSize: 12, titleColor: HEX.dk1, ...axisText, objectName: "reliability chart",
});
const lrHigh = D.fs.indexOf("high");
card(s, "xgb ece", 8.4, 1.4, 4.33, 1.55, D.xgboost.ece[lrHigh].toFixed(3), C.accent1, "XGBoost ECE (expert masses)",
  "On the diagonal, no fix needed");
card(s, "lr ece", 8.4, 3.12, 4.33, 1.55, `${D.logreg.ece[lrHigh].toFixed(3)} → ${D.logreg.ece_platt[lrHigh].toFixed(3)}`,
  C.accent2, "LogReg ECE: raw → Platt", "Platt only stretches; can't fix a bent curve");
card(s, "iso", 8.4, 4.84, 4.33, 1.55, D.logreg.ece_isotonic[lrHigh].toFixed(3), C.accent3,
  "LogReg after isotonic", "Any monotone shape; fit on the cal split");
s.addNotes(`A good AUC means the model ranks events well. Calibration asks something different: when the model says 70% signal, is it right 70% of the time? This plot compares predicted probability with the observed signal rate; a perfect model sits on the grey diagonal.

XGBoost sits right on it: expected calibration error 0.006 here, and 0.004 to 0.006 on every feature set, with no correction needed. Logistic regression bends away, ECE 0.048 on the expert masses, because its linear assumption is wrong, not because of noise. We fit two standard fixes on a separate calibration split. Platt scaling only stretches the curve, so it can't fix the shape and leaves ECE at 0.048. Isotonic regression can follow any monotone shape and brings it down to 0.006.

ECE: split predictions into 15 equal-width bins; in each, compare the average predicted probability with the actual signal fraction; ECE is the size-weighted average of those gaps.`);

// ---------- 5. compute ----------
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Calibration & compute" });
s.addText("Seconds of compute buy 0.14+ AUC", { placeholder: "title" });
const crow = [[hdr("Features"), hdr("LogReg time"), hdr("XGBoost time"), hdr("XGBoost AUC"), hdr("Gain")]];
D.fs.forEach((f, i) => {
  const isBest = f === "geo_high", isValue = f === "all";
  const fill = { color: isBest || isValue ? "FCE3D7" : (i % 2 ? HEX.lt1 : HEX.lt2) };
  const t = (v) => (v < 1 ? v.toFixed(2) : v.toFixed(1)) + " s";
  crow.push([D.fs_label[i], t(D.logreg.train_seconds[i]), t(D.xgboost.train_seconds[i]),
    fmt3(D.xgboost.roc_auc[i]), "+" + fmt3(D.xgboost.roc_auc[i] - D.logreg.roc_auc[i])]
    .map((v, j) => ({ text: v, options: { fill, color: C.text1, bold: (isBest || isValue) && j === 3,
      align: j === 0 ? "left" : "center" } })));
});
s.addTable(crow, { x: M, y: 1.45, w: 7.6, colW: [2.6, 1.25, 1.35, 1.35, 1.05], fontSize: 14, valign: "middle",
  rowH: 0.55, border: { type: "solid", pt: 0.5, color: "D9DEE6" }, objectName: "compute table" });
caption(s, "Training time on 200k events (Apple M3, XGBoost on all CPU cores); mean of 5 seeds. Highlighted: best value and best overall.",
  M, 5.0, 7.6, "compute caption");
card(s, "lr time", 8.6, 1.45, 4.13, 1.5, "< 0.2 s", C.accent2, "Logistic regression", "any feature set");
card(s, "xgb time", 8.6, 3.1, 4.13, 1.5, "1.5–11.6 s", C.accent1, "Tuned XGBoost", "deeper trees, more features");
card(s, "value", 8.6, 4.75, 4.13, 1.5, "4.8 s → 0.823", C.accent1, "Best value", "XGBoost on the paper's features");
s.addText("Best overall: XGBoost on geo + high, 0.838 in about 10 s", { x: M, y: 5.6, w: 7.6, h: 0.5,
  fontSize: 16, bold: true, color: C.text2, margin: 0, isTextBox: true, objectName: "best overall line" });
s.addNotes(`On compute: logistic regression trains in under a fifth of a second. XGBoost takes 1.5 to 12 seconds on 200k events. Those few seconds buy 0.14 to 0.17 AUC. The best value is XGBoost on the paper's features, 4.8 seconds for 0.823; the best overall is geometry plus masses, 0.838 in about 10 seconds.

All timings are at the 250k budget on the same machine; we don't compare times from the 1M+ runs, which ran under memory pressure.

Transition: So XGBoost is accurate, calibrated and cheap. Next, [teammate] will explain what the model actually learned.`);

// ---------- 6. backup ----------
pres.addSection({ title: "Backup" });
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Backup" });
s.addText("Backup: calibration before and after", { placeholder: "title" });
const bl = [...D.fs.map((f) => "LogReg · " + f.replace("_", " + ")), ...D.fs.map((f) => "XGBoost · " + f.replace("_", " + "))];
const cat = (k) => [...D.logreg[k], ...D.xgboost[k]];
s.addChart(pres.charts.BAR, [
  { name: "Raw", labels: bl, values: cat("ece") },
  { name: "Platt", labels: bl, values: cat("ece_platt") },
  { name: "Isotonic", labels: bl, values: cat("ece_isotonic") },
], {
  x: M, y: 1.35, w: 8.2, h: 5.45, barDir: "bar", barGrouping: "clustered", catAxisOrientation: "maxMin",
  barGapWidthPct: 45, chartColors: [HEX.accent5, HEX.accent4, HEX.accent3], valAxisMinVal: 0, valAxisMaxVal: 0.05,
  valAxisMajorUnit: 0.01, valAxisLabelFormatCode: "0.00", showLegend: true, legendPos: "b", legendFontSize: 12,
  valGridLine: { color: "D9DEE6", size: 0.75 }, catGridLine: { style: "none" }, showTitle: true,
  title: "Expected calibration error (lower is better), mean of 5 seeds", titleFontSize: 12, titleColor: HEX.dk1,
  ...axisText, catAxisLabelFontSize: 11, objectName: "ECE before after chart",
});
s.addText([
  { text: "Calibrators are fit on the separate cal split (25k events), never on val or test", options: { bullet: true, breakLine: true } },
  { text: "Platt: σ(a·logit p + b). Monotone, AUC unchanged, can only stretch the curve", options: { bullet: true, breakLine: true } },
  { text: "Isotonic: any monotone step function. Fixes LogReg's bent curve", options: { bullet: true, breakLine: true } },
  { text: "XGBoost is calibrated already: log-loss objective and early stopping on log loss", options: { bullet: true } },
], { x: 9.1, y: 1.5, w: 3.63, h: 5.0, fontSize: 14, color: C.text1, paraSpaceAfter: 10, margin: 0, valign: "top",
  isTextBox: true, objectName: "calibration notes" });
s.addNotes("Backup if asked about calibration methods. Platt cannot change the shape of the reliability curve because it is an affine map of the logit; isotonic can. XGBoost is calibrated already because it is trained and early-stopped on log loss, a proper scoring rule. Older findings that boosting is miscalibrated were for boosted stumps with exponential loss.");

(async () => {
  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("wrote " + path.resolve(OUT));
})();
