// NegoPlay — academic presentation (Hebrew, RTL) in the logo's colours.
// Build:  node reports/build_deck.js   (run from the NegoPlay root)
const pptxgen = require("pptxgenjs");

const C = {
  navy:  "16324F",
  steel: "2C5F8A",
  ice:   "CADCFC",
  gray:  "6B7785",
  light: "F4F6F8",
  white: "FFFFFF",
  ink:   "1F2A33",
  green: "1F7A4D",
  red:   "C0392B",
};
const F = "Segoe UI";
const MONO = "Consolas";
const IMG = "docs/images";

const p = new pptxgen();
p.defineLayout({ name: "W", width: 13.333, height: 7.5 });
p.layout = "W";
p.author = "Anna Ben-Shushan";
p.title = "NegoPlay — From Bridge to Negotiation";
const W = 13.333, H = 7.5;

const sh = () => ({ type: "outer", color: "000000", blur: 9, offset: 3, angle: 135, opacity: 0.16 });

// ----- shared furniture -------------------------------------------------
function brandChip(s, x, y, size) {
  // navy rounded tile + small ice diamond (a card-suit motif echoing the logo)
  s.addShape(p.shapes.ROUNDED_RECTANGLE, { x, y, w: size, h: size, rectRadius: size * 0.22, fill: { color: C.navy } });
  const d = size * 0.36;
  s.addShape(p.shapes.RECTANGLE, { x: x + size / 2 - d / 2, y: y + size / 2 - d / 2, w: d, h: d, fill: { color: C.ice }, rotate: 45 });
}
function header(s, title, kicker) {
  brandChip(s, W - 1.05, 0.45, 0.5);
  if (kicker) s.addText(kicker, { x: 0.6, y: 0.5, w: W - 1.9, h: 0.32, align: "right", rtlMode: true,
    fontFace: F, fontSize: 13, color: C.steel, bold: true, charSpacing: 1, margin: 0 });
  s.addText(title, { x: 0.6, y: kicker ? 0.8 : 0.55, w: W - 1.9, h: 0.85, align: "right", rtlMode: true,
    fontFace: F, fontSize: 30, color: C.navy, bold: true, margin: 0 });
}
function footer(s, n, dark) {
  const col = dark ? C.ice : C.gray;
  s.addText([
    { text: "Nego", options: { color: dark ? C.white : C.navy, bold: true } },
    { text: "Play", options: { color: C.steel, bold: true } },
    { text: "   ·   ברידג׳ ← משא ומתן AI", options: { color: col } },
  ], { x: 0.6, y: H - 0.5, w: 7, h: 0.3, align: "left", fontFace: F, fontSize: 10, margin: 0 });
  s.addText(String(n).padStart(2, "0"), { x: W - 1.1, y: H - 0.5, w: 0.5, h: 0.3,
    align: "right", fontFace: F, fontSize: 10, color: col, margin: 0 });
}
function fit(maxW, maxH, ow, oh) { // contain within box, return {w,h}
  const r = ow / oh; let w = maxW, h = w / r; if (h > maxH) { h = maxH; w = h * r; } return { w, h };
}

// =======================================================================
// 1 — TITLE
// =======================================================================
let s = p.addSlide();
s.background = { color: C.navy };
// faint motif squares
s.addShape(p.shapes.ROUNDED_RECTANGLE, { x: -0.6, y: 5.7, w: 2.4, h: 2.4, rectRadius: 0.5, fill: { color: C.steel, transparency: 78 } });
s.addShape(p.shapes.ROUNDED_RECTANGLE, { x: 11.9, y: -0.7, w: 2.2, h: 2.2, rectRadius: 0.5, fill: { color: C.steel, transparency: 82 } });
// white card holding the logo
{
  const cw = 6.6, ch = 2.25, cx = (W - cw) / 2, cy = 1.45;
  s.addShape(p.shapes.ROUNDED_RECTANGLE, { x: cx, y: cy, w: cw, h: ch, rectRadius: 0.16, fill: { color: C.white }, shadow: sh() });
  const lg = fit(5.4, 1.55, 2267, 667);
  s.addImage({ path: `${IMG}/negoplay_logo_clean.png`, x: (W - lg.w) / 2, y: cy + (ch - lg.h) / 2, w: lg.w, h: lg.h });
}
s.addText("מטבלאות ברידג׳ לשולחן המשא ומתן", { x: 1, y: 4.05, w: W - 2, h: 0.6, align: "center", rtlMode: true,
  fontFace: F, fontSize: 24, color: C.white, bold: true });
s.addText("האם סגנון קבלת ההחלטות עובר בין תחומים שונים?", { x: 1, y: 4.7, w: W - 2, h: 0.5, align: "center", rtlMode: true,
  fontFace: F, fontSize: 16, color: C.ice });
s.addText("אנה בן־שושן   ·   קורס AI Development Expert", { x: 1, y: 6.4, w: W - 2, h: 0.4, align: "center", rtlMode: true,
  fontFace: F, fontSize: 13, color: C.ice, charSpacing: 1 });

// =======================================================================
// 2 — THE PROBLEM
// =======================================================================
s = p.addSlide(); s.background = { color: C.light };
header(s, "הבעיה: משא ומתן הוא קופסה שחורה", "המוטיבציה");
s.addText([
  { text: "משא ומתן עסקי הוא מהמיומנויות החשובות בתעשייה — אך כמעט בלתי אפשרי לחקור כמותית.", options: { bullet: true, breakLine: true } },
  { text: "התמלילים חסויים, התוצאות אינן מתוקננות, ואין דאטה ציבורית שמקשרת בין סגנון קבלת ההחלטות של אדם להתנהגותו במשא ומתן.", options: { bullet: true, breakLine: true } },
  { text: "אין דרך טבעית למדוד “סגנון” ולבדוק אם הוא עקבי בין מצבים.", options: { bullet: true } },
], { x: 6.7, y: 2.05, w: 6.0, h: 2.0, align: "right", rtlMode: true, fontFace: F, fontSize: 16, color: C.ink, paraSpaceAfter: 10 });

// highlight card
s.addShape(p.shapes.ROUNDED_RECTANGLE, { x: 6.7, y: 4.35, w: 6.0, h: 2.0, rectRadius: 0.12, fill: { color: C.navy }, shadow: sh() });
s.addText("הפתרון: ברידג׳ כמעבדה התנהגותית", { x: 6.9, y: 4.55, w: 5.6, h: 0.45, align: "right", rtlMode: true, fontFace: F, fontSize: 17, bold: true, color: C.white, margin: 0 });
s.addText("כל יד היא רצף של החלטות תחת אי־ודאות — כמה להתחייב, מתי לקחת סיכון, מתי להילחם — ולכל החלטה יש תוצאה מדידה. בדיוק מה שחסר במשא ומתן.",
  { x: 6.9, y: 5.05, w: 5.6, h: 1.2, align: "right", rtlMode: true, fontFace: F, fontSize: 14, color: C.ice, margin: 0 });

// left visual: bridge -> negotiation
function tile(x, label, sub, fillc, txtc) {
  s.addShape(p.shapes.ROUNDED_RECTANGLE, { x, y: 3.0, w: 2.5, h: 1.6, rectRadius: 0.14, fill: { color: fillc }, shadow: sh() });
  s.addText(label, { x, y: 3.25, w: 2.5, h: 0.5, align: "center", rtlMode: true, fontFace: F, fontSize: 18, bold: true, color: txtc });
  s.addText(sub, { x, y: 3.75, w: 2.5, h: 0.7, align: "center", rtlMode: true, fontFace: F, fontSize: 12, color: txtc });
}
tile(3.55, "ברידג׳", "המעבדה", C.navy, C.white);     // right tile
tile(0.6, "משא ומתן", "היעד", C.steel, C.white);      // left tile
s.addText("←", { x: 2.95, y: 3.15, w: 0.7, h: 1.0, align: "center", valign: "middle", fontFace: F, fontSize: 40, color: C.gray, margin: 0 });
footer(s, 2);

// =======================================================================
// 3 — PIPELINE
// =======================================================================
s = p.addSlide(); s.background = { color: C.white };
header(s, "איך זה עובד — צינור בן ארבעה שלבים", "ארכיטקטורה");
const steps = [
  { n: "1", t: "גילוי פרופילים", d: "‎149,208 ידיים ← 8 מאפיינים לכל שחקן (ML)" },
  { n: "2", t: "חילוץ מיומנויות", d: "ידיים אמיתיות ← 5–7 מיומנויות מובנות (LLM)" },
  { n: "3", t: "בניית סוכנים", d: "כל פרופיל ← סוכן LLM עם כרטיס דמות" },
  { n: "4", t: "סימולציה והצלבה", d: "ברידג׳ + משא ומתן ← מתאם בין־תחומי" },
];
const cw = 2.85, gap = 0.33, total = 4 * cw + 3 * gap, x0 = (W - total) / 2, cy = 2.6, chh = 2.5;
steps.forEach((st, i) => {
  // RTL: step 1 on the right
  const x = x0 + (3 - i) * (cw + gap);
  s.addShape(p.shapes.ROUNDED_RECTANGLE, { x, y: cy, w: cw, h: chh, rectRadius: 0.12,
    fill: { color: i % 2 ? C.steel : C.navy }, shadow: sh() });
  s.addShape(p.shapes.OVAL, { x: x + cw / 2 - 0.42, y: cy - 0.42, w: 0.84, h: 0.84, fill: { color: C.white }, shadow: sh() });
  s.addText(st.n, { x: x + cw / 2 - 0.42, y: cy - 0.42, w: 0.84, h: 0.84, align: "center", valign: "middle",
    fontFace: F, fontSize: 26, bold: true, color: i % 2 ? C.steel : C.navy, margin: 0 });
  s.addText(st.t, { x: x + 0.1, y: cy + 0.7, w: cw - 0.2, h: 0.6, align: "center", rtlMode: true, fontFace: F, fontSize: 16, bold: true, color: C.white, margin: 0 });
  s.addText(st.d, { x: x + 0.18, y: cy + 1.3, w: cw - 0.36, h: 1.0, align: "center", rtlMode: true, fontFace: F, fontSize: 12, color: C.ice, margin: 0 });
  if (i < 3) s.addText("←", { x: x - gap - 0.05, y: cy + chh / 2 - 0.35, w: gap + 0.1, h: 0.7, align: "center", valign: "middle", fontFace: F, fontSize: 26, color: C.gray, margin: 0 });
});
s.addText("ML קלאסי  ←  LLM, מחובר בעזרת קבצים שמורים — כל מספר ניתן לשחזור.",
  { x: 0.6, y: 5.7, w: W - 1.2, h: 0.5, align: "center", rtlMode: true, fontFace: F, fontSize: 14, italic: true, color: C.gray });
footer(s, 3);

// =======================================================================
// 4 — STAGE 1 + MATH (continuum finding)
// =======================================================================
s = p.addSlide(); s.background = { color: C.light };
header(s, "שלב 1 — גילוי פרופילים (והממצא המפתיע)", "ML · מתמטיקה");
{
  const d = fit(6.7, 4.0, 1438, 877);
  s.addShape(p.shapes.ROUNDED_RECTANGLE, { x: 0.55, y: 1.95, w: d.w + 0.3, h: d.h + 0.3, rectRadius: 0.1, fill: { color: C.white }, shadow: sh() });
  s.addImage({ path: `${IMG}/clustering_kmeans_fails.png`, x: 0.7, y: 2.1, w: d.w, h: d.h });
}
s.addText([
  { text: "תקנון מאפיינים (z־score) וסינון שונות (CV ≥ 0.10) — מסירים מאפיינים שאינם מבחינים.", options: { bullet: true, breakLine: true } },
  { text: "הממצא: שחקני עלית אינם נחלקים לקבוצות נפרדות — הם יוצרים רצף התנהגותי חלק (silhouette ≈ 0.24, הרבה מתחת ל־0.5).", options: { bullet: true, breakLine: true } },
  { text: "לכן: 5 פרופילים = 4 ארכיטיפים (קצוות הרצף) + Generalist baseline, עם מבחן בינומי חד־צדדי (p < 0.05).", options: { bullet: true } },
], { x: 7.7, y: 2.1, w: 5.0, h: 2.9, align: "right", rtlMode: true, fontFace: F, fontSize: 15, color: C.ink, paraSpaceAfter: 9 });
s.addShape(p.shapes.ROUNDED_RECTANGLE, { x: 7.7, y: 5.15, w: 5.0, h: 1.05, rectRadius: 0.1, fill: { color: C.navy } });
s.addText("z = (x − μ) / σ        CV = σ / |μ| ≥ 0.10",
  { x: 7.8, y: 5.4, w: 4.8, h: 0.55, align: "center", fontFace: MONO, fontSize: 16, color: C.white, margin: 0 });
footer(s, 4);

// =======================================================================
// 5 — THE FIVE PROFILES
// =======================================================================
s = p.addSlide(); s.background = { color: C.white };
header(s, "חמשת הפרופילים — קבוצת הבסיס האסטרטגית", "התוצר של שלב 1");
{
  const d = fit(5.4, 4.0, 998, 787);
  s.addImage({ path: `${IMG}/radar_profiles.png`, x: 0.7, y: 2.2, w: d.w, h: d.h });
}
const profs = [
  ["d62728", "Slam Hunter", "צייד סלאמים — אגרסיבי, מכריז חוזים גבוהים"],
  ["ff7f0e", "Fighter", "לוחם — מכפיל (double) יריבים בתדירות גבוהה"],
  ["2ca02c", "NT Specialist", "מומחה NT — מעדיף חוזי no־trump"],
  ["1f77b4", "Insurance Player", "שחקן ביטוח — שמרן, עוצר מתחת למשחק"],
  ["9aa0a6", "Generalist", "בסיס — מרכז ההתפלגות, נקודת ההשוואה"],
];
let py = 2.15;
profs.forEach((pr) => {
  s.addShape(p.shapes.OVAL, { x: 12.55, y: py + 0.12, w: 0.2, h: 0.2, fill: { color: pr[0] } });
  s.addText(pr[1], { x: 9.4, y: py, w: 3.05, h: 0.34, align: "right", rtlMode: true, fontFace: F, fontSize: 15, bold: true, color: C.navy, margin: 0 });
  s.addText(pr[2], { x: 6.4, y: py + 0.33, w: 6.05, h: 0.4, align: "right", rtlMode: true, fontFace: F, fontSize: 12.5, color: C.gray, margin: 0 });
  py += 0.88;
});
s.addText("כל שחקן אמיתי הוא תערובת של הארכיטיפים האלה (cardinal set).",
  { x: 6.4, y: 6.35, w: 6.05, h: 0.4, align: "right", rtlMode: true, fontFace: F, fontSize: 13, italic: true, color: C.steel, margin: 0 });
footer(s, 5);

// =======================================================================
// 6 — STAGES 2-3 (skills + agents)
// =======================================================================
s = p.addSlide(); s.background = { color: C.light };
header(s, "שלבים 2–3 — ממיומנויות לסוכנים", "LLM");
const rows = [
  ["חילוץ מיומנויות", "ידיים אמיתיות נשלחות ל־LLM בחבילות, שמחזיר 5–7 מיומנויות אופייניות כ־JSON מובנה."],
  ["כרטיס דמות", "כל פרופיל הופך לסוכן. הכרטיס מכיל רק מיומנויות שחולצו מברידג׳ — לעולם לא תארי אישיות."],
  ["הכלל המרכזי", "identity־disentangled: כל התנהגות חוצת־תחומים חייבת לנבוע מהמיומנויות, לא מהתווית."],
  ["טמפרטורה", "ברידג׳ ב־0 (עקביות), משא ומתן ב־0.7 (גיוון). כל קריאה נרשמת ומתומחרת."],
];
let ry = 2.05;
rows.forEach((r) => {
  s.addShape(p.shapes.ROUNDED_RECTANGLE, { x: 6.55, y: ry, w: 6.2, h: 1.0, rectRadius: 0.1, fill: { color: C.white }, shadow: sh() });
  s.addShape(p.shapes.ROUNDED_RECTANGLE, { x: 12.5, y: ry, w: 0.25, h: 1.0, rectRadius: 0.05, fill: { color: C.steel } });
  s.addText(r[0], { x: 6.75, y: ry + 0.1, w: 5.6, h: 0.35, align: "right", rtlMode: true, fontFace: F, fontSize: 15, bold: true, color: C.navy, margin: 0 });
  s.addText(r[1], { x: 6.75, y: ry + 0.43, w: 5.6, h: 0.52, align: "right", rtlMode: true, fontFace: F, fontSize: 12, color: C.ink, margin: 0 });
  ry += 1.15;
});
// example character-card snippet (left)
s.addShape(p.shapes.ROUNDED_RECTANGLE, { x: 0.7, y: 2.5, w: 5.4, h: 2.3, rectRadius: 0.12, fill: { color: C.navy }, shadow: sh() });
s.addText("דוגמה — כרטיס דמות (Fighter)", { x: 0.9, y: 2.7, w: 5.0, h: 0.4, align: "right", rtlMode: true, fontFace: F, fontSize: 14, bold: true, color: C.ice, margin: 0 });
s.addText('"doubles opponents at 6–8× the field rate"',
  { x: 0.9, y: 3.25, w: 5.0, h: 0.5, align: "left", fontFace: MONO, fontSize: 13, color: C.white, margin: 0 });
s.addText("← עובדה שחולצה ממכרזים אמיתיים, לא “היה אגרסיבי”. כך התנהגות חייבת לצמוח מהנתונים.",
  { x: 0.9, y: 3.9, w: 5.0, h: 0.8, align: "right", rtlMode: true, fontFace: F, fontSize: 12.5, color: C.ice, margin: 0 });
footer(s, 6);

// =======================================================================
// 7 — BRIDGE METRIC (ZI-C + double-dummy)
// =======================================================================
s = p.addSlide(); s.background = { color: C.white };
header(s, "מדד ברידג׳ אמין — סוכן אקראי (ZI־C) + Double־Dummy", "שלב 4 · אימות");
{
  const d = fit(10.2, 3.7, 1935, 887);
  s.addImage({ path: `${IMG}/metric_fix_monkey_dd.png`, x: (W - d.w) / 2, y: 1.75, w: d.w, h: d.h });
}
s.addText("מדד טוב חייב לדרג סוכן אקראי אחרון. המדד הראשוני נכשל — הקוף ניצח (0.54). אחרי ניקוד לפי האם החוזה באמת מצליח (double־dummy), הקוף צונח לאחרון (0.10) וכל פרופיל מנצח אותו.",
  { x: 1.2, y: 6.05, w: W - 2.4, h: 0.85, align: "center", rtlMode: true, fontFace: F, fontSize: 14, color: C.ink });
footer(s, 7);

// =======================================================================
// 8 — SKILL SPECTRUM
// =======================================================================
s = p.addSlide(); s.background = { color: C.light };
header(s, "ספקטרום הכישורים — מאקראי ועד משחק מושלם", "שלב 4 · נתונים אמיתיים");
{
  const d = fit(10.4, 3.8, 1815, 771);
  s.addShape(p.shapes.ROUNDED_RECTANGLE, { x: (W - d.w) / 2 - 0.12, y: 1.7, w: d.w + 0.24, h: d.h + 0.24, rectRadius: 0.1, fill: { color: C.white }, shadow: sh() });
  s.addImage({ path: `${IMG}/real_skill_spectrum.png`, x: (W - d.w) / 2, y: 1.82, w: d.w, h: d.h });
}
s.addText("על הלוחות האמיתיים: סוכן אקראי (ZI־C) ב־−7 IMP  ←  שחקני עלית ≈ 0  ←  משחק מושלם ‎+1.1 IMP. בין העלית, הפרופילים האגרסיביים (Slam Hunter, Fighter) מובילים.",
  { x: 1.2, y: 6.1, w: W - 2.4, h: 0.8, align: "center", rtlMode: true, fontFace: F, fontSize: 14, color: C.ink });
footer(s, 8);

// =======================================================================
// 9 — THE DISCOVERY (rho = +0.80)
// =======================================================================
s = p.addSlide(); s.background = { color: C.white };
header(s, "התגלית — הסגנון עובר בין תחומים", "התוצאה המרכזית");
{
  const d = fit(5.6, 4.2, 1330, 1032);
  s.addImage({ path: `${IMG}/style_alignment.png`, x: 0.7, y: 2.0, w: d.w, h: d.h });
}
s.addText("ρ = +0.80", { x: 6.7, y: 2.15, w: 6.0, h: 1.0, align: "right", rtlMode: false, fontFace: F, fontSize: 60, bold: true, color: C.navy, margin: 0 });
s.addText("מתאם דרגות ספירמן (1.0 = התאמה מושלמת)", { x: 6.7, y: 3.2, w: 6.0, h: 0.4, align: "right", rtlMode: true, fontFace: F, fontSize: 13, color: C.gray, margin: 0 });
s.addText([
  { text: "מי שאגרסיבי בברידג׳ — מתמקח אגרסיבי במשא ומתן.", options: { bullet: true, breakLine: true } },
  { text: "מי ששמרן בברידג׳ — נשאר שמרן במשא ומתן.", options: { bullet: true, breakLine: true } },
  { text: "מעל היעד שנקבע מראש (0.70) — הסגנון עובר חזק.", options: { bullet: true } },
], { x: 6.7, y: 3.8, w: 6.0, h: 2.2, align: "right", rtlMode: true, fontFace: F, fontSize: 16, color: C.ink, paraSpaceAfter: 10 });
footer(s, 9);

// =======================================================================
// 10 — THE CONTROL (rho = -0.90)
// =======================================================================
s = p.addSlide(); s.background = { color: C.light };
header(s, "הבקרה — האם זה לא טאוטולוגי?", "robustness");
{
  const d = fit(10.2, 3.5, 1870, 774);
  s.addShape(p.shapes.ROUNDED_RECTANGLE, { x: (W - d.w) / 2 - 0.12, y: 1.7, w: d.w + 0.24, h: d.h + 0.24, rectRadius: 0.1, fill: { color: C.white }, shadow: sh() });
  s.addImage({ path: `${IMG}/style_transfer_control.png`, x: (W - d.w) / 2, y: 1.82, w: d.w, h: d.h });
}
s.addText([
  { text: "החלפנו לכל פרופיל את מיומנויות הברידג׳ עם אלו של הפרופיל ", options: {} },
  { text: "ההפוך", options: { bold: true, color: C.red } },
  { text: ". המתאם התהפך מ־+0.80 ל־", options: {} },
  { text: "ρ = −0.90", options: { bold: true, color: C.red } },
  { text: " (p = 0.037). מסקנה: ההתנהגות נישאת בידי המיומנויות מהברידג׳, לא בידי התווית — לא טאוטולוגי.", options: {} },
], { x: 1.0, y: 5.95, w: W - 2.0, h: 0.95, align: "center", rtlMode: true, fontFace: F, fontSize: 14, color: C.ink });
footer(s, 10);

// =======================================================================
// 11 — WINNING vs STYLE
// =======================================================================
s = p.addSlide(); s.background = { color: C.white };
header(s, "ניצחון מול סגנון — ההבחנה החשובה", "תובנה");
function compare(x, title, rho, txt, col) {
  s.addShape(p.shapes.ROUNDED_RECTANGLE, { x, y: 2.1, w: 5.7, h: 3.1, rectRadius: 0.14, fill: { color: C.white }, shadow: sh() });
  s.addShape(p.shapes.RECTANGLE, { x, y: 2.1, w: 5.7, h: 0.7, fill: { color: col } });
  s.addText(title, { x: x + 0.2, y: 2.15, w: 5.3, h: 0.6, align: "right", valign: "middle", rtlMode: true, fontFace: F, fontSize: 18, bold: true, color: C.white, margin: 0 });
  s.addText(rho, { x: x + 0.2, y: 2.95, w: 5.3, h: 0.9, align: "right", rtlMode: false, fontFace: F, fontSize: 40, bold: true, color: col, margin: 0 });
  s.addText(txt, { x: x + 0.3, y: 3.95, w: 5.1, h: 1.1, align: "right", rtlMode: true, fontFace: F, fontSize: 14, color: C.ink, margin: 0 });
}
compare(7.0, "סגנון (Style)", "ρ = +0.80", "חזק ומבוקר. סוכן אגרסיבי בברידג׳ מתנהג אגרסיבי במשא ומתן — הסגנון עובר.", C.steel);
compare(0.65, "ניצחון (Winning)", "ρ ≈ +0.2", "רועש ותלוי־יריב. האם סגנון מנצח תלוי בכללי התמורה ובמתנגד — ולכן לא עובר נקי.", C.gray);
s.addShape(p.shapes.ROUNDED_RECTANGLE, { x: 2.9, y: 5.55, w: 7.5, h: 0.85, rectRadius: 0.12, fill: { color: C.navy } });
s.addText("השורה התחתונה: הסגנון עובר חזק; הניצחון רועש.",
  { x: 2.9, y: 5.55, w: 7.5, h: 0.85, align: "center", valign: "middle", rtlMode: true, fontFace: F, fontSize: 18, bold: true, color: C.white, margin: 0 });
footer(s, 11);

// =======================================================================
// 12 — MATH AT A GLANCE
// =======================================================================
s = p.addSlide(); s.background = { color: C.light };
header(s, "המתמטיקה במבט", "פורמליזציה");
const formulas = [
  ["מדיניות הסוכן האקראי (ZI־C)", "π(a|s) = 1 / |A_legal(s)|", "בחירה אחידה מתוך הפעולות החוקיות בלבד."],
  ["עומק הדרישה במשא ומתן", "θ = (P_ask − P_open) / (P_ask − P_floor)", "כמה נמוך הסוכן פותח, על פני טווח הוויתור של המוכר."],
  ["מתאם דרגות ספירמן", "ρ = 1 − 6Σd² / [K(K²−1)],  K = 5", "ההלימה בין דירוג האגרסיביות בשני התחומים."],
];
let fy = 2.05;
formulas.forEach((fm) => {
  s.addShape(p.shapes.ROUNDED_RECTANGLE, { x: 1.3, y: fy, w: 10.7, h: 1.45, rectRadius: 0.12, fill: { color: C.white }, shadow: sh() });
  s.addShape(p.shapes.RECTANGLE, { x: 11.75, y: fy, w: 0.25, h: 1.45, fill: { color: C.navy } });
  s.addText(fm[0], { x: 6.6, y: fy + 0.13, w: 5.0, h: 0.4, align: "right", rtlMode: true, fontFace: F, fontSize: 15, bold: true, color: C.navy, margin: 0 });
  s.addText(fm[2], { x: 6.6, y: fy + 0.6, w: 5.0, h: 0.7, align: "right", rtlMode: true, fontFace: F, fontSize: 12.5, color: C.gray, margin: 0 });
  s.addText(fm[1], { x: 1.5, y: fy + 0.35, w: 4.9, h: 0.75, align: "left", valign: "middle", fontFace: MONO, fontSize: 15, color: C.steel, margin: 0 });
  fy += 1.62;
});
footer(s, 12);

// =======================================================================
// 13 — LIMITATIONS
// =======================================================================
s = p.addSlide(); s.background = { color: C.white };
header(s, "מגבלות — בכנות מלאה", "אקדמיה");
const lims = [
  ["חמישה פרופילים בלבד", "ההספק הסטטיסטי נמוך והמובהקות אינה מספקת — המתאם הוא אינדיקציה חזקה, לא הוכחה. דגימת קצוות ההתפלגות היא בחירה מכוונת שממקסמת את האות."],
  ["משא ומתן מדומה", "התנהגות המוכר כוילה על 5,247 משאי־ומתן אמיתיים (Craigslist), אך אין דאטה אמיתית של אותו אדם בשני התחומים."],
  ["בלבול אפשרי", "פרופיל Slam Hunter מוגדר לפי הכרזת סלאמים — ייתכן שהוא מייצג חלקית גם חוזק שחקן כללי."],
];
let ly = 2.1;
lims.forEach((lm) => {
  s.addShape(p.shapes.ROUNDED_RECTANGLE, { x: 1.3, y: ly, w: 10.7, h: 1.35, rectRadius: 0.12, fill: { color: C.light }, line: { color: C.steel, width: 1 } });
  s.addText(lm[0], { x: 6.7, y: ly + 0.15, w: 5.1, h: 0.4, align: "right", rtlMode: true, fontFace: F, fontSize: 16, bold: true, color: C.steel, margin: 0 });
  s.addText(lm[1], { x: 1.5, y: ly + 0.15, w: 10.0, h: 1.05, align: "right", rtlMode: true, fontFace: F, fontSize: 13, color: C.ink, margin: 0 });
  // put title on a colored chip at the right
  ly += 1.55;
});
footer(s, 13);

// =======================================================================
// 14 — CONCLUSION
// =======================================================================
s = p.addSlide(); s.background = { color: C.navy };
s.addShape(p.shapes.ROUNDED_RECTANGLE, { x: 11.7, y: -0.7, w: 2.3, h: 2.3, rectRadius: 0.5, fill: { color: C.steel, transparency: 80 } });
s.addText("סיכום", { x: 0.7, y: 0.7, w: 6, h: 0.8, align: "right", rtlMode: true, fontFace: F, fontSize: 34, bold: true, color: C.white });
brandChip(s, W - 1.15, 0.7, 0.55);
const concl = [
  "שחקני עלית הם רצף התנהגותי, לא קבוצות — הארכיטיפים יושבים בקצוות.",
  "הסגנון עובר חזק בין תחומים (מתאם של 0.80), ומבוקר על־ידי בדיקת היפוך שמתהפכת לשלילי (מינוס 0.90).",
  "הניצחון לא עובר נקי — הוא תכונה של (סגנון × סביבה).",
];
let ccy = 2.0;
concl.forEach((t) => {
  s.addShape(p.shapes.OVAL, { x: 12.5, y: ccy + 0.05, w: 0.28, h: 0.28, fill: { color: C.ice } });
  s.addText(t, { x: 1.0, y: ccy - 0.05, w: 11.3, h: 0.7, align: "right", rtlMode: true, fontFace: F, fontSize: 18, color: C.white, margin: 0 });
  ccy += 0.95;
});
s.addShape(p.shapes.ROUNDED_RECTANGLE, { x: 1.0, y: 5.05, w: 11.3, h: 1.25, rectRadius: 0.12, fill: { color: C.white, transparency: 8 } });
s.addText("מה הלאה: עוד פרופילים לחיזוק ההספק, מדידת סגנון רב־ממדית במשא ומתן, שכפול על מודלים נוספים, ובהמשך — דאטה אנושית אמיתית.",
  { x: 1.25, y: 5.2, w: 10.8, h: 0.95, align: "right", rtlMode: true, fontFace: F, fontSize: 14, color: C.navy, margin: 0 });
s.addText([
  { text: "תודה!  ", options: { bold: true, color: C.white } },
  { text: "Nego", options: { bold: true, color: C.white } },
  { text: "Play", options: { bold: true, color: C.ice } },
], { x: 0.7, y: 6.7, w: 12, h: 0.5, align: "center", rtlMode: true, fontFace: F, fontSize: 18 });

p.writeFile({ fileName: "reports/NegoPlay_presentation.pptx" }).then((f) => console.log("WROTE", f));
