require("dotenv").config();

const express = require("express");
const cors = require("cors");
const http = require("http");
const { WebSocketServer, WebSocket } = require("ws");
const OpenAI = require("openai");

const app = express();
const server = http.createServer(app);
const wss = new WebSocketServer({ server, path: "/stream-audio" });

const port = Number(process.env.PORT) || 3000;
const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

// ───────────────────────────────────────────────────────────────────────────────
// Config
// ───────────────────────────────────────────────────────────────────────────────
const DEEPGRAM_API_KEY = process.env.DEEPGRAM_API_KEY;
const ANALYZE_MODEL = process.env.OPENAI_ANALYZE_MODEL || "gpt-4o-mini";
const MAX_HISTORY_TURNS = Number(process.env.MAX_HISTORY_TURNS) || 28;
const SESSION_TTL_MS = 1000 * 60 * 60 * 3;

const ANALYSIS_INTERVAL_MS = Number(process.env.ANALYSIS_INTERVAL_MS) || 4500;
const MIN_NEW_CHARS_FOR_ANALYSIS =
  Number(process.env.MIN_NEW_CHARS_FOR_ANALYSIS) || 45;

const SECRET_TOKEN = process.env.SECRET_TOKEN;

// gating
const MIN_PRIORITY_TO_SURFACE = Number(
  process.env.MIN_PRIORITY_TO_SURFACE ?? 74
);
const MIN_CONFIDENCE_TO_SURFACE = Number(
  process.env.MIN_CONFIDENCE_TO_SURFACE ?? 0.76
);

const ANSWER_MIN_PRIORITY = Number(process.env.ANSWER_MIN_PRIORITY ?? 60);
const ANSWER_MIN_CONFIDENCE = Number(
  process.env.ANSWER_MIN_CONFIDENCE ?? 0.62
);

const FACTCHECK_MIN_PRIORITY = Number(
  process.env.FACTCHECK_MIN_PRIORITY ?? 84
);
const FACTCHECK_MIN_CONFIDENCE = Number(
  process.env.FACTCHECK_MIN_CONFIDENCE ?? 0.88
);

const CLINICAL_QUESTION_MIN_PRIORITY = Number(
  process.env.CLINICAL_QUESTION_MIN_PRIORITY ?? 76
);
const CLINICAL_QUESTION_MIN_CONFIDENCE = Number(
  process.env.CLINICAL_QUESTION_MIN_CONFIDENCE ?? 0.74
);

const COMMENTARY_MIN_PRIORITY = Number(
  process.env.COMMENTARY_MIN_PRIORITY ?? 78
);
const COMMENTARY_MIN_CONFIDENCE = Number(
  process.env.COMMENTARY_MIN_CONFIDENCE ?? 0.78
);

const TEACHING_MIN_PRIORITY = Number(
  process.env.TEACHING_MIN_PRIORITY ?? 80
);
const TEACHING_MIN_CONFIDENCE = Number(
  process.env.TEACHING_MIN_CONFIDENCE ?? 0.8
);

const REFERENCE_MIN_PRIORITY = Number(
  process.env.REFERENCE_MIN_PRIORITY ?? 68
);
const REFERENCE_MIN_CONFIDENCE = Number(
  process.env.REFERENCE_MIN_CONFIDENCE ?? 0.68
);

const BUBBLE_SIMILARITY_THRESHOLD = Number(
  process.env.BUBBLE_SIMILARITY_THRESHOLD ?? 0.66
);

const LABEL_SIMILARITY_THRESHOLD = Number(
  process.env.LABEL_SIMILARITY_THRESHOLD ?? 0.82
);

const TRANSCRIPT_OVERLAP_SUPPRESSION_THRESHOLD = Number(
  process.env.TRANSCRIPT_OVERLAP_SUPPRESSION_THRESHOLD ?? 0.72
);

const MODE_COOLDOWN_MS = Number(process.env.MODE_COOLDOWN_MS) || 12000;
const TOPIC_COOLDOWN_MS = Number(process.env.TOPIC_COOLDOWN_MS) || 90000;

// ───────────────────────────────────────────────────────────────────────────────
// Express
// ───────────────────────────────────────────────────────────────────────────────
app.use(
  cors({
    origin: "*",
    methods: ["GET", "POST", "OPTIONS"],
    allowedHeaders: ["Content-Type", "x-api-key"],
  })
);

app.use(express.json({ limit: "25mb" }));

app.get("/", (_req, res) => {
  res.json({
    ok: true,
    service: "talky-clinical-copilot",
    status: "running",
  });
});

app.get("/health", (_req, res) => {
  res.json({
    ok: true,
    service: "talky-clinical-copilot",
    sessions: sessions.size,
  });
});

// ───────────────────────────────────────────────────────────────────────────────
// Auth
// ───────────────────────────────────────────────────────────────────────────────
app.use((req, res, next) => {
  if (
    req.method === "OPTIONS" ||
    req.path === "/" ||
    req.path === "/health"
  ) {
    return next();
  }

  const token = req.headers["x-api-key"];

  if (!SECRET_TOKEN) {
    console.warn("WARNING: SECRET_TOKEN not set");
    return next();
  }

  if (token !== SECRET_TOKEN) {
    return res.status(401).json({ error: "Unauthorized" });
  }

  next();
});

// ───────────────────────────────────────────────────────────────────────────────
// Session store
// ───────────────────────────────────────────────────────────────────────────────
const sessions = new Map();

function getSession(id = "default") {
  if (!sessions.has(id)) {
    sessions.set(id, {
      id,
      turns: [],
      oneLiner: "",
      title: "Talky",
      shownBubbles: [],
      surfacedTopicKeys: [],
      surfacedModes: [],
      lastActiveAt: Date.now(),
      lastSurfacedExpanded: "",
      lastSurfacedLabel: "",
    });
  }

  const session = sessions.get(id);
  session.lastActiveAt = Date.now();
  return session;
}

function addTurn(session, text) {
  const clean = normalizeWS(text);
  if (!clean) return;

  session.turns.push({
    text: clean,
    ts: Date.now(),
  });

  if (session.turns.length > MAX_HISTORY_TURNS) {
    session.turns = session.turns.slice(-MAX_HISTORY_TURNS);
  }
}

function buildWindow(session) {
  return session.turns.map((t) => t.text).join("\n");
}

function rememberSurfacedBubble(session, entry) {
  const cleanExpanded = normalizeWS(entry?.expanded || "");
  const cleanLabel = normalizeWS(entry?.label || "");
  const cleanTopicKey = normalizeTopicKey(entry?.topicKey || "");
  const cleanMode = normalizeMode(entry?.mode || "none");
  const now = Date.now();

  if (cleanExpanded) {
    session.shownBubbles.unshift({
      text: cleanExpanded,
      label: cleanLabel,
      topicKey: cleanTopicKey,
      mode: cleanMode,
      ts: now,
    });
  }

  if (session.shownBubbles.length > 24) {
    session.shownBubbles = session.shownBubbles.slice(0, 24);
  }

  if (cleanTopicKey) {
    session.surfacedTopicKeys.unshift({
      key: cleanTopicKey,
      mode: cleanMode,
      ts: now,
    });

    if (session.surfacedTopicKeys.length > 30) {
      session.surfacedTopicKeys = session.surfacedTopicKeys.slice(0, 30);
    }
  }

  if (cleanMode && cleanMode !== "none") {
    session.surfacedModes.unshift({
      mode: cleanMode,
      ts: now,
    });

    if (session.surfacedModes.length > 20) {
      session.surfacedModes = session.surfacedModes.slice(0, 20);
    }
  }
}

setInterval(() => {
  const now = Date.now();
  for (const [id, session] of sessions.entries()) {
    if (now - session.lastActiveAt > SESSION_TTL_MS) {
      sessions.delete(id);
    }
  }
}, 1000 * 60 * 30);

// ───────────────────────────────────────────────────────────────────────────────
// Utils
// ───────────────────────────────────────────────────────────────────────────────
function safeJson(text, fallback = null) {
  try {
    return JSON.parse(text);
  } catch {
    return fallback;
  }
}

function normalizeWS(text) {
  return String(text || "")
    .replace(/\r/g, "")
    .replace(/[ \t]+/g, " ")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

function normalizeFlat(text) {
  return normalizeWS(text).replace(/\n+/g, " ").trim();
}

function clamp(str, max) {
  const s = String(str || "");
  return s.length <= max ? s : s.slice(0, max);
}

function tokenize(text) {
  return normalizeFlat(text)
    .toLowerCase()
    .replace(/[^\w\s/-]/g, " ")
    .split(/\s+/)
    .filter((w) => w.length > 2);
}

function jaccardSimilarity(a, b) {
  const A = new Set(tokenize(a));
  const B = new Set(tokenize(b));

  if (!A.size || !B.size) return 0;

  let intersection = 0;
  for (const word of A) {
    if (B.has(word)) intersection++;
  }

  const union = new Set([...A, ...B]).size;
  return union ? intersection / union : 0;
}

function normalizeTopicKey(value) {
  return String(value || "")
    .toLowerCase()
    .replace(/[^\w/-]+/g, "-")
    .replace(/-+/g, "-")
    .replace(/^-|-$/g, "")
    .slice(0, 80);
}

function normalizeMode(mode) {
  const m = String(mode || "none").toLowerCase().trim();
  const allowed = new Set([
    "answer",
    "fact_check",
    "clinical_question",
    "commentary",
    "teaching",
    "reference",
    "none",
  ]);
  return allowed.has(m) ? m : "none";
}

function modeRank(mode) {
  switch (mode) {
    case "answer":
      return 6;
    case "fact_check":
      return 5;
    case "clinical_question":
      return 4;
    case "commentary":
      return 3;
    case "reference":
      return 2;
    case "teaching":
      return 1;
    default:
      return 0;
  }
}

function getModeThresholds(mode) {
  switch (mode) {
    case "answer":
      return {
        minPriority: ANSWER_MIN_PRIORITY,
        minConfidence: ANSWER_MIN_CONFIDENCE,
      };
    case "fact_check":
      return {
        minPriority: FACTCHECK_MIN_PRIORITY,
        minConfidence: FACTCHECK_MIN_CONFIDENCE,
      };
    case "clinical_question":
      return {
        minPriority: CLINICAL_QUESTION_MIN_PRIORITY,
        minConfidence: CLINICAL_QUESTION_MIN_CONFIDENCE,
      };
    case "commentary":
      return {
        minPriority: COMMENTARY_MIN_PRIORITY,
        minConfidence: COMMENTARY_MIN_CONFIDENCE,
      };
    case "teaching":
      return {
        minPriority: TEACHING_MIN_PRIORITY,
        minConfidence: TEACHING_MIN_CONFIDENCE,
      };
    case "reference":
      return {
        minPriority: REFERENCE_MIN_PRIORITY,
        minConfidence: REFERENCE_MIN_CONFIDENCE,
      };
    default:
      return {
        minPriority: MIN_PRIORITY_TO_SURFACE,
        minConfidence: MIN_CONFIDENCE_TO_SURFACE,
      };
  }
}

function parseModelJson(text) {
  const direct = safeJson(text);
  if (direct) return direct;

  const fenced = String(text || "").match(/```json\s*([\s\S]*?)```/i);
  if (fenced?.[1]) return safeJson(fenced[1]);

  return null;
}

function extractBubbleEntries(input) {
  if (!Array.isArray(input)) return [];

  return input
    .map((b) => ({
      text: normalizeWS(b?.expanded || b?.short || ""),
      label: normalizeWS(b?.short || ""),
      topicKey: normalizeTopicKey(b?.topicKey || ""),
      mode: normalizeMode(
        frontendCategoryToMode(b?.type || b?.mode || "commentary")
      ),
      ts: Date.now(),
    }))
    .filter((b) => b.text || b.label)
    .slice(0, 20);
}

function frontendCategoryToMode(category) {
  switch (String(category || "").toLowerCase().trim()) {
    case "question":
      return "answer";
    case "factcheck":
      return "fact_check";
    case "reference":
      return "reference";
    case "insight":
      return "commentary";
    default:
      return "commentary";
  }
}

function modeToFrontendCategory(mode) {
  switch (normalizeMode(mode)) {
    case "answer":
      return "question";
    case "fact_check":
      return "factcheck";
    case "reference":
      return "reference";
    case "clinical_question":
    case "commentary":
    case "teaching":
      return "insight";
    default:
      return "none";
  }
}

function makeLabelForMode(mode, label, title) {
  const cleanLabel = normalizeFlat(label);
  const cleanTitle = normalizeFlat(title);

  const base = cleanLabel || cleanTitle || "";

  if (!base) {
    switch (mode) {
      case "answer":
        return "Answer";
      case "fact_check":
        return "Correction";
      case "clinical_question":
        return "Ask";
      case "commentary":
        return "Lean";
      case "teaching":
        return "Pearl";
      case "reference":
        return "Reference";
      default:
        return "";
    }
  }

  const trimmed = clamp(base, 22);

  switch (mode) {
    case "answer":
      return clamp(`Ans: ${trimmed}`, 28);
    case "fact_check":
      return clamp(`Fix: ${trimmed}`, 28);
    case "clinical_question":
      return clamp(`Ask: ${trimmed}`, 28);
    case "commentary":
      return clamp(`Lean: ${trimmed}`, 28);
    case "teaching":
      return clamp(`Pearl: ${trimmed}`, 28);
    case "reference":
      return clamp(`Ref: ${trimmed}`, 28);
    default:
      return clamp(trimmed, 28);
  }
}

function bodyLooksGeneric(text) {
  const clean = normalizeFlat(text).toLowerCase();
  if (!clean) return true;

  const genericPatterns = [
    "correlate clinically",
    "consider further workup",
    "clinical correlation",
    "follow up clinically",
    "could be considered",
    "depends on the clinical context",
    "additional imaging may be helpful",
    "continue to monitor",
    "rule out stroke",
    "further evaluation is warranted",
    "recommend discussing with the team",
  ];

  return genericPatterns.some((p) => clean.includes(p));
}

function transcriptOverlapTooHigh(body, recentTranscript, transcriptWindow) {
  const compareA = normalizeFlat(recentTranscript || "");
  const compareB = normalizeFlat(transcriptWindow || "");
  const cleanBody = normalizeFlat(body || "");

  if (!cleanBody) return true;

  const simRecent = jaccardSimilarity(cleanBody, compareA);
  const simWindow = jaccardSimilarity(cleanBody, compareB);

  return Math.max(simRecent, simWindow) >= TRANSCRIPT_OVERLAP_SUPPRESSION_THRESHOLD;
}

function findNearDuplicate(body, label, history) {
  const cleanBody = normalizeFlat(body);
  const cleanLabel = normalizeFlat(label);

  for (const h of history || []) {
    const textSim = cleanBody
      ? jaccardSimilarity(cleanBody, h?.text || "")
      : 0;
    const labelSim = cleanLabel
      ? jaccardSimilarity(cleanLabel, h?.label || "")
      : 0;

    if (
      textSim >= BUBBLE_SIMILARITY_THRESHOLD ||
      labelSim >= LABEL_SIMILARITY_THRESHOLD
    ) {
      return true;
    }
  }

  return false;
}

function topicRecentlySurfaced(session, topicKey) {
  const clean = normalizeTopicKey(topicKey);
  if (!clean) return false;

  const now = Date.now();
  return (session.surfacedTopicKeys || []).some(
    (t) => t.key === clean && now - t.ts < TOPIC_COOLDOWN_MS
  );
}

function modeRecentlySurfaced(session, mode) {
  const clean = normalizeMode(mode);
  if (!clean || clean === "none") return false;

  const now = Date.now();
  return (session.surfacedModes || []).some(
    (m) => m.mode === clean && now - m.ts < MODE_COOLDOWN_MS
  );
}

function recentHistoryForPrompt(session, bubbles = []) {
  const merged = [
    ...extractBubbleEntries(bubbles),
    ...(session.shownBubbles || []),
  ];

  const seen = new Set();
  const out = [];

  for (const item of merged) {
    const key = `${item.topicKey}|${item.label}|${item.text}`;
    if (seen.has(key)) continue;
    seen.add(key);
    out.push(item);
    if (out.length >= 12) break;
  }

  return out;
}

function buildRecentContext(session, maxTurns = 8) {
  const turns = session.turns.slice(-maxTurns);
  return turns.map((t) => t.text).join("\n");
}

// Compact model schema → compatibility schema for current frontend
function normalizeAnalysis(parsed) {
  if (!parsed || typeof parsed !== "object") {
    return {
      mode: "none",
      priority: 0,
      confidence: 0,
      title: "Talky",
      one_liner: "",
      bubble_label: "",
      question_detected: true,
      answer: "",
      answer_expanded: "",
      insight: "",
      insight_expanded: "",
      category: "none",
      summary: "",
      content: "",
      topic_key: "",
    };
  }

  const mode = normalizeMode(parsed.mode);
  let priority = Number(parsed.priority);
  let confidence = Number(parsed.confidence);

  if (!Number.isFinite(priority)) priority = 0;
  if (!Number.isFinite(confidence)) confidence = 0;

  priority = Math.max(0, Math.min(100, Math.round(priority)));
  confidence = Math.max(0, Math.min(1, confidence));

  const title = clamp(normalizeFlat(parsed.title || ""), 48) || "Talky";
  const summary = clamp(normalizeWS(parsed.summary || ""), 900);
  const rawLabel = clamp(normalizeFlat(parsed.label || ""), 32);
  const body = clamp(normalizeWS(parsed.body || ""), 1600);
  const topicKey = normalizeTopicKey(parsed.topic_key || "");

  const thresholds = getModeThresholds(mode);

  const globalPass =
    priority >= MIN_PRIORITY_TO_SURFACE &&
    confidence >= MIN_CONFIDENCE_TO_SURFACE;

  const modePass =
    priority >= thresholds.minPriority &&
    confidence >= thresholds.minConfidence;

  const shouldSurface =
    mode !== "none" && !!body && !bodyLooksGeneric(body) && (modePass || (mode === "answer" && globalPass));

  const frontendCategory = shouldSurface ? modeToFrontendCategory(mode) : "none";
  const bubbleLabel = shouldSurface ? makeLabelForMode(mode, rawLabel, title) : "";
  const isQuestionLike = shouldSurface && mode === "answer";

  return {
    mode: shouldSurface ? mode : "none",
    priority: shouldSurface ? priority : 0,
    confidence: shouldSurface ? confidence : 0,
    title,
    one_liner: summary || "Conversation in progress",
    bubble_label: bubbleLabel,
    question_detected: isQuestionLike,
    answer: isQuestionLike ? bubbleLabel || "Answer" : "",
    answer_expanded: isQuestionLike ? body : "",
    insight: !isQuestionLike && shouldSurface ? bubbleLabel || "Insight" : "",
    insight_expanded: !isQuestionLike && shouldSurface ? body : "",
    category: frontendCategory,
    summary,
    content: body,
    topic_key: topicKey,
  };
}

// ───────────────────────────────────────────────────────────────────────────────
// Prompt
// ───────────────────────────────────────────────────────────────────────────────
function buildPrompt({
  recentTranscript,
  transcriptWindow,
  priorSummary,
  priorTitle,
  recentContext,
  previousBubbles,
}) {
  const bubbleHistory = previousBubbles?.length
    ? previousBubbles
        .map((b, i) => {
          const parts = [];
          if (b.mode) parts.push(`mode=${b.mode}`);
          if (b.topicKey) parts.push(`topic=${b.topicKey}`);
          if (b.label) parts.push(`label=${b.label}`);
          if (b.text) parts.push(`text=${b.text}`);
          return `${i + 1}. ${parts.join(" | ")}`;
        })
        .join("\n")
    : "none";

  return `You are Talky, an elite real-time neurologic clinical copilot embedded in smart glasses worn by a vascular neurologist.

IDENTITY
- Think like a world-renowned neurologist silently listening at bedside, table rounds, consult rounds, conference, clinic, or stroke alert.
- Your job is NOT to summarize the transcript.
- Your job is to add the single most useful thought that would actually help in the moment.

WHO YOU ARE HELPING
- The wearer is already a neurologist.
- Do not explain basic concepts.
- Do not give textbook filler.
- Do not pad.
- Be concise, sharp, and high-yield.

PRIORITY ORDER
1. Answer a direct question asked aloud.
2. Correct a medically false or unsafe statement if confidence is high.
3. Ask ONE high-yield clinical question that would materially change diagnosis or management.
4. Provide concise diagnostic or management commentary if there is a strong lean with a lock-in discriminator.
5. Provide a short teaching pearl only if it is genuinely useful and niche.
6. Otherwise return none.

YOUR MODES
- answer
- fact_check
- clinical_question
- commentary
- teaching
- reference
- none

MODE DEFINITIONS
- answer: someone directly asked a question aloud and you can answer it.
- fact_check: the speaker said something materially incorrect, misleading, or unsafe.
- clinical_question: ask exactly one missing question whose answer would significantly move the case.
- commentary: your best high-value attending-level interpretation or next discriminator.
- teaching: short niche pearl only when truly useful.
- reference: brief protocol/detail lookup when explicitly asked for a reference-style fact.
- none: nothing worth surfacing.

STRICT RULES
- Prefer silence over low-value output.
- Do not restate the transcript.
- Do not paraphrase what was already said unless correcting it.
- Do not generate generic suggestions.
- Do not say "consider more workup" unless you specify exactly why it matters now.
- If asking a clinical question, ask only ONE question.
- If providing commentary, include the strongest reason and what would lock it in or move it.
- If fact checking, be very careful and only do it when confidence is high.
- Avoid repeating recent surfaced items even if worded differently.
- Teaching pearls should be short and niche, not basic definitions.
- Do not output more than one idea.

OUTPUT JSON ONLY
Return ONLY valid JSON with this exact schema:
{
  "mode": "answer | fact_check | clinical_question | commentary | teaching | reference | none",
  "priority": 0-100,
  "confidence": 0.0-1.0,
  "title": "2-5 words",
  "label": "very short topic label, 1-4 words",
  "body": "1-4 dense sentences, usually 35-140 words, no filler",
  "summary": "rolling 1-4 sentence conversation summary for UI context",
  "topic_key": "stable short slug for the core idea"
}

SCORING GUIDANCE
- answer should usually have the highest priority if a real question was asked.
- fact_check should be rare and high-confidence.
- commentary should only fire if it is genuinely additive.
- teaching should be less preferred than answer, fact_check, clinical_question, and commentary.
- none is the correct choice very often.

GOOD BEHAVIOR EXAMPLES
- If they ask: "What findings suggest MMN rather than ALS?" -> answer directly.
- If they mention CSF pleocytosis in a neuroinflammatory case -> maybe ask whether it is lymphocytic vs neutrophilic if that would truly move the differential.
- If the conversation strongly fits ADEM vs MS -> commentary can say why and what locks it in.
- If someone states an actually incorrect medical claim -> fact_check, but only if confidence is high.

BAD BEHAVIOR
- repeating history that was already spoken
- generic next steps
- generic definitions
- filler teaching
- multiple questions
- broad low-yield differentials

PREVIOUS SURFACED ITEMS (do not repeat conceptually):
${bubbleHistory}

PRIOR TITLE:
${priorTitle || "none"}

PRIOR SUMMARY:
${priorSummary || "none"}

RECENT CONTEXT WINDOW:
${recentContext || "none"}

FULL TRANSCRIPT WINDOW:
${transcriptWindow || "none"}

MOST RECENT TRANSCRIPT:
${recentTranscript || "none"}

Return ONLY valid JSON.`;
}

async function analyzeTranscript({
  recentTranscript,
  transcriptWindow,
  priorSummary,
  priorTitle,
  recentContext,
  previousBubbles,
}) {
  const prompt = buildPrompt({
    recentTranscript: clamp(recentTranscript, 4500),
    transcriptWindow: clamp(transcriptWindow, 9000),
    priorSummary: clamp(priorSummary || "", 900),
    priorTitle: clamp(priorTitle || "", 120),
    recentContext: clamp(recentContext || "", 5000),
    previousBubbles: previousBubbles || [],
  });

  const response = await openai.chat.completions.create({
    model: ANALYZE_MODEL,
    temperature: 0.08,
    max_tokens: 700,
    response_format: { type: "json_object" },
    messages: [
      { role: "system", content: prompt },
      { role: "user", content: "Analyze the live conversation and return JSON only." },
    ],
  });

  const outputText = response.choices?.[0]?.message?.content?.trim() || "{}";
  const parsed = parseModelJson(outputText);
  const normalized = normalizeAnalysis(parsed);

  if (!normalized.one_liner) normalized.one_liner = "Conversation in progress";
  if (!normalized.title) normalized.title = "Talky";

  return normalized;
}

function shouldSuppressAnalysis({
  analyzed,
  session,
  priorHistory,
  recentTranscript,
  transcriptWindow,
}) {
  if (!analyzed || analyzed.mode === "none") {
    return { suppress: true, reason: "none" };
  }

  const body = analyzed.answer_expanded || analyzed.insight_expanded || "";
  const label =
    analyzed.answer || analyzed.insight || analyzed.bubble_label || "";
  const mode = analyzed.mode;
  const topicKey = analyzed.topic_key || "";

  if (!body) {
    return { suppress: true, reason: "empty_body" };
  }

  if (bodyLooksGeneric(body)) {
    return { suppress: true, reason: "generic" };
  }

  if (findNearDuplicate(body, label, priorHistory)) {
    return { suppress: true, reason: "duplicate_history" };
  }

  if (
    session.lastSurfacedExpanded &&
    jaccardSimilarity(body, session.lastSurfacedExpanded) >=
      BUBBLE_SIMILARITY_THRESHOLD
  ) {
    return { suppress: true, reason: "duplicate_last_expanded" };
  }

  if (
    session.lastSurfacedLabel &&
    label &&
    jaccardSimilarity(label, session.lastSurfacedLabel) >=
      LABEL_SIMILARITY_THRESHOLD
  ) {
    return { suppress: true, reason: "duplicate_last_label" };
  }

  if (topicKey && topicRecentlySurfaced(session, topicKey)) {
    return { suppress: true, reason: "topic_cooldown" };
  }

  if (
    mode !== "answer" &&
    mode !== "fact_check" &&
    modeRecentlySurfaced(session, mode)
  ) {
    return { suppress: true, reason: "mode_cooldown" };
  }

  if (
    mode !== "answer" &&
    mode !== "fact_check" &&
    transcriptOverlapTooHigh(body, recentTranscript, transcriptWindow)
  ) {
    return { suppress: true, reason: "too_close_to_transcript" };
  }

  return { suppress: false, reason: "" };
}

// ───────────────────────────────────────────────────────────────────────────────
// Deepgram
// ───────────────────────────────────────────────────────────────────────────────
function openDeepgramSocket() {
  return new Promise((resolve, reject) => {
    if (!DEEPGRAM_API_KEY) {
      reject(new Error("DEEPGRAM_API_KEY not set"));
      return;
    }

    const url = [
      "wss://api.deepgram.com/v1/listen",
      "?model=nova-3-medical",
      "&language=en-US",
      "&encoding=linear16",
      "&sample_rate=16000",
      "&channels=1",
      "&interim_results=true",
      "&utterance_end_ms=2000",
      "&punctuate=true",
      "&smart_format=true",
      "&endpointing=300",
    ].join("");

    const dg = new WebSocket(url, {
      headers: { Authorization: `Token ${DEEPGRAM_API_KEY}` },
    });

    const timeout = setTimeout(() => {
      reject(new Error("Deepgram connection timeout"));
    }, 10000);

    dg.on("open", () => {
      clearTimeout(timeout);
      console.log("Deepgram ready");
      resolve(dg);
    });

    dg.on("error", (err) => {
      clearTimeout(timeout);
      reject(err);
    });

    dg.on("close", (code, reason) => {
      console.log(`Deepgram closed: ${code} "${reason.toString()}"`);
    });
  });
}

// ───────────────────────────────────────────────────────────────────────────────
// Client WebSocket handler
// ───────────────────────────────────────────────────────────────────────────────
wss.on("connection", async (clientWs, req) => {
  const url = new URL(req.url, `http://${req.headers.host}`);
  const token = url.searchParams.get("token");

  if (SECRET_TOKEN && token !== SECRET_TOKEN) {
    console.log("Rejected WS connection: invalid token");
    clientWs.close(1008, "Invalid token");
    return;
  }

  console.log("Client connected:", req.socket.remoteAddress);

  let dgSocket = null;
  let sessionId = "default";
  let accumulatedText = "";
  let lastAnalyzedLen = 0;
  let analysisTimer = null;
  let isAnalyzing = false;
  let isStopped = false;

  function stopAnalysisTimer() {
    if (analysisTimer) {
      clearInterval(analysisTimer);
      analysisTimer = null;
    }
  }

  function sendJson(payload) {
    if (clientWs.readyState === WebSocket.OPEN) {
      clientWs.send(JSON.stringify(payload));
    }
  }

  async function runAnalysis({ final = false, bubbles = [] } = {}) {
    if (isAnalyzing) return;
    if (!accumulatedText.trim()) return;

    const newChars = accumulatedText.length - lastAnalyzedLen;
    if (!final && newChars < MIN_NEW_CHARS_FOR_ANALYSIS) return;

    isAnalyzing = true;

    try {
      const session = getSession(sessionId);
      const transcriptWindow = buildWindow(session);
      const recentContext = buildRecentContext(session, 8);
      const priorHistory = recentHistoryForPrompt(session, bubbles);

      const analyzed = await analyzeTranscript({
        recentTranscript: accumulatedText,
        transcriptWindow,
        recentContext,
        priorSummary: session.oneLiner,
        priorTitle: session.title,
        previousBubbles: priorHistory,
      });

      if (analyzed.title) session.title = analyzed.title;
      if (analyzed.one_liner) session.oneLiner = analyzed.one_liner;

      const suppression = shouldSuppressAnalysis({
        analyzed,
        session,
        priorHistory,
        recentTranscript: accumulatedText,
        transcriptWindow,
      });

      if (suppression.suppress) {
        lastAnalyzedLen = accumulatedText.length;

        sendJson({
          type: "analysis",
          title: session.title || analyzed.title || "Talky",
          one_liner: analyzed.one_liner || session.oneLiner || "Conversation in progress",
          bubble_label: "",
          question_detected: false,
          answer: "",
          answer_expanded: "",
          insight: "",
          insight_expanded: "",
          category: "none",
          confidence: analyzed.confidence || 0,
          priority: analyzed.priority || 0,
          mode: "none",
          topic_key: analyzed.topic_key || "",
        });
        return;
      }

      lastAnalyzedLen = accumulatedText.length;

      const surfaced = analyzed.answer_expanded || analyzed.insight_expanded || "";
      const surfacedLabel =
        analyzed.answer || analyzed.insight || analyzed.bubble_label || "";

      session.lastSurfacedExpanded = surfaced;
      session.lastSurfacedLabel = surfacedLabel;

      rememberSurfacedBubble(session, {
        expanded: surfaced,
        label: surfacedLabel,
        topicKey: analyzed.topic_key || "",
        mode: analyzed.mode,
      });

      sendJson({
        type: "analysis",
        title: analyzed.title,
        one_liner: analyzed.one_liner,
        bubble_label: analyzed.bubble_label || "",
        question_detected: analyzed.question_detected,
        answer: analyzed.answer || "",
        answer_expanded: analyzed.answer_expanded || "",
        insight: analyzed.insight || "",
        insight_expanded: analyzed.insight_expanded || "",
        category: analyzed.category,
        confidence: analyzed.confidence,
        priority: analyzed.priority,
        mode: analyzed.mode,
        topic_key: analyzed.topic_key || "",
      });
    } catch (err) {
      console.error("Analysis error:", err.message);
      sendJson({
        type: "error",
        error: "Analysis failed",
      });
    } finally {
      isAnalyzing = false;
    }
  }

  function startAnalysisTimer() {
    stopAnalysisTimer();
    analysisTimer = setInterval(async () => {
      await runAnalysis({ final: false });
    }, ANALYSIS_INTERVAL_MS);
  }

  try {
    dgSocket = await openDeepgramSocket();
    startAnalysisTimer();
  } catch (err) {
    console.error("Failed to open Deepgram socket:", err.message);
    sendJson({ type: "error", error: "Failed to connect to Deepgram" });
    clientWs.close();
    return;
  }

  dgSocket.on("message", async (data) => {
    try {
      const msg = JSON.parse(data.toString());
      if (msg.type !== "Results") return;

      const transcript = msg.channel?.alternatives?.[0]?.transcript?.trim();
      if (!transcript) return;

      if (msg.is_final) {
        accumulatedText = normalizeWS(
          accumulatedText ? `${accumulatedText} ${transcript}` : transcript
        );

        addTurn(getSession(sessionId), transcript);

        sendJson({
          type: "final_transcript",
          text: transcript,
          fullText: accumulatedText,
        });
      } else {
        sendJson({
          type: "transcript",
          text: transcript,
        });
      }
    } catch (err) {
      console.error("Deepgram message parse error:", err.message);
    }
  });

  dgSocket.on("error", (err) => {
    console.error("Deepgram WS error:", err.message);
    sendJson({ type: "error", error: "Deepgram stream error" });
  });

  clientWs.on("message", async (data, isBinary) => {
    if (isStopped) return;

    if (isBinary) {
      if (dgSocket?.readyState === WebSocket.OPEN) {
        dgSocket.send(data);
      }
      return;
    }

    const msg = safeJson(data.toString());
    if (!msg) return;

    if (msg.type === "start") {
      sessionId = String(msg.sessionId || "default").slice(0, 64);
      getSession(sessionId);
      return;
    }

    if (msg.type === "stop") {
      isStopped = true;
      stopAnalysisTimer();
      console.log("Stop received, doing final analysis.");

      sessionId = String(msg.sessionId || "default").slice(0, 64);

      if (dgSocket?.readyState === WebSocket.OPEN) {
        dgSocket.send(JSON.stringify({ type: "CloseStream" }));
      }

      const transcript = normalizeWS(accumulatedText || msg.fullTranscript || "");
      if (!transcript.trim()) {
        sendJson({ type: "stopped" });
        return;
      }

      const session = getSession(sessionId);
      if (!session.oneLiner && msg.oneLiner) {
        session.oneLiner = normalizeWS(msg.oneLiner);
      }

      await runAnalysis({
        final: true,
        bubbles: msg.bubbles || [],
      });

      sendJson({ type: "stopped" });
      return;
    }
  });

  clientWs.on("close", () => {
    console.log("Client disconnected");
    stopAnalysisTimer();

    if (dgSocket?.readyState === WebSocket.OPEN) {
      try {
        dgSocket.close();
      } catch {}
    }
  });

  clientWs.on("error", (err) => {
    console.error("Client WS error:", err.message);
  });
});

// ───────────────────────────────────────────────────────────────────────────────
// Maintenance routes
// ───────────────────────────────────────────────────────────────────────────────
app.post("/session/clear", (req, res) => {
  const id = String(req.body?.sessionId || "default");
  sessions.delete(id);
  res.json({ ok: true, sessionId: id });
});

app.use((err, _req, res, _next) => {
  console.error("Unhandled server error:", err);
  res.status(500).json({
    error: "Internal server error",
    details: err?.message || String(err),
  });
});

server.listen(port, "0.0.0.0", () => {
  console.log(`Server listening on port ${port}`);
  console.log(`Deepgram: ${DEEPGRAM_API_KEY ? "configured" : "MISSING KEY"}`);
  console.log(
    `OpenAI: ${process.env.OPENAI_API_KEY ? "configured" : "MISSING KEY"}`
  );
  console.log(`Secret token: ${SECRET_TOKEN ? "configured" : "MISSING TOKEN"}`);
});
