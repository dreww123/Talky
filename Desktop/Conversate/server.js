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
const MAX_HISTORY_TURNS = Number(process.env.MAX_HISTORY_TURNS) || 20;
const SESSION_TTL_MS = 1000 * 60 * 60 * 3;
const ANALYSIS_INTERVAL_MS = Number(process.env.ANALYSIS_INTERVAL_MS) || 10000;
const MIN_NEW_CHARS_FOR_ANALYSIS =
  Number(process.env.MIN_NEW_CHARS_FOR_ANALYSIS) || 30;
const MIN_CONFIDENCE_TO_SURFACE = Number(
  process.env.MIN_CONFIDENCE_TO_SURFACE ?? 0.68
);
const SECRET_TOKEN = process.env.SECRET_TOKEN;

app.use(
  cors({
    origin: "*",
    methods: ["GET", "POST", "OPTIONS"],
    allowedHeaders: ["Content-Type", "x-api-key"],
  })
);

app.use(express.json({ limit: "25mb" }));

// ───────────────────────────────────────────────────────────────────────────────
// Auth Middleware
// ───────────────────────────────────────────────────────────────────────────────
app.use((req, res, next) => {
  // Allow CORS preflight through
  if (req.method === "OPTIONS") {
    return res.sendStatus(204);
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
      title: "Clinical Copilot",
      shownBubbles: [],
      lastActiveAt: Date.now(),
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

function clamp(str, max) {
  const s = String(str || "");
  return s.length <= max ? s : s.slice(0, max);
}

function extractBubbleTexts(bubbles) {
  if (!Array.isArray(bubbles)) return [];
  return bubbles
    .map((b) => normalizeWS(b?.short || b?.expanded || ""))
    .filter(Boolean)
    .slice(0, 12);
}

function dedupe(text, history) {
  const clean = normalizeWS(text);
  if (!clean) return "";
  const normalized = clean.toLowerCase();
  const seen = new Set(
    (history || []).map((h) => normalizeWS(h).toLowerCase())
  );
  return seen.has(normalized) ? "" : clean;
}

function parseModelJson(text) {
  const direct = safeJson(text);
  if (direct) return direct;

  const fenced = String(text || "").match(/```json\s*([\s\S]*?)```/i);
  if (fenced?.[1]) return safeJson(fenced[1]);

  return null;
}

// Compact model schema → compatibility schema for current frontend
function normalizeAnalysis(parsed) {
  if (!parsed || typeof parsed !== "object") {
    return {
      title: "",
      one_liner: "",
      question_detected: false,
      answer: "",
      answer_expanded: "",
      insight: "",
      insight_expanded: "",
      category: "insight",
      confidence: 0,
      summary: "",
      content: "",
    };
  }

  const rawCategory = String(parsed.category || "none").toLowerCase().trim();
  const allowed = new Set([
    "question",
    "factcheck",
    "insight",
    "reference",
    "none",
  ]);
  const category = allowed.has(rawCategory) ? rawCategory : "none";

  let confidence = 0;
  if (
    typeof parsed.confidence === "number" &&
    Number.isFinite(parsed.confidence)
  ) {
    confidence = Math.max(0, Math.min(1, parsed.confidence));
  }

  const title = clamp(normalizeWS(parsed.title || ""), 30);
  const oneLiner = clamp(normalizeWS(parsed.summary || ""), 120);
  const content = clamp(normalizeWS(parsed.content || ""), 220);

  const shouldSurface =
    category !== "none" &&
    !!content &&
    confidence >= MIN_CONFIDENCE_TO_SURFACE;

  const isQuestion = shouldSurface && category === "question";

  return {
    title,
    one_liner: oneLiner,
    question_detected: isQuestion,
    answer: isQuestion ? content : "",
    answer_expanded: isQuestion ? content : "",
    insight: !isQuestion && shouldSurface ? content : "",
    insight_expanded: !isQuestion && shouldSurface ? content : "",
    category: shouldSurface ? category : "insight",
    confidence,
    summary: oneLiner,
    content,
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
  previousBubbles,
}) {
  const bubbleHistory = previousBubbles?.length
    ? previousBubbles.map((b, i) => `${i + 1}. ${b}`).join("\n")
    : "none";

  return `You are a real-time clinical decision-support assistant embedded in smart glasses worn by an attending vascular neurologist.

USER PROFILE
- Attending vascular neurologist
- Expert in stroke care, neurology, and standard protocols
- Does NOT need definitions, teaching, or basic explanations
- Values speed, precision, and high-signal insights only

CONTEXT
You are processing a live clinical conversation (rounds, consults, clinic, ED, stroke alert, teaching, consults).

YOUR JOB
Continuously analyze the conversation and return ONLY high-value output.

You MUST return ONLY valid JSON in this exact format:
{
  "title": "≤4 words",
  "summary": "≤12 words",
  "category": "question | factcheck | insight | reference | none",
  "content": "≤20 words",
  "confidence": 0.0-1.0
}

DECISION LOGIC
1. If a direct clinical question is asked:
   - category = "question"
   - content = direct expert answer

2. If someone says something factually wrong, unsafe, or guideline-deviant:
   - category = "factcheck"
   - content = concise correction

3. If no question was asked:
   - only output category = "insight" if there is something genuinely high-value, such as:
     - missing critical decision data
     - contradiction in discussion
     - non-obvious contraindication or risk
     - subtle guideline nuance
     - high-impact clinical pitfall

4. If a specific dose, protocol detail, or reference value is explicitly requested:
   - category = "reference"

5. If nothing high-value is present:
   - category = "none"
   - content = ""

STRICT QUALITY BAR
- Do NOT define basic terms
- Do NOT teach
- Do NOT repeat obvious facts
- Do NOT repeat prior bubbles
- Do NOT restate the running summary
- If uncertain or low-value, return "none"

STYLE
- Extremely concise
- Expert-level
- Decision-changing only
- Prefer fragments over full sentences
- No fluff

EXAMPLES OF GOOD OUTPUTS
Factcheck:
{"title":"TNK Decision","summary":"Thrombolysis decision underway","category":"factcheck","content":"Prior stroke >3 months alone not TNK contraindication","confidence":0.93}

Insight:
{"title":"LVO Consult","summary":"Possible EVT pathway","category":"insight","content":"No anticoagulation status mentioned before lytic decision","confidence":0.89}

Question:
{"title":"Stroke Alert","summary":"Acute ischemic stroke evaluation","category":"question","content":"CTA before EVT; do not delay if strong LVO suspicion","confidence":0.91}

BAD OUTPUTS
- Definitions
- Generic teaching
- Restating obvious facts
- Repeating earlier bubbles

PREVIOUS BUBBLES (do not repeat):
${bubbleHistory}

PRIOR TITLE:
${priorTitle || "none"}

PRIOR SUMMARY:
${priorSummary || "none"}

FULL TRANSCRIPT WINDOW:
${transcriptWindow || "none"}

MOST RECENT TRANSCRIPT:
${recentTranscript || "none"}

Return ONLY valid JSON. No markdown. No commentary.`;
}

async function analyzeTranscript({
  recentTranscript,
  transcriptWindow,
  priorSummary,
  priorTitle,
  previousBubbles,
}) {
  const prompt = buildPrompt({
    recentTranscript: clamp(recentTranscript, 5000),
    transcriptWindow: clamp(transcriptWindow, 15000),
    priorSummary: clamp(priorSummary || "", 300),
    priorTitle: clamp(priorTitle || "", 60),
    previousBubbles: previousBubbles || [],
  });

  const response = await openai.chat.completions.create({
    model: ANALYZE_MODEL,
    temperature: 0.2,
    response_format: { type: "json_object" },
    messages: [
      { role: "system", content: prompt },
      { role: "user", content: "Analyze the transcript and return JSON." },
    ],
  });

  const outputText = response.choices?.[0]?.message?.content?.trim() || "{}";
  const parsed = parseModelJson(outputText);
  const normalized = normalizeAnalysis(parsed);

  if (!normalized.one_liner) normalized.one_liner = "Conversation in progress";
  if (!normalized.title) normalized.title = "Clinical Copilot";

  normalized.answer = dedupe(normalized.answer, previousBubbles);
  normalized.answer_expanded = normalized.answer;

  normalized.insight = dedupe(normalized.insight, previousBubbles);
  normalized.insight_expanded = normalized.insight;

  if (!normalized.answer && !normalized.insight) {
    normalized.question_detected = false;
  }

  return normalized;
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
      "&utterance_end_ms=3000",
      "&smart_format=false",
      "&punctuate=false",
      "&diarize=false",
      "&endpointing=false",
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
      const previousBubbles = extractBubbleTexts(bubbles);

      const analyzed = await analyzeTranscript({
        recentTranscript: accumulatedText,
        transcriptWindow,
        priorSummary: session.oneLiner,
        priorTitle: session.title,
        previousBubbles,
      });

      if (analyzed.title) session.title = analyzed.title;
      if (analyzed.one_liner) session.oneLiner = analyzed.one_liner;

      lastAnalyzedLen = accumulatedText.length;

      sendJson({
        type: "analysis",
        title: analyzed.title,
        one_liner: analyzed.one_liner,
        question_detected: analyzed.question_detected,
        answer: analyzed.answer || "",
        answer_expanded: analyzed.answer_expanded || "",
        insight: analyzed.insight || "",
        insight_expanded: analyzed.insight_expanded || "",
        category: analyzed.category,
        confidence: analyzed.confidence,
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
      } catch { }
    }
  });

  clientWs.on("error", (err) => {
    console.error("Client WS error:", err.message);
  });
});

// ───────────────────────────────────────────────────────────────────────────────
// HTTP routes
// ───────────────────────────────────────────────────────────────────────────────
app.get("/health", (_req, res) => {
  res.json({
    ok: true,
    service: "clinical-copilot",
    sessions: sessions.size,
  });
});

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

server.listen(port, () => {
  console.log(`Server listening on port ${port}`);
  console.log(`Deepgram: ${DEEPGRAM_API_KEY ? "configured" : "MISSING KEY"}`);
  console.log(
    `OpenAI: ${process.env.OPENAI_API_KEY ? "configured" : "MISSING KEY"}`
  );
  console.log(`Secret token: ${SECRET_TOKEN ? "configured" : "MISSING TOKEN"}`);
});