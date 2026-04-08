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

// Faster refresh for live medical conversations
const ANALYSIS_INTERVAL_MS = Number(process.env.ANALYSIS_INTERVAL_MS) || 5000;
const MIN_NEW_CHARS_FOR_ANALYSIS =
  Number(process.env.MIN_NEW_CHARS_FOR_ANALYSIS) || 35;

const MIN_CONFIDENCE_TO_SURFACE = Number(
  process.env.MIN_CONFIDENCE_TO_SURFACE ?? 0.78
);

const QUESTION_CONFIDENCE_TO_SURFACE = Number(
  process.env.QUESTION_CONFIDENCE_TO_SURFACE ?? 0.62
);

const BUBBLE_SIMILARITY_THRESHOLD = Number(
  process.env.BUBBLE_SIMILARITY_THRESHOLD ?? 0.66
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
// Public routes
// ───────────────────────────────────────────────────────────────────────────────
app.get("/", (_req, res) => {
  res.json({
    ok: true,
    service: "clinical-copilot",
    status: "running",
  });
});

app.get("/health", (_req, res) => {
  res.json({
    ok: true,
    service: "clinical-copilot",
    sessions: sessions.size,
  });
});

// ───────────────────────────────────────────────────────────────────────────────
// Auth Middleware
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
      title: "Clinical Copilot",
      shownBubbles: [],
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

function rememberSurfacedBubble(session, text) {
  const clean = normalizeWS(text);
  if (!clean) return;

  session.shownBubbles.unshift(clean);
  if (session.shownBubbles.length > 20) {
    session.shownBubbles = session.shownBubbles.slice(0, 20);
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

function clamp(str, max) {
  const s = String(str || "");
  return s.length <= max ? s : s.slice(0, max);
}

function extractBubbleTexts(bubbles) {
  if (!Array.isArray(bubbles)) return [];
  return bubbles
    .map((b) => normalizeWS(b?.expanded || b?.short || ""))
    .filter(Boolean)
    .slice(0, 20);
}

function tokenize(text) {
  return normalizeWS(text)
    .toLowerCase()
    .replace(/[^\w\s]/g, " ")
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

function isNearDuplicate(text, history) {
  const clean = normalizeWS(text);
  if (!clean) return false;

  for (const h of history || []) {
    if (jaccardSimilarity(clean, h) >= BUBBLE_SIMILARITY_THRESHOLD) {
      return true;
    }
  }

  return false;
}

function dedupe(text, history) {
  const clean = normalizeWS(text);
  if (!clean) return "";

  return isNearDuplicate(clean, history) ? "" : clean;
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
      bubble_label: "",
      question_detected: false,
      answer: "",
      answer_expanded: "",
      insight: "",
      insight_expanded: "",
      category: "none",
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

  const title = clamp(normalizeWS(parsed.title || ""), 48);
  const oneLiner = clamp(normalizeWS(parsed.summary || ""), 700);
  const bubbleLabel = clamp(normalizeWS(parsed.label || ""), 28);
  const content = clamp(normalizeWS(parsed.content || ""), 2200);

  const minConfidence =
    category === "question"
      ? QUESTION_CONFIDENCE_TO_SURFACE
      : MIN_CONFIDENCE_TO_SURFACE;

  const shouldSurface =
    category !== "none" &&
    !!content &&
    confidence >= minConfidence;

  const isQuestion = shouldSurface && category === "question";

  return {
    title,
    one_liner: oneLiner,
    bubble_label: bubbleLabel,
    question_detected: isQuestion,
    answer: isQuestion ? bubbleLabel || title || "Question" : "",
    answer_expanded: isQuestion ? content : "",
    insight: !isQuestion && shouldSurface ? bubbleLabel || title || "Insight" : "",
    insight_expanded: !isQuestion && shouldSurface ? content : "",
    category: shouldSurface ? category : "none",
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
Return ONLY high-value output. Most analysis cycles should return "none".
Do NOT keep reissuing the same insight during a pause. If the best output is the same as a recent bubble, return "none".

You MUST return ONLY valid JSON in this exact format:
{
  "title": "≤5 words",
  "summary": "rolling conversation summary, 1-4 sentences, can be long if the conversation is complex",
  "label": "hyper-abbreviated bubble label, 1-3 words, e.g. Eliquis Trials",
  "category": "question | factcheck | insight | reference | none",
  "content": "2-6 dense expert sentences, 90-260 words, include the non-obvious why, the competing concern, and what changes management",
  "confidence": 0.0-1.0
}

DECISION LOGIC
1. If a direct clinical question is asked:
   - category = "question"
   - content = direct expert answer
   - label = ultra-short answer/topic label

2. If someone says something factually wrong, unsafe, or guideline-deviant:
   - category = "factcheck"
   - content = concise correction plus why it matters
   - label = ultra-short correction label

3. If no question was asked:
   - only output category = "insight" if there is something genuinely management-changing, such as:
     - missing critical decision data
     - contradiction in discussion
     - non-obvious contraindication or risk
     - subtle guideline nuance
     - high-impact clinical pitfall
     - a better-fit diagnosis/workup explanation that changes urgency or priority

4. If a specific dose, protocol detail, or reference value is explicitly requested:
   - category = "reference"

5. If nothing high-value is present:
   - category = "none"
   - label = ""
   - content = ""

STRICT QUALITY BAR
- Never suggest a generic next test unless you explain the specific reason it matters now
- Bad example: "MRI brain would be helpful to rule out stroke"
- Good example: "Positional trigger with ear fullness after trauma pushes vestibular injury above central ischemia. Central ischemia stays on the table if symptoms are persistent, atypical, or paired with focal findings, but MRI is not just 'because stroke' here; it matters only if the bedside story stops fitting a peripheral process."
- Do NOT define basic terms
- Do NOT teach
- Do NOT restate the obvious plan
- Do NOT produce textbook filler
- Do NOT repeat prior bubbles, even with slightly different wording
- If the thought is generic, low-yield, or obvious to a vascular neurologist, return "none"
- Prefer one strong insight over several weak ones
- Expanded content should feel like a concise attending-level reasoning note, not a slogan
- label must be extremely short and glanceable

STYLE
- Expert-to-expert
- High-signal
- Specific
- Non-obvious
- summary should actually capture the main points of the conversation and may be long if needed
- content should include nuance, not just recommendation

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
    recentTranscript: clamp(recentTranscript, 9000),
    transcriptWindow: clamp(transcriptWindow, 22000),
    priorSummary: clamp(priorSummary || "", 1200),
    priorTitle: clamp(priorTitle || "", 80),
    previousBubbles: previousBubbles || [],
  });

  const response = await openai.chat.completions.create({
    model: ANALYZE_MODEL,
    temperature: 0.12,
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
      "&utterance_end_ms=2000",
      "&punctuate=false",
      "&smart_format=false",
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

      const priorHistory = [
        ...extractBubbleTexts(bubbles),
        ...(session.shownBubbles || []),
      ];

      const analyzed = await analyzeTranscript({
        recentTranscript: accumulatedText,
        transcriptWindow,
        priorSummary: session.oneLiner,
        priorTitle: session.title,
        previousBubbles: priorHistory,
      });

      if (analyzed.title) session.title = analyzed.title;
      if (analyzed.one_liner) session.oneLiner = analyzed.one_liner;

      const expandedCandidate =
        analyzed.answer_expanded || analyzed.insight_expanded || "";

      const labelCandidate =
        analyzed.answer || analyzed.insight || analyzed.bubble_label || "";

      const duplicateExpanded =
        !!expandedCandidate &&
        (isNearDuplicate(expandedCandidate, priorHistory) ||
          isNearDuplicate(expandedCandidate, [
            session.lastSurfacedExpanded,
          ]));

      const duplicateLabel =
        !!labelCandidate &&
        isNearDuplicate(labelCandidate, [
          session.lastSurfacedLabel,
        ]);

      if (
        analyzed.category !== "none" &&
        (duplicateExpanded || duplicateLabel)
      ) {
        analyzed.category = "none";
        analyzed.question_detected = false;
        analyzed.answer = "";
        analyzed.answer_expanded = "";
        analyzed.insight = "";
        analyzed.insight_expanded = "";
        analyzed.bubble_label = "";
      }

      lastAnalyzedLen = accumulatedText.length;

      if (analyzed.answer_expanded || analyzed.insight_expanded) {
        const surfaced = analyzed.answer_expanded || analyzed.insight_expanded;
        const surfacedLabel = analyzed.answer || analyzed.insight || analyzed.bubble_label || "";
        session.lastSurfacedExpanded = surfaced;
        session.lastSurfacedLabel = surfacedLabel;
        rememberSurfacedBubble(session, surfaced);
      }

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