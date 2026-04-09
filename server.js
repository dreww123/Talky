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
const MAX_HISTORY_TURNS = Number(process.env.MAX_HISTORY_TURNS) || 24;
const SESSION_TTL_MS = 1000 * 60 * 60 * 3;

const ANALYSIS_INTERVAL_MS = Number(process.env.ANALYSIS_INTERVAL_MS) || 5000;
const MIN_NEW_CHARS_FOR_ANALYSIS =
  Number(process.env.MIN_NEW_CHARS_FOR_ANALYSIS) || 35;

const MIN_CONFIDENCE_TO_SURFACE = Number(
  process.env.MIN_CONFIDENCE_TO_SURFACE ?? 0.76
);
const QUESTION_COMMENTARY_CONFIDENCE = Number(
  process.env.QUESTION_COMMENTARY_CONFIDENCE ?? 0.62
);
const FACTCHECK_CONFIDENCE = Number(
  process.env.FACTCHECK_CONFIDENCE ?? 0.62
);
const ANSWER_CONFIDENCE = Number(
  process.env.ANSWER_CONFIDENCE ?? 0.50
);

const BUBBLE_SIMILARITY_THRESHOLD = Number(
  process.env.BUBBLE_SIMILARITY_THRESHOLD ?? 0.66
);

const SECRET_TOKEN = process.env.SECRET_TOKEN;

// Optional comma-separated list in .env, e.g.
// DEEPGRAM_KEYWORDS=eliquis,apixaban,tnk,thalamus,hemianopia,basilar
const DEEPGRAM_KEYWORDS = String(process.env.DEEPGRAM_KEYWORDS || "")
  .split(",")
  .map((s) => s.trim())
  .filter(Boolean);

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
      segmentIndex: 0,
    });
  }

  const session = sessions.get(id);
  session.lastActiveAt = Date.now();
  return session;
}

function resetPatientContext(session) {
  session.turns = [];
  session.oneLiner = "";
  session.title = "Clinical Copilot";
  session.shownBubbles = [];
  session.lastSurfacedExpanded = "";
  session.lastSurfacedLabel = "";
  session.segmentIndex += 1;
  session.lastActiveAt = Date.now();
}

function shouldStartNewPatientSegment(text, session) {
  const raw = normalizeWS(text);
  if (!raw) return false;

  const lower = raw.toLowerCase();

  const strongBoundary =
    /\b(next patient|new patient|moving on|our next case|another patient|next case|room \d+|bed \d+|for the next patient)\b/i.test(
      lower
    );

  if (strongBoundary) return true;

  const softBoundary =
    /^(this is|we have|there is)\s+(an?\s+)?\d{1,3}[- ]year[- ]old\b/i.test(raw) ||
    /^\d{1,3}[- ]year[- ]old\b/i.test(raw);

  if (softBoundary && session.turns.length >= 6) return true;

  return false;
}

function addTurn(session, rawText, analysisText = rawText) {
  const cleanRaw = normalizeWS(rawText);
  const cleanAnalysis = normalizeWS(analysisText);

  if (!cleanRaw || !cleanAnalysis) return;

  if (shouldStartNewPatientSegment(cleanRaw, session)) {
    resetPatientContext(session);
  }

  session.turns.push({
    text: cleanAnalysis,
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
// Middleware
// ───────────────────────────────────────────────────────────────────────────────
app.use(
  cors({
    origin: "*",
    methods: ["GET", "POST", "OPTIONS"],
    allowedHeaders: ["Content-Type", "x-api-key"],
  })
);

app.use(express.json({ limit: "25mb" }));

// Public routes
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

// Auth
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

function looksLikeDirectQuestion(text) {
  const clean = normalizeWS(text);
  if (!clean) return false;

  return (
    /\?/.test(clean) ||
    /\b(do you know|what about|how about|did we|have we|was there|is there|what is|why is|how do|how does|could this|would this|can you explain|do we know)\b/i.test(
      clean
    )
  );
}

// Build a speaker-aware transcript for analysis context when diarization exists
function formatStructuredTranscript(msg) {
  const alt = msg?.channel?.alternatives?.[0];
  const transcript = normalizeWS(alt?.transcript || "");
  const words = alt?.words;

  if (!transcript) return "";
  if (!Array.isArray(words) || !words.length) return transcript;

  let currentSpeaker = words[0].speaker ?? 0;
  let bucket = [];
  const segments = [];

  for (const w of words) {
    const speaker = w.speaker ?? currentSpeaker;
    const token = normalizeWS(w.punctuated_word || w.word || "");
    if (!token) continue;

    if (speaker !== currentSpeaker && bucket.length) {
      segments.push({
        speaker: currentSpeaker,
        text: bucket.join(" "),
      });
      bucket = [];
      currentSpeaker = speaker;
    }

    bucket.push(token);
  }

  if (bucket.length) {
    segments.push({
      speaker: currentSpeaker,
      text: bucket.join(" "),
    });
  }

  if (segments.length <= 1) return transcript;

  return segments
    .map((s) => `Speaker ${s.speaker}: ${normalizeWS(s.text)}`)
    .join("\n");
}

// ───────────────────────────────────────────────────────────────────────────────
// Analysis normalization
// ───────────────────────────────────────────────────────────────────────────────
function normalizeAnalysis(parsed) {
  if (!parsed || typeof parsed !== "object") {
    return {
      title: "",
      one_liner: "",
      bubble_label: "",
      output_type: "none",
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

  const rawType = String(parsed.output_type || "none").toLowerCase().trim();
  const allowedTypes = new Set([
    "question_commentary",
    "insight",
    "factcheck",
    "answer",
    "none",
  ]);
  const outputType = allowedTypes.has(rawType) ? rawType : "none";

  let confidence = 0;
  if (
    typeof parsed.confidence === "number" &&
    Number.isFinite(parsed.confidence)
  ) {
    confidence = Math.max(0, Math.min(1, parsed.confidence));
  }

  const title = clamp(normalizeWS(parsed.title || ""), 56);
  const oneLiner = clamp(normalizeWS(parsed.summary || ""), 1600);
  const bubbleLabel = clamp(normalizeWS(parsed.label || ""), 32);
  const content = clamp(normalizeWS(parsed.content || ""), 2800);

  let minConfidence = MIN_CONFIDENCE_TO_SURFACE;
  if (outputType === "question_commentary") {
    minConfidence = QUESTION_COMMENTARY_CONFIDENCE;
  } else if (outputType === "factcheck") {
    minConfidence = FACTCHECK_CONFIDENCE;
  } else if (outputType === "answer") {
    minConfidence = ANSWER_CONFIDENCE;
  }

  const shouldSurface =
    outputType !== "none" &&
    !!content &&
    confidence >= minConfidence;

  const mappedCategory =
    outputType === "factcheck"
      ? "factcheck"
      : outputType === "insight"
        ? "insight"
        : "question"; // both answers and question/commentary get question icon/lane

  const labelFallback =
    bubbleLabel ||
    title ||
    (outputType === "factcheck"
      ? "Fact Check"
      : outputType === "answer"
        ? "Answer"
        : outputType === "question_commentary"
          ? "Question"
          : outputType === "insight"
            ? "Insight"
            : "");

  return {
    title,
    one_liner: oneLiner,
    bubble_label: labelFallback,
    output_type: outputType,
    question_detected: shouldSurface && outputType === "answer",

    answer: shouldSurface && outputType === "answer" ? labelFallback : "",
    answer_expanded: shouldSurface && outputType === "answer" ? content : "",

    insight:
      shouldSurface && outputType !== "answer" ? labelFallback : "",
    insight_expanded:
      shouldSurface && outputType !== "answer" ? content : "",

    category: shouldSurface ? mappedCategory : "none",
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

  return `You are a real-time clinical copilot embedded in smart glasses worn by an attending vascular neurologist.

USER PROFILE
- Attending vascular neurologist
- Expert in stroke care, neurology, and hospital medicine workflow
- Does NOT need textbook definitions or basic teaching
- Wants sharp, novel, management-relevant thinking in real time

IMPORTANT
- The conversation may be about:
  - active patient cases
  - checkout / handoff across many patients
  - radiology
  - pharmacology
  - microbiology
  - neuroanatomy
  - cortical processing
  - trial data
  - general medical teaching
- Do NOT assume every conversation is about one patient
- Do NOT assume every neurologic symptom is stroke-related
- Time course matters enormously:
  - acute / hyperacute -> vascular may fit
  - chronic progressive -> stroke usually does NOT fit
- Do NOT catastrophize
- If the transcript already contains a benign explanation (fatigue, effort dependence, chronic baseline, pain limitation), do not override it with alarmist stroke language unless new objective facts demand it

PRIMARY ROLE
You are not here to restate what was already said.
You are here to act like a super-doctor in the wearer's field of view:
- identify missing questions to ask
- identify missing labs, imaging, or exam details
- detect contradictions
- add niche expert knowledge when useful
- directly answer questions spoken in the room
- fact-check incorrect or unsafe statements

OUTPUT MODES
You MUST return ONLY valid JSON in this exact format:
{
  "title": "≤5 words",
  "summary": "rolling conversation summary, 1-4 sentences, can be long if the conversation is complex",
  "label": "hyper-abbreviated bubble label, 1-4 words, e.g. Eliquis Trials",
  "output_type": "question_commentary | insight | factcheck | answer | none",
  "content": "2-6 dense expert sentences, 90-260 words, include the non-obvious why and what changes next steps",
  "confidence": 0.0-1.0
}

MODE DEFINITIONS

1. question_commentary
Default high-value mode when no direct spoken question is being answered.
Use this for:
- questions the team should ask next
- missing data to obtain
- commentary like 'this is giving me X vibes; to confirm that we need Y'
- noting that the current narrative does not fit the tempo/localization
Examples:
- 'Any UMN signs?'
- 'What was the total nucleated count? Lymphocyte predominant?'
- 'Two-year pure motor progression argues against stroke tempo; structural or degenerative motor pathway disease fits better.'

2. insight
Use only for niche, high-yield knowledge that adds something not already said:
- trial nuance
- obscure mechanism
- radiology pearl
- unusual syndrome association
- pharmacology nuance

3. factcheck
Use when someone says something clearly incorrect or unsafe.
Be direct and concise, but professional.

4. answer
Use when a direct spoken question is asked in the room, even if it is about anatomy, pharmacology, radiology, micro, or another topic.
These should surface quickly.
Do NOT ignore direct questions just because they are not about an active patient.

5. none
If you do not have a non-obvious, useful contribution, return none.

STRICT QUALITY BAR
- Most cycles should return none
- Never just repeat the plan already discussed
- Never give generic filler like 'MRI would be helpful'
- Never recommend emergent workup solely because an exam is effort-dependent or unreliable
- If the transcript already explains why a finding is probably benign or non-neurologic, prefer that unless there is objective contradiction
- If the tempo is incompatible with stroke, say so directly
- Favor localization, tempo, pathophysiology, and what data is missing
- Prefer novel questions and commentary over passive summary
- label must be ultra-short and glanceable
- summary should actually reflect the evolving conversation, not just say 'Listening'

GOOD EXAMPLES
Bad:
'MRI is necessary to rule out stroke.'
Good:
'Two-year progressive pure motor syndrome is not a stroke tempo. MRI may still help localization, but the vascular frame is a poor fit; think chronic motor pathway disease or structural lesion first.'

Bad:
'Effort dependence makes deterioration a real risk. Emergent workup needed.'
Good:
'Night resident says the change was effort-dependent and not reproducible, which argues against true neurologic worsening. No new objective deficit is described.'

Bad:
'Aspirin is a prothrombin inhibitor.'
Good factcheck:
'Incorrect: aspirin is an antiplatelet via COX-1 inhibition, not a prothrombin inhibitor.'

PREVIOUS BUBBLES (do not repeat):
${bubbleHistory}

PRIOR TITLE:
${priorTitle || "none"}

PRIOR SUMMARY:
${priorSummary || "none"}

FULL CONTEXT WINDOW:
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
    transcriptWindow: clamp(transcriptWindow, 26000),
    priorSummary: clamp(priorSummary || "", 1800),
    priorTitle: clamp(priorTitle || "", 90),
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

    const params = [
      "model=nova-3-medical",
      "language=en-US",
      "encoding=linear16",
      "sample_rate=16000",
      "channels=4",
      "interim_results=false",
      "utterance_end_ms=2300",
      "vad_events=true",
      "punctuate=true",
      "smart_format=true",
      "endpointing=true",
      "diarize=true",
    ];

    for (const kw of DEEPGRAM_KEYWORDS) {
      params.push(`keywords=${encodeURIComponent(kw)}`);
    }

    const url = `wss://api.deepgram.com/v1/listen?${params.join("&")}`;

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

    const newChunk = accumulatedText.slice(lastAnalyzedLen);
    const newChars = newChunk.length;
    const urgentQuestion = looksLikeDirectQuestion(newChunk);

    if (!final && newChars < MIN_NEW_CHARS_FOR_ANALYSIS && !urgentQuestion) {
      return;
    }

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
        analyzed.output_type = "none";
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
        const surfacedLabel =
          analyzed.answer || analyzed.insight || analyzed.bubble_label || "";
        session.lastSurfacedExpanded = surfaced;
        session.lastSurfacedLabel = surfacedLabel;
        rememberSurfacedBubble(session, surfaced);
      }

      sendJson({
        type: "analysis",
        title: analyzed.title,
        one_liner: analyzed.one_liner,
        bubble_label: analyzed.bubble_label || "",
        output_type: analyzed.output_type,
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

      const alt = msg.channel?.alternatives?.[0];
      const transcript = normalizeWS(alt?.transcript || "");
      if (!transcript) return;

      const structuredTranscript = formatStructuredTranscript(msg);

      if (msg.is_final) {
        accumulatedText = normalizeWS(
          accumulatedText ? `${accumulatedText} ${transcript}` : transcript
        );

        addTurn(getSession(sessionId), transcript, structuredTranscript);

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
// HTTP routes
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
  console.log(
    `Deepgram keywords: ${DEEPGRAM_KEYWORDS.length ? DEEPGRAM_KEYWORDS.join(", ") : "none"}`
  );
});
