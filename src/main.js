document.body.innerHTML = `
  <h1>Clinical Copilot</h1>
  <p id="status">Starting...</p>
  <pre id="details"></pre>
`;

async function init() {
  const status = document.getElementById("status");
  const details = document.getElementById("details");

  try {
    const sdk = await import("@evenrealities/even_hub_sdk");
    const bridge = await sdk.waitForEvenAppBridge();

    const STORAGE_KEY = "clinical_copilot_v11_transcript_tail";
    const MAX_BUBBLES = 8;
    const VISIBLE_BUBBLES = 3;
    const SAFE_CHAR_LIMIT = 360;
    const W = 46;
    const TRANSCRIPT_LINES = 3;
    const MIN_AUDIO_BYTES = 2000;
    const RECONNECT_DELAY_MS = 1200;
    const TRANSCRIPT_RENDER_CAP = 2600;
    const TRANSCRIPT_KEEP_TAIL = 1800;

    const TOKEN = "REPLACE_WITH_YOUR_REAL_TOKEN";
    const API_BASE = "https://talky-production-fa31.up.railway.app";
    const WS_BASE = "wss://talky-production-fa31.up.railway.app";

    let isListening = false;
    let isShuttingDown = false;
    let totalBytes = 0;
    let streamSocket = null;

    let oneLiner = "";
    let conversationTitle = "Clinical Copilot";

    let liveTranscript = "";
    let committedTranscript = "";
    let fullTranscript = "";

    let bubbles = [];
    let hoverIndex = "title";
    let expanded = null;
    let expandedSnapshot = null;
    let exitSelection = 1;

    let liveIndicatorOn = false;
    let indicatorInterval = null;
    let reconnectTimer = null;
    let hasAutoStarted = false;
    let lastError = "";

    let pendingUi = {
      title: null,
      oneLiner: null,
      bubbles: [],
    };

    function setStatus(msg) {
      status.textContent = msg;
      lastError = msg;
    }

    function saveState() {
      try {
        localStorage.setItem(
          STORAGE_KEY,
          JSON.stringify({
            conversationTitle,
            oneLiner,
            bubbles: bubbles.slice(0, MAX_BUBBLES),
          })
        );
      } catch {}
    }

    try {
      localStorage.removeItem(STORAGE_KEY);
    } catch {}

    function centerStr(str, width) {
      const s = String(str || "").slice(0, width);
      const total = Math.max(0, width - s.length);
      const left = Math.floor(total / 2);
      const right = total - left;
      return " ".repeat(left) + s + " ".repeat(right);
    }

    function wrapText(text, maxChars) {
      const normalized = String(text || "").replace(/\s+/g, " ").trim();
      if (!normalized) return [""];

      const words = normalized.split(" ");
      const lines = [];
      let cur = "";

      for (const word of words) {
        if (!cur) {
          cur = word;
          continue;
        }

        const candidate = cur + " " + word;
        if (candidate.length <= maxChars) {
          cur = candidate;
        } else {
          lines.push(cur);
          cur = word;
        }
      }

      if (cur) lines.push(cur);
      return lines.length ? lines : [""];
    }

    function normalizeDisplayText(text) {
      return String(text || "").replace(/\s+/g, " ").trim();
    }

    function clampTail(text, maxLen, keepTailLen) {
      const s = normalizeDisplayText(text);
      if (s.length <= maxLen) return s;
      return s.slice(-keepTailLen).trim();
    }

    function tokenize(text) {
      return normalizeDisplayText(text)
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

    function buildTranscriptTail() {
      const combined = normalizeDisplayText(
        [committedTranscript, liveTranscript].filter(Boolean).join(" ")
      );

      if (!combined) {
        return Array(TRANSCRIPT_LINES).fill("");
      }

      const wrapped = wrapText(combined, W);
      const tail = wrapped.slice(-TRANSCRIPT_LINES);

      while (tail.length < TRANSCRIPT_LINES) {
        tail.unshift("");
      }

      return tail;
    }

    const ICONS = {
      question: "?",
      reference: "=",
      factcheck: "!",
      insight: "*",
    };

    function icon(type) {
      return ICONS[type] || "*";
    }

    function abbreviateBubbleLabel(text, type = "insight") {
      const clean = normalizeDisplayText(text);
      if (!clean) {
        if (type === "reference") return "Reference";
        if (type === "question") return "Question";
        if (type === "factcheck") return "Correction";
        return "Insight";
      }

      return clean.slice(0, 28);
    }

    function bubbleExists(expandedText, shortText = "") {
      const full = normalizeDisplayText(expandedText);
      const short = normalizeDisplayText(shortText);

      return bubbles.some((b) => {
        const fullDup =
          full && jaccardSimilarity(full, b.expanded || "") >= 0.66;
        const shortDup =
          short && jaccardSimilarity(short, b.short || "") >= 0.8;
        return fullDup || shortDup;
      });
    }

    function pushBubble(type, short, expandedText = "") {
      const fullExpanded = normalizeDisplayText(expandedText || short);
      const shortLabel = abbreviateBubbleLabel(short || expandedText, type);

      if (!fullExpanded || !shortLabel) return;
      if (bubbleExists(fullExpanded, shortLabel)) return;

      bubbles.unshift({
        type: type || "insight",
        short: shortLabel,
        expanded: fullExpanded,
      });

      if (bubbles.length > MAX_BUBBLES) {
        bubbles = bubbles.slice(0, MAX_BUBBLES);
      }

      saveState();
    }

    function queueBubble(type, short, expandedText = "") {
      const fullExpanded = normalizeDisplayText(expandedText || short);
      const shortLabel = abbreviateBubbleLabel(short || expandedText, type);
      if (!fullExpanded || !shortLabel) return;

      const alreadyQueued = pendingUi.bubbles.some((b) => {
        return (
          jaccardSimilarity(fullExpanded, b.expanded || "") >= 0.66 ||
          jaccardSimilarity(shortLabel, b.short || "") >= 0.8
        );
      });

      if (alreadyQueued || bubbleExists(fullExpanded, shortLabel)) return;

      pendingUi.bubbles.push({
        type,
        short: shortLabel,
        expanded: fullExpanded,
      });
    }

    function applyPendingUi() {
      if (pendingUi.title) conversationTitle = pendingUi.title;
      if (pendingUi.oneLiner) oneLiner = pendingUi.oneLiner;

      for (const b of pendingUi.bubbles) {
        pushBubble(b.type, b.short, b.expanded);
      }

      pendingUi = {
        title: null,
        oneLiner: null,
        bubbles: [],
      };
    }

    function freezeExpandedSnapshot() {
      if (expanded === "title") {
        expandedSnapshot = {
          mode: "title",
          title: conversationTitle,
          summary: oneLiner,
        };
        return;
      }

      if (typeof expanded === "number" && bubbles[expanded]) {
        const b = bubbles[expanded];
        expandedSnapshot = {
          mode: "bubble",
          type: b.type,
          short: b.short,
          expanded: b.expanded,
        };
        return;
      }

      expandedSnapshot = null;
    }

    function buildScreen() {
      const lines = [];
      const flash = liveIndicatorOn ? "*" : " ";

      if (expanded === "exit") {
        lines.push(centerStr("Exit Clinical Copilot?", W));
        lines.push("");
        lines.push(exitSelection === 0 ? "> YES - exit" : "  YES - exit");
        lines.push(exitSelection === 1 ? "> NO  - stay" : "  NO  - stay");
        lines.push("");
        return lines.join("\n");
      }

      if (expanded === "title") {
        const snap = expandedSnapshot || {
          title: conversationTitle,
          summary: oneLiner,
        };

        lines.push(centerStr(snap.title || conversationTitle, W - 1) + " ");
        lines.push("");

        const summaryLines = wrapText(
          snap.summary || "No summary yet.",
          W
        );

        for (const l of summaryLines) lines.push(l);

        return lines.join("\n");
      }

      if (typeof expanded === "number") {
        const b =
          expandedSnapshot?.mode === "bubble"
            ? expandedSnapshot
            : bubbles[expanded];

        if (b) {
          lines.push(centerStr(conversationTitle, W - 1) + " ");
          lines.push("");
          lines.push(("[" + icon(b.type) + "] " + b.short).slice(0, W));
          lines.push("");
          const detailLines = wrapText(b.expanded || b.short, W);
          for (const l of detailLines) lines.push(l);
          lines.push("");
          return lines.join("\n");
        }
      }

      const titleArea = W - 1;
      const titleDisplay = conversationTitle.slice(0, titleArea - 2);

      if (hoverIndex === "title") {
        const centered = centerStr(titleDisplay, titleArea - 2);
        lines.push("> " + centered + flash);
      } else {
        lines.push(
          centerStr(conversationTitle.slice(0, titleArea), titleArea) + flash
        );
      }

      lines.push("");

      if (bubbles.length === 0) {
        for (let i = 0; i < VISIBLE_BUBBLES; i++) lines.push("");
      } else {
        let visibleStart = 0;
        if (typeof hoverIndex === "number") {
          visibleStart =
            hoverIndex < VISIBLE_BUBBLES
              ? 0
              : hoverIndex - VISIBLE_BUBBLES + 1;
        }
        visibleStart = Math.max(
          0,
          Math.min(
            visibleStart,
            Math.max(0, bubbles.length - VISIBLE_BUBBLES)
          )
        );

        const visible = bubbles.slice(
          visibleStart,
          visibleStart + VISIBLE_BUBBLES
        );
        for (let i = 0; i < visible.length; i++) {
          const bubble = visible[i];
          const realIdx = visibleStart + i;
          const prefix = hoverIndex === realIdx ? "> " : "  ";
          const bubbleText = "[" + icon(bubble.type) + "] " + bubble.short;
          lines.push(prefix + bubbleText.slice(0, W - 2));
        }

        while (lines.length < 5) lines.push("");
      }

      lines.push("");

      const transcriptTail = buildTranscriptTail();
      for (const line of transcriptTail) {
        lines.push(line);
      }

      return lines.join("\n");
    }

    async function updateGlassesText(text) {
      const safeText =
        text.length > SAFE_CHAR_LIMIT
          ? text.slice(0, SAFE_CHAR_LIMIT - 3) + "..."
          : text;

      await bridge.textContainerUpgrade({
        containerID: 1,
        containerName: "mainText",
        contentOffset: 0,
        contentLength: safeText.length,
        content: safeText,
      });
    }

    async function render() {
      const text = buildScreen();

      details.textContent =
        `hover=${hoverIndex} expanded=${expanded}\n` +
        `listening=${isListening} bytes=${totalBytes}\n` +
        `title=${conversationTitle}\n` +
        `summary=${oneLiner}\n` +
        `api=${API_BASE}\n` +
        `ws=${WS_BASE}\n` +
        `tokenLoaded=${TOKEN ? "yes" : "no"}\n` +
        `lastError=${lastError}\n` +
        `committed=${committedTranscript}\n` +
        `live=${liveTranscript}\n\n` +
        `display:\n${text}`;

      await updateGlassesText(text);
    }

    function startIndicator() {
      stopIndicator();
      liveIndicatorOn = true;

      indicatorInterval = setInterval(async () => {
        if (!isListening) return;
        if (expanded !== null) return;

        liveIndicatorOn = !liveIndicatorOn;
        try {
          await render();
        } catch {}
      }, 600);
    }

    function stopIndicator() {
      if (indicatorInterval) {
        clearInterval(indicatorInterval);
        indicatorInterval = null;
      }
      liveIndicatorOn = false;
    }

    function queueReconnect() {
      if (isShuttingDown || reconnectTimer || isListening) return;

      reconnectTimer = setTimeout(async () => {
        reconnectTimer = null;
        try {
          await startListening();
        } catch {}
      }, RECONNECT_DELAY_MS);
    }

    function openStream() {
      return new Promise((resolve, reject) => {
        const ws = new WebSocket(
          `${WS_BASE}/stream-audio?token=${encodeURIComponent(TOKEN)}`
        );
        ws.binaryType = "arraybuffer";

        const timeout = setTimeout(() => {
          reject(new Error("WS timeout"));
        }, 8000);

        ws.onopen = () => {
          clearTimeout(timeout);
          resolve(ws);
        };

        ws.onerror = (event) => {
          clearTimeout(timeout);
          console.error("WS error", event);
          reject(new Error("WS failed"));
        };

        ws.onclose = (event) => {
          console.error("WS close", event.code, event.reason);
          streamSocket = null;

          if (!isShuttingDown) {
            isListening = false;
            stopIndicator();
            setStatus(`Reconnecting... WS ${event.code} ${event.reason || ""}`);
            queueReconnect();
          }
        };

        ws.onmessage = async (event) => {
          try {
            const msg = JSON.parse(event.data);

            if (msg.type === "transcript" && msg.text) {
              liveTranscript = normalizeDisplayText(msg.text);
              if (expanded === null) await render();
            }

            if (msg.type === "final_transcript" && msg.text) {
              const finalized = normalizeDisplayText(msg.text);

              committedTranscript = normalizeDisplayText(
                committedTranscript
                  ? `${committedTranscript} ${finalized}`
                  : finalized
              );

              committedTranscript = clampTail(
                committedTranscript,
                TRANSCRIPT_RENDER_CAP,
                TRANSCRIPT_KEEP_TAIL
              );

              fullTranscript =
                normalizeDisplayText(msg.fullText) ||
                committedTranscript ||
                fullTranscript;

              liveTranscript = "";

              if (expanded === null) await render();
            }

            if (msg.type === "analysis") {
              if (expanded !== null) {
                if (msg.title) pendingUi.title = msg.title;
                if (msg.one_liner) pendingUi.oneLiner = msg.one_liner;

                if (msg.answer) {
                  queueBubble(
                    "question",
                    msg.bubble_label || msg.answer,
                    msg.answer_expanded || msg.answer
                  );
                } else if (msg.insight) {
                  queueBubble(
                    msg.category || "insight",
                    msg.bubble_label || msg.insight,
                    msg.insight_expanded || msg.insight
                  );
                }
                return;
              }

              if (msg.title) conversationTitle = msg.title;
              if (msg.one_liner) oneLiner = msg.one_liner;

              if (msg.answer) {
                pushBubble(
                  "question",
                  msg.bubble_label || msg.answer,
                  msg.answer_expanded || msg.answer
                );
              } else if (msg.insight) {
                pushBubble(
                  msg.category || "insight",
                  msg.bubble_label || msg.insight,
                  msg.insight_expanded || msg.insight
                );
              }

              setStatus(isListening ? "Listening" : "Updated");
              await render();
            }

            if (msg.type === "stopped") {
              setStatus("Stopped");
              if (expanded === null) await render();
            }

            if (msg.type === "error") {
              setStatus("ERR: " + String(msg.error || "").slice(0, 50));
              if (expanded === null) await render();
            }
          } catch (err) {
            console.error("WS message error:", err);
          }
        };
      });
    }

    async function startListening() {
      if (isListening || isShuttingDown) return;

      try {
        setStatus("Connecting...");
        await render();

        const clearResp = await fetch(`${API_BASE}/session/clear`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "x-api-key": TOKEN,
          },
          body: JSON.stringify({ sessionId: "default" }),
        });

        if (!clearResp.ok) {
          throw new Error(`session/clear failed: ${clearResp.status}`);
        }

        streamSocket = await openStream();

        if (streamSocket?.readyState === WebSocket.OPEN) {
          streamSocket.send(
            JSON.stringify({
              type: "start",
              sessionId: "default",
            })
          );
        }

        isListening = true;
        totalBytes = 0;
        liveTranscript = "";
        committedTranscript = "";
        fullTranscript = "";

        setStatus("Listening");

        startIndicator();
        await bridge.audioControl(true);
        await render();
      } catch (err) {
        setStatus("Connect failed: " + (err?.message || ""));
        streamSocket = null;
        isListening = false;
        stopIndicator();
        await render();
        queueReconnect();
      }
    }

    async function stopListening({ final = true } = {}) {
      if (!isListening && !streamSocket) return;

      isListening = false;
      stopIndicator();

      try {
        await bridge.audioControl(false);
      } catch {}

      setStatus(final ? "Processing..." : "Stopped");

      if (
        final &&
        streamSocket?.readyState === WebSocket.OPEN &&
        totalBytes >= MIN_AUDIO_BYTES
      ) {
        streamSocket.send(
          JSON.stringify({
            type: "stop",
            sessionId: "default",
            fullTranscript,
            oneLiner,
            bubbles: bubbles.slice(0, MAX_BUBBLES),
          })
        );

        await new Promise((r) => setTimeout(r, 700));
      }

      if (streamSocket) {
        try {
          streamSocket.close();
        } catch {}
        streamSocket = null;
      }

      await render();
    }

    async function shutdownApp() {
      isShuttingDown = true;

      if (reconnectTimer) {
        clearTimeout(reconnectTimer);
        reconnectTimer = null;
      }

      try {
        await stopListening({ final: true });
      } catch {}

      try {
        await bridge.shutDownPageContainer({ exitMode: 0 });
      } catch (err) {
        console.error("Shutdown error:", err);
      }
    }

    bridge.createStartUpPageContainer({
      containerTotalNum: 1,
      textObject: [
        {
          xPosition: 110,
          yPosition: 35,
          width: 820,
          height: 250,
          borderWidth: 0,
          borderColor: 0,
          borderRadius: 0,
          paddingLength: 0,
          containerID: 1,
          containerName: "mainText",
          isEventCapture: 1,
          content: "Starting...",
        },
      ],
      listObject: [],
      imageObject: [],
    });

    setStatus("Launching...");
    await render();

    if (!hasAutoStarted) {
      hasAutoStarted = true;
      await startListening();
    }

    bridge.onEvenHubEvent(async (event) => {
      const sysType = event?.sysEvent?.eventType;
      const textType = event?.textEvent?.eventType;
      const audioPcm = event?.audioEvent?.audioPcm;

      if (
        audioPcm &&
        isListening &&
        streamSocket?.readyState === WebSocket.OPEN
      ) {
        totalBytes += audioPcm?.length || 0;
        streamSocket.send(
          audioPcm instanceof Uint8Array ? audioPcm : new Uint8Array(audioPcm)
        );
        return;
      }

      const isSingleTap =
        (event?.textEvent &&
          (textType === 0 || textType === undefined || textType === null) &&
          !audioPcm) ||
        (event?.sysEvent &&
          (sysType === undefined || sysType === null) &&
          !event?.textEvent &&
          !audioPcm);

      const isDoubleTap = sysType === 3 || textType === 3;
      const isScrollUp = textType === 1;
      const isScrollDown = textType === 2;

      if (isDoubleTap) {
        if (expanded === "title" || typeof expanded === "number") {
          expanded = null;
          expandedSnapshot = null;
          applyPendingUi();
          await render();
          return;
        }

        if (expanded === "exit") {
          expanded = null;
          expandedSnapshot = null;
          await render();
          return;
        }

        expanded = "exit";
        exitSelection = 1;
        await render();
        return;
      }

      if (isScrollUp) {
        if (expanded === "exit") {
          exitSelection = 0;
          await render();
          return;
        }

        if (expanded !== null) return;

        if (typeof hoverIndex === "number" && hoverIndex === 0) {
          hoverIndex = "title";
        } else if (typeof hoverIndex === "number") {
          hoverIndex--;
        } else if (hoverIndex === null) {
          hoverIndex = bubbles.length > 0 ? 0 : "title";
        }

        await render();
        return;
      }

      if (isScrollDown) {
        if (expanded === "exit") {
          exitSelection = 1;
          await render();
          return;
        }

        if (expanded !== null) return;

        if (hoverIndex === "title") {
          hoverIndex = bubbles.length > 0 ? 0 : "title";
        } else if (typeof hoverIndex === "number") {
          if (hoverIndex < bubbles.length - 1) hoverIndex++;
        } else {
          hoverIndex = bubbles.length > 0 ? 0 : "title";
        }

        await render();
        return;
      }

      if (isSingleTap) {
        if (expanded === "exit") {
          if (exitSelection === 0) {
            await shutdownApp();
          } else {
            expanded = null;
            expandedSnapshot = null;
            await render();
          }
          return;
        }

        if (expanded === "title" || typeof expanded === "number") {
          expanded = null;
          expandedSnapshot = null;
          applyPendingUi();
          await render();
          return;
        }

        if (hoverIndex === "title") {
          expanded = "title";
          freezeExpandedSnapshot();
          await render();
          return;
        }

        if (typeof hoverIndex === "number" && bubbles[hoverIndex]) {
          expanded = hoverIndex;
          freezeExpandedSnapshot();
          await render();
          return;
        }

        await render();
      }
    });
  } catch (err) {
    console.error("Error:", err);
    document.getElementById("status").textContent = "Failed.";
    document.getElementById("details").textContent =
      err?.stack || err?.message || String(err);
  }
}

init();