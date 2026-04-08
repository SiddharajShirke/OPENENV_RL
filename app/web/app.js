const state = {
  sessionId: null,
  taskId: "district_backlog_easy",
  agentPolicy: "backlog_clearance",
  availableAgents: [],
  trace: [],
  running: false,
};

const AGENTS_FALLBACK = ["urgent_first", "oldest_first", "backlog_clearance"];

const els = {
  taskSelect: document.getElementById("taskSelect"),
  agentSelect: document.getElementById("agentSelect"),
  stepsInput: document.getElementById("stepsInput"),
  startRunBtn: document.getElementById("startRunBtn"),
  resetSessionBtn: document.getElementById("resetSessionBtn"),
  statusLine: document.getElementById("statusLine"),
  stepTableBody: document.querySelector("#stepTable tbody"),
  runChart: document.getElementById("runChart"),
  benchTaskSelect: document.getElementById("benchTaskSelect"),
  benchRunsInput: document.getElementById("benchRunsInput"),
  benchStepsInput: document.getElementById("benchStepsInput"),
  runBenchmarkBtn: document.getElementById("runBenchmarkBtn"),
  benchChart: document.getElementById("benchChart"),
  benchTableBody: document.querySelector("#benchTable tbody"),
  kpiReward: document.getElementById("kpiReward"),
  kpiBacklog: document.getElementById("kpiBacklog"),
  kpiCompleted: document.getElementById("kpiCompleted"),
  kpiSla: document.getElementById("kpiSla"),
  kpiFairness: document.getElementById("kpiFairness"),
  kpiScore: document.getElementById("kpiScore"),
};

function setStatus(msg) {
  els.statusLine.textContent = msg;
}

async function api(path, options = {}) {
  const response = await fetch(`/api${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  let payload = null;
  try {
    payload = await response.json();
  } catch (e) {
    payload = null;
  }
  if (!response.ok) {
    const detail = payload && payload.detail ? payload.detail : `${response.status}`;
    throw new Error(`API ${path} failed: ${detail}`);
  }
  return payload;
}

function setLoading(isLoading) {
  state.running = isLoading;
  els.startRunBtn.disabled = isLoading;
  els.resetSessionBtn.disabled = isLoading;
  els.runBenchmarkBtn.disabled = isLoading;
}

function formatFloat(v) {
  return Number(v).toFixed(2);
}

function updateKpis(step) {
  if (!step) return;
  const totalReward = state.trace.reduce((sum, row) => sum + row.reward, 0);
  els.kpiReward.textContent = formatFloat(totalReward);
  els.kpiBacklog.textContent = `${step.backlog}`;
  els.kpiCompleted.textContent = `${step.completed}`;
  els.kpiSla.textContent = `${step.slaBreaches}`;
  els.kpiFairness.textContent = formatFloat(step.fairnessGap);
}

function renderAction(actionObj) {
  if (!actionObj || typeof actionObj !== "object") {
    return "unknown";
  }
  const actionType = actionObj.action_type || "unknown";
  const extras = [];
  if (actionObj.service) extras.push(`service=${actionObj.service}`);
  if (actionObj.target_service) extras.push(`target=${actionObj.target_service}`);
  if (typeof actionObj.officer_delta === "number") extras.push(`delta=${actionObj.officer_delta}`);
  if (actionObj.priority_mode) extras.push(`mode=${actionObj.priority_mode}`);
  return extras.length ? `${actionType} (${extras.join(", ")})` : actionType;
}

function appendStepRow(row) {
  const tr = document.createElement("tr");
  const status = row.done ? "done" : "running";
  tr.innerHTML = `
    <td>${row.step}</td>
    <td>${row.day}</td>
    <td>${row.action}</td>
    <td>${formatFloat(row.reward)}</td>
    <td>${row.backlog}</td>
    <td>${row.completed}</td>
    <td>${row.slaBreaches}</td>
    <td>${status}</td>
  `;
  els.stepTableBody.appendChild(tr);
}

function clearRunView() {
  state.trace = [];
  els.stepTableBody.innerHTML = "";
  els.kpiReward.textContent = "0.00";
  els.kpiBacklog.textContent = "0";
  els.kpiCompleted.textContent = "0";
  els.kpiSla.textContent = "0";
  els.kpiFairness.textContent = "0.00";
  els.kpiScore.textContent = "-";
  drawRunChart([]);
}

function drawAxes(ctx, w, h, pad) {
  ctx.strokeStyle = "#2f2f2f";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(pad, pad);
  ctx.lineTo(pad, h - pad);
  ctx.lineTo(w - pad, h - pad);
  ctx.stroke();
}

function drawSeries(ctx, points, color, pad, w, h, yMax) {
  if (!points.length) return;
  const xStep = (w - pad * 2) / Math.max(points.length - 1, 1);
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.beginPath();
  points.forEach((v, i) => {
    const x = pad + i * xStep;
    const y = h - pad - (v / Math.max(yMax, 1e-6)) * (h - pad * 2);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();
}

function drawRunChart(trace) {
  const canvas = els.runChart;
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;
  const pad = 34;

  ctx.clearRect(0, 0, w, h);
  drawAxes(ctx, w, h, pad);

  if (!trace.length) return;

  const rewards = trace.map((x) => Math.max(0, x.reward));
  const backlogs = trace.map((x) => x.backlog);
  const yMax = Math.max(...rewards, ...backlogs, 1);

  drawSeries(ctx, rewards, "#ffffff", pad, w, h, yMax);
  drawSeries(ctx, backlogs, "#7a7a7a", pad, w, h, yMax);

  ctx.fillStyle = "#d2d2d2";
  ctx.font = "12px Segoe UI";
  ctx.fillText("reward", pad + 6, pad + 8);
  ctx.fillText("backlog", pad + 70, pad + 8);
}

function drawBenchmarkChart(agentResults) {
  const canvas = els.benchChart;
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;
  const pad = 34;

  ctx.clearRect(0, 0, w, h);
  drawAxes(ctx, w, h, pad);

  if (!agentResults.length) return;

  const barAreaW = w - pad * 2;
  const slotW = barAreaW / agentResults.length;

  agentResults.forEach((agent, idx) => {
    const cx = pad + idx * slotW + slotW / 2;
    const barW = Math.max(24, slotW * 0.48);
    const barH = (h - pad * 2) * Math.min(1, Math.max(0, agent.average_score));
    const topY = h - pad - barH;

    ctx.fillStyle = "#ffffff";
    ctx.fillRect(cx - barW / 2, topY, barW, barH);

    ctx.fillStyle = "#9a9a9a";
    agent.runs.forEach((run, runIdx) => {
      const jitter = ((runIdx % 7) - 3) * 2.5;
      const dotY = h - pad - (h - pad * 2) * Math.min(1, Math.max(0, run.score));
      ctx.beginPath();
      ctx.arc(cx + jitter, dotY, 3, 0, Math.PI * 2);
      ctx.fill();
    });

    ctx.fillStyle = "#d0d0d0";
    ctx.font = "11px Segoe UI";
    ctx.textAlign = "center";
    ctx.fillText(agent.agent_policy, cx, h - 10);
  });

  ctx.textAlign = "start";
}

async function resetSession() {
  if (state.sessionId) {
    try {
      await api(`/sessions/${state.sessionId}`, { method: "DELETE" });
    } catch (err) {
      // Ignore stale session cleanup errors; reset will still create a fresh session.
    }
  }

  state.taskId = els.taskSelect.value;
  const payload = await api("/reset", {
    method: "POST",
    body: JSON.stringify({ task_id: state.taskId }),
  });
  state.sessionId = payload.session_id;
  clearRunView();
  setStatus(`Session ready: ${state.sessionId.slice(0, 8)}... (${state.taskId})`);
}

async function runSimulation() {
  const requestedSteps = Number(els.stepsInput.value || 0);
  if (!requestedSteps || requestedSteps < 1) {
    setStatus("Enter a valid step count.");
    return;
  }

  setLoading(true);
  try {
    if (!state.sessionId || state.taskId !== els.taskSelect.value) {
      await resetSession();
    }

    state.agentPolicy = els.agentSelect.value;
    setStatus(`Running ${requestedSteps} steps with ${state.agentPolicy}...`);

    for (let i = 0; i < requestedSteps; i += 1) {
      const stepRes = await api("/autostep", {
        method: "POST",
        body: JSON.stringify({
          session_id: state.sessionId,
          agent_policy: state.agentPolicy,
        }),
      });

      const obs = stepRes.observation;
      const row = {
        step: state.trace.length + 1,
        day: obs.day,
        action: renderAction(stepRes.action),
        reward: Number(stepRes.reward || 0),
        backlog: obs.total_backlog,
        completed: obs.total_completed,
        slaBreaches: obs.total_sla_breaches,
        fairnessGap: Number(obs.fairness_gap || 0),
        done: !!stepRes.done,
      };
      state.trace.push(row);
      appendStepRow(row);
      updateKpis(row);
      drawRunChart(state.trace);

      if (stepRes.done) break;
    }

    const gradeRes = await api("/grade", {
      method: "POST",
      body: JSON.stringify({ session_id: state.sessionId }),
    });
    els.kpiScore.textContent = formatFloat(gradeRes.score);
    setStatus(`Run finished. Score: ${formatFloat(gradeRes.score)} (${gradeRes.grader_name})`);
  } catch (err) {
    setStatus(err.message);
  } finally {
    setLoading(false);
  }
}

async function runBenchmark() {
  setLoading(true);
  try {
    const taskId = els.benchTaskSelect.value;
    const runs = Number(els.benchRunsInput.value || 0);
    const maxSteps = Number(els.benchStepsInput.value || 0);
    if (!runs || !maxSteps) {
      setStatus("Benchmark inputs are invalid.");
      return;
    }

    const benchmarkAgents = state.availableAgents.length ? state.availableAgents : AGENTS_FALLBACK;
    setStatus(`Running benchmark on ${taskId} with ${benchmarkAgents.length} agents...`);

    const res = await api("/benchmark", {
      method: "POST",
      body: JSON.stringify({
        task_id: taskId,
        runs,
        max_steps: maxSteps,
        agent_policies: benchmarkAgents,
      }),
    });

    els.benchTableBody.innerHTML = "";
    res.agent_results.forEach((agent) => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td>${agent.agent_policy}</td>
        <td>${formatFloat(agent.average_score)}</td>
        <td>${formatFloat(agent.min_score)}</td>
        <td>${formatFloat(agent.max_score)}</td>
      `;
      els.benchTableBody.appendChild(tr);
    });
    drawBenchmarkChart(res.agent_results);
    setStatus("Benchmark completed.");
  } catch (err) {
    setStatus(err.message);
  } finally {
    setLoading(false);
  }
}

async function init() {
  setLoading(true);
  try {
    const health = await api("/health");
    const tasksRes = await api("/tasks");
    const agents = await api("/agents").catch(() => AGENTS_FALLBACK);

    tasksRes.tasks.forEach((task) => {
      const optA = new Option(task, task);
      const optB = new Option(task, task);
      els.taskSelect.add(optA);
      els.benchTaskSelect.add(optB);
    });

    state.availableAgents = agents.length ? agents : AGENTS_FALLBACK;
    state.availableAgents.forEach((agent) => {
      els.agentSelect.add(new Option(agent, agent));
    });

    els.taskSelect.value = health.available_tasks.includes("district_backlog_easy")
      ? "district_backlog_easy"
      : tasksRes.tasks[0];
    els.benchTaskSelect.value = els.taskSelect.value;
    els.agentSelect.value = state.availableAgents.includes("backlog_clearance")
      ? "backlog_clearance"
      : state.availableAgents[0];

    await resetSession();
  } catch (err) {
    setStatus(`Initialization failed: ${err.message}`);
  } finally {
    setLoading(false);
  }
}

els.startRunBtn.addEventListener("click", runSimulation);
els.resetSessionBtn.addEventListener("click", async () => {
  setLoading(true);
  try {
    await resetSession();
  } catch (err) {
    setStatus(err.message);
  } finally {
    setLoading(false);
  }
});
els.runBenchmarkBtn.addEventListener("click", runBenchmark);

init();