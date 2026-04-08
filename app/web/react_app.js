import React, { useEffect, useMemo, useRef, useState } from "https://esm.sh/react@18.3.1";
import { createRoot } from "https://esm.sh/react-dom@18.3.1/client";

async function api(path, options = {}) {
  const res = await fetch(`/api${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  let payload = null;
  try {
    payload = await res.json();
  } catch (err) {
    payload = null;
  }
  if (!res.ok) {
    const detail = payload && payload.detail ? payload.detail : `${res.status}`;
    throw new Error(`API ${path} failed: ${detail}`);
  }
  return payload;
}

function drawAxes(ctx, w, h, pad) {
  ctx.clearRect(0, 0, w, h);
  ctx.strokeStyle = "#2f2f2f";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(pad, pad);
  ctx.lineTo(pad, h - pad);
  ctx.lineTo(w - pad, h - pad);
  ctx.stroke();
}

function LineCanvas({ pointsA, pointsB, labelA, labelB }) {
  const ref = useRef(null);

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const w = canvas.width;
    const h = canvas.height;
    const pad = 34;

    drawAxes(ctx, w, h, pad);

    const all = [...pointsA, ...pointsB];
    if (!all.length) return;

    const yMax = Math.max(...all, 1);
    const draw = (arr, color) => {
      if (!arr.length) return;
      const stepX = (w - pad * 2) / Math.max(arr.length - 1, 1);
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.beginPath();
      arr.forEach((v, i) => {
        const x = pad + i * stepX;
        const y = h - pad - (v / yMax) * (h - pad * 2);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();
    };

    draw(pointsA, "#ffffff");
    draw(pointsB, "#808080");

    ctx.fillStyle = "#d5d5d5";
    ctx.font = "12px Segoe UI";
    ctx.fillText(labelA, pad + 6, pad + 8);
    ctx.fillText(labelB, pad + 92, pad + 8);
  }, [pointsA, pointsB, labelA, labelB]);

  return React.createElement("canvas", { ref, width: 1200, height: 320 });
}

function CompareCanvas({ baselineScore, rlScore }) {
  const ref = useRef(null);

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const w = canvas.width;
    const h = canvas.height;
    const pad = 36;
    drawAxes(ctx, w, h, pad);

    if (baselineScore == null || rlScore == null) return;

    const bars = [
      { name: "baseline", score: baselineScore, color: "#9a9a9a", x: w * 0.35 },
      { name: "phase2", score: rlScore, color: "#ffffff", x: w * 0.65 },
    ];

    bars.forEach((bar) => {
      const barW = 120;
      const barH = (h - pad * 2) * Math.max(0, Math.min(1, bar.score));
      const y = h - pad - barH;
      ctx.fillStyle = bar.color;
      ctx.fillRect(bar.x - barW / 2, y, barW, barH);
      ctx.fillStyle = "#dddddd";
      ctx.font = "13px Segoe UI";
      ctx.textAlign = "center";
      ctx.fillText(`${bar.name}: ${bar.score.toFixed(3)}`, bar.x, h - 10);
    });

    ctx.textAlign = "start";
  }, [baselineScore, rlScore]);

  return React.createElement("canvas", { ref, width: 1200, height: 300 });
}

function formatNumber(value, digits = 2) {
  if (value == null || Number.isNaN(Number(value))) return "-";
  return Number(value).toFixed(digits);
}

function App() {
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState("Initializing...");

  const [tasks, setTasks] = useState([]);
  const [agents, setAgents] = useState([]);
  const [components, setComponents] = useState([]);
  const [models, setModels] = useState([]);

  const [taskId, setTaskId] = useState("district_backlog_easy");
  const [agentPolicy, setAgentPolicy] = useState("backlog_clearance");
  const [steps, setSteps] = useState(40);
  const [sessionId, setSessionId] = useState("");

  const [manualSeed, setManualSeed] = useState("");
  const [manualActionJson, setManualActionJson] = useState('{\n  "action_type": "advance_time"\n}');
  const [manualOutput, setManualOutput] = useState("{}");

  const [baselineTrace, setBaselineTrace] = useState([]);
  const [graderScore, setGraderScore] = useState(null);

  const [benchmarkRows, setBenchmarkRows] = useState([]);

  const [modelPath, setModelPath] = useState("results/best_model/phase2_final.zip");
  const [modelType, setModelType] = useState("maskable");
  const [rlMaxSteps, setRlMaxSteps] = useState(80);
  const [rlRun, setRlRun] = useState(null);
  const [rlEval, setRlEval] = useState([]);

  const [compareData, setCompareData] = useState({ baseline: null, rl: null });
  const [workflowOutput, setWorkflowOutput] = useState("");
  const [workflowMeta, setWorkflowMeta] = useState(null);

  useEffect(() => {
    const init = async () => {
      setLoading(true);
      try {
        const [health, tasksRes, agentsRes, componentsRes, modelsRes] = await Promise.all([
          api("/health"),
          api("/tasks"),
          api("/agents"),
          api("/workflows/components"),
          api("/rl/models"),
        ]);

        const taskList = tasksRes.tasks || [];
        const agentList = agentsRes || [];
        const modelList = (modelsRes.models || []).filter((m) => m.exists);

        setTasks(taskList);
        setAgents(agentList);
        setComponents(componentsRes.components || []);
        setModels(modelsRes.models || []);

        const defaultTask = taskList.includes("district_backlog_easy") ? "district_backlog_easy" : taskList[0];
        setTaskId(defaultTask || "district_backlog_easy");

        const defaultAgent = agentList.includes("backlog_clearance") ? "backlog_clearance" : (agentList[0] || "backlog_clearance");
        setAgentPolicy(defaultAgent);

        const phase2 = modelList.find((m) => m.path.toLowerCase().includes("phase2_final")) || modelList[0];
        if (phase2) {
          setModelPath(phase2.path);
          setModelType(phase2.model_type);
        }

        setStatus(`API ready (v${health.version}).`);
      } catch (err) {
        setStatus(err.message);
      } finally {
        setLoading(false);
      }
    };

    init();
  }, []);

  const baselineRewards = useMemo(() => baselineTrace.map((x) => Math.max(0, Number(x.reward || 0))), [baselineTrace]);
  const baselineBacklog = useMemo(() => baselineTrace.map((x) => Number(x.backlog || 0)), [baselineTrace]);

  const baselineKpi = useMemo(() => {
    const totalReward = baselineTrace.reduce((sum, row) => sum + Number(row.reward || 0), 0);
    const last = baselineTrace.length ? baselineTrace[baselineTrace.length - 1] : null;
    return {
      reward: totalReward,
      backlog: last ? last.backlog : 0,
      completed: last ? last.completed : 0,
      sla: last ? last.sla_breaches : 0,
      fairness: last ? last.fairness_gap : 0,
    };
  }, [baselineTrace]);

  const activeModel = useMemo(() => models.find((m) => m.path === modelPath), [models, modelPath]);

  const manualReset = async () => {
    setLoading(true);
    try {
      const payload = {
        task_id: taskId,
      };
      if (manualSeed.trim()) {
        payload.seed = Number(manualSeed.trim());
      }
      const res = await api("/reset", {
        method: "POST",
        body: JSON.stringify(payload),
      });
      setSessionId(res.session_id);
      setManualOutput(JSON.stringify(res, null, 2));
      setStatus(`Session created: ${res.session_id}`);
    } catch (err) {
      setStatus(err.message);
    } finally {
      setLoading(false);
    }
  };

  const manualStep = async () => {
    if (!sessionId) {
      setStatus("Create a session first with Reset.");
      return;
    }
    setLoading(true);
    try {
      const action = JSON.parse(manualActionJson);
      const res = await api("/step", {
        method: "POST",
        body: JSON.stringify({ session_id: sessionId, action }),
      });
      setManualOutput(JSON.stringify(res, null, 2));
      setStatus(`Manual step done. reward=${formatNumber(res.reward)}`);
    } catch (err) {
      setStatus(err.message);
    } finally {
      setLoading(false);
    }
  };

  const manualState = async () => {
    if (!sessionId) {
      setStatus("Create a session first with Reset.");
      return;
    }
    setLoading(true);
    try {
      const res = await api("/state", {
        method: "POST",
        body: JSON.stringify({ session_id: sessionId, include_action_history: true }),
      });
      setManualOutput(JSON.stringify(res, null, 2));
      setStatus("State fetched.");
    } catch (err) {
      setStatus(err.message);
    } finally {
      setLoading(false);
    }
  };

  const manualGrade = async () => {
    if (!sessionId) {
      setStatus("Create a session first with Reset.");
      return;
    }
    setLoading(true);
    try {
      const res = await api("/grade", {
        method: "POST",
        body: JSON.stringify({ session_id: sessionId }),
      });
      setManualOutput(JSON.stringify(res, null, 2));
      setStatus(`Grade score=${formatNumber(res.score, 3)} (${res.grader_name})`);
    } catch (err) {
      setStatus(err.message);
    } finally {
      setLoading(false);
    }
  };

  const resetBaselineSession = async () => {
    const res = await api("/reset", {
      method: "POST",
      body: JSON.stringify({ task_id: taskId }),
    });
    setSessionId(res.session_id);
    setBaselineTrace([]);
    setGraderScore(null);
    return res.session_id;
  };

  const runBaseline = async () => {
    setLoading(true);
    try {
      let sid = sessionId;
      if (!sid) {
        sid = await resetBaselineSession();
      }

      const rows = [];
      for (let i = 0; i < Number(steps); i += 1) {
        const stepRes = await api("/autostep", {
          method: "POST",
          body: JSON.stringify({ session_id: sid, agent_policy: agentPolicy }),
        });

        rows.push({
          step: rows.length + 1,
          day: stepRes.observation.day,
          action: stepRes.action.action_type,
          reward: Number(stepRes.reward || 0),
          backlog: stepRes.observation.total_backlog,
          completed: stepRes.observation.total_completed,
          sla_breaches: stepRes.observation.total_sla_breaches,
          fairness_gap: Number(stepRes.observation.fairness_gap || 0),
          done: stepRes.done,
        });

        if (stepRes.done) break;
      }

      setBaselineTrace(rows);

      const gradeRes = await api("/grade", {
        method: "POST",
        body: JSON.stringify({ session_id: sid }),
      });
      setGraderScore(Number(gradeRes.score));
      setStatus(`Baseline run done. score=${formatNumber(gradeRes.score, 3)}`);
    } catch (err) {
      setStatus(err.message);
    } finally {
      setLoading(false);
    }
  };

  const runBenchmark = async () => {
    if (!agents.length) {
      setStatus("No baseline agents available.");
      return;
    }
    setLoading(true);
    try {
      const res = await api("/benchmark", {
        method: "POST",
        body: JSON.stringify({
          task_id: taskId,
          runs: 3,
          max_steps: Number(steps),
          agent_policies: agents,
        }),
      });
      setBenchmarkRows(res.agent_results || []);
      setStatus("Baseline benchmark done.");
    } catch (err) {
      setStatus(err.message);
    } finally {
      setLoading(false);
    }
  };

  const runTrainedEpisode = async () => {
    setLoading(true);
    try {
      const res = await api("/rl/run", {
        method: "POST",
        body: JSON.stringify({
          task_id: taskId,
          model_path: modelPath,
          model_type: modelType,
          max_steps: Number(rlMaxSteps),
        }),
      });
      setRlRun(res);
      setStatus(`Trained run done. score=${formatNumber(res.grader_score, 3)} (${res.grader_name})`);
    } catch (err) {
      setStatus(err.message);
    } finally {
      setLoading(false);
    }
  };

  const evaluateTrainedModel = async () => {
    setLoading(true);
    try {
      const res = await api("/rl/evaluate", {
        method: "POST",
        body: JSON.stringify({
          model_path: modelPath,
          model_type: modelType,
          episodes: 3,
          task_ids: tasks,
        }),
      });
      setRlEval(res.results || []);
      setStatus(`Trained evaluation done. avg=${formatNumber(res.average_grader_score, 3)}`);
    } catch (err) {
      setStatus(err.message);
    } finally {
      setLoading(false);
    }
  };

  const compareBaselineVsPhase2 = async () => {
    setLoading(true);
    try {
      const [base, rl] = await Promise.all([
        api("/benchmark", {
          method: "POST",
          body: JSON.stringify({
            task_id: taskId,
            runs: 3,
            max_steps: Number(steps),
            agent_policies: [agentPolicy],
          }),
        }),
        api("/rl/evaluate", {
          method: "POST",
          body: JSON.stringify({
            model_path: modelPath,
            model_type: modelType,
            episodes: 3,
            task_ids: [taskId],
          }),
        }),
      ]);

      const baselineScore = base.agent_results && base.agent_results.length
        ? Number(base.agent_results[0].average_score)
        : null;
      const rlScore = rl.results && rl.results.length
        ? Number(rl.results[0].grader_score)
        : null;

      setCompareData({ baseline: baselineScore, rl: rlScore });
      setStatus("Comparison done.");
    } catch (err) {
      setStatus(err.message);
    } finally {
      setLoading(false);
    }
  };

  const workflowIdForComponent = (componentName) => {
    if (componentName === "baseline_openai.py") return "baseline_openai";
    if (componentName === "inference.py") return "inference";
    if (componentName === "phase2_final.zip") return "phase2_eval";
    return null;
  };

  const runWorkflowFromUi = async (workflowId) => {
    setLoading(true);
    try {
      const payload = {
        workflow_id: workflowId,
        max_steps: Number(steps),
        episodes: 3,
        model_path: modelPath,
        model_type: modelType,
        timeout_seconds: 240,
      };
      const res = await api("/workflows/run", {
        method: "POST",
        body: JSON.stringify(payload),
      });
      setWorkflowMeta({
        workflow_id: res.workflow_id,
        exit_code: res.exit_code,
        duration_seconds: res.duration_seconds,
        timed_out: res.timed_out,
        command: res.command,
      });
      const out = [
        "$ " + (res.command || []).join(" "),
        "",
        "STDOUT:",
        res.stdout || "",
        "",
        "STDERR:",
        res.stderr || "",
      ].join("\n");
      setWorkflowOutput(out);
      setStatus(
        `Workflow ${res.workflow_id} finished. exit_code=${res.exit_code}, duration=${formatNumber(res.duration_seconds, 2)}s`
      );
    } catch (err) {
      setStatus(err.message);
    } finally {
      setLoading(false);
    }
  };

  return React.createElement(
    "div",
    { className: "shell" },
    React.createElement(
      "header",
      { className: "hero" },
      React.createElement("h1", null, "Gov Workflow OpenEnv - React Console"),
      React.createElement(
        "p",
        null,
        "Shows OpenEnv API execution, baseline/inference workflow visibility, and trained Phase 2 RL model behavior from one screen."
      )
    ),

    React.createElement("div", { className: "status" }, status),

    React.createElement(
      "section",
      { className: "panel" },
      React.createElement("h2", null, "Workflow Components Visibility"),
      React.createElement(
        "div",
        { className: "grid cols-2" },
        ...components.map((c) =>
          React.createElement(
            "article",
            { key: c.component, className: "panel" },
            React.createElement("h3", null, c.component),
            React.createElement("div", { className: `badge ${c.available ? "ok" : ""}` }, c.available ? "available" : "missing"),
            React.createElement("p", { className: "small" }, c.description),
            c.command ? React.createElement("pre", null, c.command) : null,
            workflowIdForComponent(c.component)
              ? React.createElement(
                  "div",
                  { className: "btn-row", style: { marginTop: "8px" } },
                  React.createElement(
                    "button",
                    {
                      className: "secondary",
                      onClick: () => runWorkflowFromUi(workflowIdForComponent(c.component)),
                      disabled: loading,
                    },
                    "Run In Frontend"
                  )
                )
              : null,
            c.notes ? React.createElement("p", { className: "small" }, c.notes) : null
          )
        )
      ),
      workflowMeta
        ? React.createElement(
            "div",
            { className: "small", style: { marginTop: "10px" } },
            `Last run: ${workflowMeta.workflow_id} | exit=${workflowMeta.exit_code} | timeout=${workflowMeta.timed_out ? "true" : "false"} | duration=${formatNumber(workflowMeta.duration_seconds, 2)}s`
          )
        : null,
      workflowOutput ? React.createElement("pre", { style: { marginTop: "10px" } }, workflowOutput) : null
    ),

    React.createElement(
      "section",
      { className: "panel" },
      React.createElement("h2", null, "OpenEnv API Runner (step/reset/state/grade)"),
      React.createElement(
        "div",
        { className: "form-row" },
        React.createElement(
          "label",
          null,
          "Task",
          React.createElement(
            "select",
            { value: taskId, onChange: (e) => setTaskId(e.target.value) },
            ...tasks.map((t) => React.createElement("option", { key: t, value: t }, t))
          )
        ),
        React.createElement(
          "label",
          null,
          "Seed (optional)",
          React.createElement("input", {
            value: manualSeed,
            onChange: (e) => setManualSeed(e.target.value),
            placeholder: "11",
          })
        ),
        React.createElement(
          "label",
          null,
          "Session ID",
          React.createElement("input", {
            value: sessionId,
            onChange: (e) => setSessionId(e.target.value),
            placeholder: "auto after reset",
          })
        )
      ),
      React.createElement(
        "label",
        { style: { marginTop: "10px" } },
        "Action JSON for /step",
        React.createElement("textarea", {
          value: manualActionJson,
          onChange: (e) => setManualActionJson(e.target.value),
        })
      ),
      React.createElement(
        "div",
        { className: "btn-row", style: { marginTop: "10px" } },
        React.createElement("button", { onClick: manualReset, disabled: loading }, "Reset"),
        React.createElement("button", { onClick: manualStep, disabled: loading }, "Step"),
        React.createElement("button", { onClick: manualState, disabled: loading }, "State"),
        React.createElement("button", { onClick: manualGrade, disabled: loading }, "Grade"),
      ),
      React.createElement("pre", { style: { marginTop: "10px" } }, manualOutput)
    ),

    React.createElement(
      "section",
      { className: "panel" },
      React.createElement("h2", null, "Baseline Agent Runner (backend policy)"),
      React.createElement(
        "div",
        { className: "form-row" },
        React.createElement(
          "label",
          null,
          "Baseline Agent",
          React.createElement(
            "select",
            { value: agentPolicy, onChange: (e) => setAgentPolicy(e.target.value) },
            ...agents.map((a) => React.createElement("option", { key: a, value: a }, a))
          )
        ),
        React.createElement(
          "label",
          null,
          "Steps",
          React.createElement("input", {
            type: "number",
            min: 1,
            max: 500,
            value: steps,
            onChange: (e) => setSteps(e.target.value),
          })
        )
      ),
      React.createElement(
        "div",
        { className: "btn-row", style: { marginTop: "10px" } },
        React.createElement("button", { onClick: runBaseline, disabled: loading }, "Run Baseline"),
        React.createElement("button", { className: "secondary", onClick: resetBaselineSession, disabled: loading }, "Reset Session"),
        React.createElement("button", { className: "secondary", onClick: runBenchmark, disabled: loading }, "Run Benchmark"),
      ),
      React.createElement(
        "div",
        { className: "kpis", style: { marginTop: "10px" } },
        React.createElement("div", { className: "kpi" }, React.createElement("div", { className: "k" }, "Total Reward"), React.createElement("div", { className: "v" }, formatNumber(baselineKpi.reward))),
        React.createElement("div", { className: "kpi" }, React.createElement("div", { className: "k" }, "Backlog"), React.createElement("div", { className: "v" }, baselineKpi.backlog)),
        React.createElement("div", { className: "kpi" }, React.createElement("div", { className: "k" }, "Completed"), React.createElement("div", { className: "v" }, baselineKpi.completed)),
        React.createElement("div", { className: "kpi" }, React.createElement("div", { className: "k" }, "SLA Breaches"), React.createElement("div", { className: "v" }, baselineKpi.sla)),
        React.createElement("div", { className: "kpi" }, React.createElement("div", { className: "k" }, "Fairness Gap"), React.createElement("div", { className: "v" }, formatNumber(baselineKpi.fairness))),
        React.createElement("div", { className: "kpi" }, React.createElement("div", { className: "k" }, "Grader Score"), React.createElement("div", { className: "v" }, graderScore == null ? "-" : formatNumber(graderScore, 3))),
      ),
      React.createElement("div", { style: { marginTop: "10px" } }, React.createElement(LineCanvas, {
        pointsA: baselineRewards,
        pointsB: baselineBacklog,
        labelA: "reward",
        labelB: "backlog",
      })),
      React.createElement(
        "div",
        { className: "table-wrap", style: { marginTop: "10px" } },
        React.createElement(
          "table",
          null,
          React.createElement(
            "thead",
            null,
            React.createElement(
              "tr",
              null,
              React.createElement("th", null, "Step"),
              React.createElement("th", null, "Day"),
              React.createElement("th", null, "Action"),
              React.createElement("th", null, "Reward"),
              React.createElement("th", null, "Backlog"),
              React.createElement("th", null, "Completed"),
              React.createElement("th", null, "SLA"),
              React.createElement("th", null, "Done")
            )
          ),
          React.createElement(
            "tbody",
            null,
            ...baselineTrace.map((r) =>
              React.createElement(
                "tr",
                { key: `b-${r.step}` },
                React.createElement("td", null, r.step),
                React.createElement("td", null, r.day),
                React.createElement("td", null, r.action),
                React.createElement("td", null, formatNumber(r.reward)),
                React.createElement("td", null, r.backlog),
                React.createElement("td", null, r.completed),
                React.createElement("td", null, r.sla_breaches),
                React.createElement("td", null, r.done ? "true" : "false")
              )
            )
          )
        )
      ),
      benchmarkRows.length
        ? React.createElement(
            "div",
            { className: "table-wrap", style: { marginTop: "10px" } },
            React.createElement(
              "table",
              null,
              React.createElement(
                "thead",
                null,
                React.createElement(
                  "tr",
                  null,
                  React.createElement("th", null, "Agent"),
                  React.createElement("th", null, "Avg Score"),
                  React.createElement("th", null, "Min"),
                  React.createElement("th", null, "Max")
                )
              ),
              React.createElement(
                "tbody",
                null,
                ...benchmarkRows.map((r) =>
                  React.createElement(
                    "tr",
                    { key: `bench-${r.agent_policy}` },
                    React.createElement("td", null, r.agent_policy),
                    React.createElement("td", null, formatNumber(r.average_score, 3)),
                    React.createElement("td", null, formatNumber(r.min_score, 3)),
                    React.createElement("td", null, formatNumber(r.max_score, 3))
                  )
                )
              )
            )
          )
        : null
    ),

    React.createElement(
      "section",
      { className: "panel" },
      React.createElement("h2", null, "Trained RL Model (Phase 2 / Phase 3)"),
      React.createElement(
        "div",
        { className: "form-row" },
        React.createElement(
          "label",
          null,
          "Model",
          React.createElement(
            "select",
            {
              value: modelPath,
              onChange: (e) => {
                const p = e.target.value;
                setModelPath(p);
                const hit = models.find((m) => m.path === p);
                if (hit) setModelType(hit.model_type);
              },
            },
            ...models.filter((m) => m.exists).map((m) =>
              React.createElement("option", { key: m.path, value: m.path }, `${m.label}`)
            )
          )
        ),
        React.createElement(
          "label",
          null,
          "Model Type",
          React.createElement(
            "select",
            { value: modelType, onChange: (e) => setModelType(e.target.value) },
            React.createElement("option", { value: "maskable" }, "maskable"),
            React.createElement("option", { value: "recurrent" }, "recurrent")
          )
        ),
        React.createElement(
          "label",
          null,
          "Max Steps",
          React.createElement("input", {
            type: "number",
            min: 1,
            max: 1000,
            value: rlMaxSteps,
            onChange: (e) => setRlMaxSteps(e.target.value),
          })
        )
      ),
      React.createElement(
        "div",
        { className: "btn-row", style: { marginTop: "10px" } },
        React.createElement("button", { onClick: runTrainedEpisode, disabled: loading }, "Run Trained Episode"),
        React.createElement("button", { className: "secondary", onClick: evaluateTrainedModel, disabled: loading }, "Evaluate Model"),
        React.createElement("button", { className: "secondary", onClick: compareBaselineVsPhase2, disabled: loading }, "Compare vs Baseline"),
      ),
      activeModel
        ? React.createElement("p", { className: "small", style: { marginTop: "10px" } }, `Using: ${activeModel.path}`)
        : null,

      rlRun
        ? React.createElement(
            "div",
            { style: { marginTop: "10px" } },
            React.createElement(
              "div",
              { className: "kpis" },
              React.createElement("div", { className: "kpi" }, React.createElement("div", { className: "k" }, "Task"), React.createElement("div", { className: "v" }, rlRun.task_id)),
              React.createElement("div", { className: "kpi" }, React.createElement("div", { className: "k" }, "Seed"), React.createElement("div", { className: "v" }, rlRun.seed)),
              React.createElement("div", { className: "kpi" }, React.createElement("div", { className: "k" }, "Total Reward"), React.createElement("div", { className: "v" }, formatNumber(rlRun.total_reward))),
              React.createElement("div", { className: "kpi" }, React.createElement("div", { className: "k" }, "Grader Score"), React.createElement("div", { className: "v" }, formatNumber(rlRun.grader_score, 3))),
            ),
            React.createElement("div", { style: { marginTop: "10px" } }, React.createElement(LineCanvas, {
              pointsA: (rlRun.trace || []).map((x) => Math.max(0, Number(x.reward || 0))),
              pointsB: (rlRun.trace || []).map((x) => Number(x.backlog || 0)),
              labelA: "rl reward",
              labelB: "rl backlog",
            })),
            React.createElement(
              "div",
              { className: "table-wrap", style: { marginTop: "10px" } },
              React.createElement(
                "table",
                null,
                React.createElement(
                  "thead",
                  null,
                  React.createElement(
                    "tr",
                    null,
                    React.createElement("th", null, "Step"),
                    React.createElement("th", null, "Action Index"),
                    React.createElement("th", null, "Action"),
                    React.createElement("th", null, "Reward"),
                    React.createElement("th", null, "Backlog"),
                    React.createElement("th", null, "Completed"),
                    React.createElement("th", null, "SLA")
                  )
                ),
                React.createElement(
                  "tbody",
                  null,
                  ...(rlRun.trace || []).map((r) =>
                    React.createElement(
                      "tr",
                      { key: `rl-${r.step}` },
                      React.createElement("td", null, r.step),
                      React.createElement("td", null, r.action_index),
                      React.createElement("td", null, r.action_label),
                      React.createElement("td", null, formatNumber(r.reward)),
                      React.createElement("td", null, r.backlog),
                      React.createElement("td", null, r.completed),
                      React.createElement("td", null, r.sla_breaches)
                    )
                  )
                )
              )
            )
          )
        : null,

      rlEval.length
        ? React.createElement(
            "div",
            { className: "table-wrap", style: { marginTop: "10px" } },
            React.createElement(
              "table",
              null,
              React.createElement(
                "thead",
                null,
                React.createElement(
                  "tr",
                  null,
                  React.createElement("th", null, "Task"),
                  React.createElement("th", null, "Score"),
                  React.createElement("th", null, "Reward"),
                  React.createElement("th", null, "Completed"),
                  React.createElement("th", null, "SLA Breaches")
                )
              ),
              React.createElement(
                "tbody",
                null,
                ...rlEval.map((r) =>
                  React.createElement(
                    "tr",
                    { key: `eval-${r.task_id}` },
                    React.createElement("td", null, r.task_id),
                    React.createElement("td", null, formatNumber(r.grader_score, 3)),
                    React.createElement("td", null, formatNumber(r.total_reward, 2)),
                    React.createElement("td", null, r.total_completed),
                    React.createElement("td", null, r.total_sla_breaches)
                  )
                )
              )
            )
          )
        : null,

      React.createElement("div", { style: { marginTop: "12px" } }, React.createElement(CompareCanvas, {
        baselineScore: compareData.baseline,
        rlScore: compareData.rl,
      }))
    )
  );
}

const rootEl = document.getElementById("app-root");
const root = createRoot(rootEl);
root.render(React.createElement(App));
window.__APP_MOUNTED__ = true;
