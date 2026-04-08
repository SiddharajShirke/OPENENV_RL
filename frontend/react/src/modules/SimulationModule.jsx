import { useEffect, useMemo, useState } from "react";
import { api, fmt } from "../api/client";
import { LineChart } from "../components/Charts";

const AGENT_MODES = [
  { value: "baseline_policy", label: "Baseline Policy" },
  { value: "llm_inference", label: "Inference-like LLM Agent" },
  { value: "trained_rl", label: "Trained RL Checkpoint" },
];

export function SimulationModule({
  tasks,
  agents,
  models,
  onStatus,
  defaultTask,
  preferredModelPath,
  preferredModelType,
}) {
  const [taskId, setTaskId] = useState(defaultTask || tasks[0] || "district_backlog_easy");
  const [agentMode, setAgentMode] = useState("baseline_policy");
  const [policyName, setPolicyName] = useState("backlog_clearance");
  const [modelPath, setModelPath] = useState(preferredModelPath || "");
  const [modelType, setModelType] = useState(preferredModelType || "maskable");
  const [maxSteps, setMaxSteps] = useState(80);
  const [seed, setSeed] = useState("");

  const [starting, setStarting] = useState(false);
  const [running, setRunning] = useState(false);
  const [runId, setRunId] = useState("");
  const [trace, setTrace] = useState([]);
  const [logs, setLogs] = useState([]);
  const [result, setResult] = useState(null);
  const [routePlan, setRoutePlan] = useState([]);
  const [historyRuns, setHistoryRuns] = useState([]);

  useEffect(() => {
    if (preferredModelPath) setModelPath(preferredModelPath);
    if (preferredModelType) setModelType(preferredModelType);
  }, [preferredModelPath, preferredModelType]);

  const refreshSimulationHistory = async () => {
    try {
      const res = await api("/history/simulations");
      setHistoryRuns(res.runs || []);
    } catch (_err) {
      setHistoryRuns([]);
    }
  };

  const loadSavedRun = async (row) => {
    try {
      const detail = await api(`/history/simulations/${row.run_id}`);
      setRunId(detail.run_id || row.run_id || "");
      setTaskId(detail.task_id || row.task_id || taskId);
      setAgentMode(detail.agent_mode || row.agent_mode || agentMode);
      setTrace(Array.isArray(detail.trace) ? detail.trace : []);
      setLogs([]);
      setRoutePlan(Array.isArray(detail.route_plan) ? detail.route_plan : detail.summary?.llm_route || []);
      setResult({
        score: detail.score ?? null,
        total_reward: detail.total_reward ?? 0,
        summary: detail.summary ?? null,
        grader_name: detail.grader_name ?? null,
        seed: detail.seed ?? null,
      });
      onStatus(`Loaded saved simulation ${String(detail.run_id || row.run_id || "").slice(0, 8)}.`);
    } catch (err) {
      onStatus(err.message);
    }
  };

  useEffect(() => {
    refreshSimulationHistory();
  }, []);

  const appendLog = (line) => {
    if (!line) return;
    setLogs((prev) => {
      const next = [...prev, line];
      return next.length > 500 ? next.slice(next.length - 500) : next;
    });
  };

  const startSimulation = async () => {
    setStarting(true);
    try {
      const payload = {
        task_id: taskId,
        agent_mode: agentMode,
        max_steps: Number(maxSteps),
        seed: seed.trim() ? Number(seed) : null,
        policy_name: policyName,
        model_path: modelPath || null,
        model_type: modelType,
      };
      const started = await api("/simulation/live/start", {
        method: "POST",
        body: JSON.stringify(payload),
      });
      setRunId(started.run_id);
      setTrace([]);
      setLogs([started.start_log]);
      setRoutePlan(Array.isArray(started.route_plan) ? started.route_plan : []);
      setResult({
        score: null,
        total_reward: 0,
        summary: null,
        grader_name: null,
        seed: started.seed,
      });
      setRunning(true);
      onStatus(`Simulation started (${started.run_id.slice(0, 8)}).`);
    } catch (err) {
      onStatus(err.message);
    } finally {
      setStarting(false);
    }
  };

  const stopSimulation = async () => {
    if (!runId) return;
    try {
      await api(`/simulation/live/${runId}/stop`, { method: "POST" });
      onStatus("Simulation stopped.");
    } catch (err) {
      onStatus(err.message);
    } finally {
      setRunning(false);
    }
  };

  useEffect(() => {
    if (!running || !runId) return undefined;
    let cancelled = false;

    const tick = async () => {
      if (cancelled) return;
      try {
        const res = await api("/simulation/live/step", {
          method: "POST",
          body: JSON.stringify({ run_id: runId }),
        });
        if (cancelled) return;
        if (res.step) {
          setTrace((prev) => [...prev, res.step]);
        }
        appendLog(res.step_log);
        if (res.done) {
          appendLog(res.end_log);
          setResult({
            score: res.score,
            total_reward: res.total_reward,
            summary: res.summary,
            grader_name: res.grader_name,
          });
          setRunning(false);
          onStatus(`Simulation completed. Score=${fmt(res.score, 3)} | Reward=${fmt(res.total_reward, 2)}`);
          refreshSimulationHistory();
          return;
        }
        const delay = agentMode === "llm_inference" ? 1200 : 300;
        setTimeout(tick, delay);
      } catch (err) {
        if (!cancelled) {
          setRunning(false);
          onStatus(err.message);
        }
      }
    };

    tick();
    return () => {
      cancelled = true;
    };
  }, [running, runId, agentMode]);

  const current = trace.length ? trace[trace.length - 1] : null;
  const rewardSeries = useMemo(() => trace.map((t) => Number(t.reward || 0)), [trace]);
  const backlogSeries = useMemo(() => trace.map((t) => Number(t.backlog || 0)), [trace]);

  return (
    <section className="module-grid">
      <article className="panel">
        <h2>Simulation Lab</h2>
        <div className="control-grid">
          <label>
            Task
            <select value={taskId} onChange={(e) => setTaskId(e.target.value)}>
              {tasks.map((task) => (
                <option key={task} value={task}>
                  {task}
                </option>
              ))}
            </select>
          </label>
          <label>
            Agent Mode
            <select value={agentMode} onChange={(e) => setAgentMode(e.target.value)}>
              {AGENT_MODES.map((m) => (
                <option key={m.value} value={m.value}>
                  {m.label}
                </option>
              ))}
            </select>
          </label>
          <label>
            Max Steps
            <input type="number" min={1} max={500} value={maxSteps} onChange={(e) => setMaxSteps(e.target.value)} />
          </label>
          <label>
            Seed (optional)
            <input value={seed} onChange={(e) => setSeed(e.target.value)} placeholder="auto random seed" />
          </label>
          {agentMode === "baseline_policy" ? (
            <label>
              Baseline Policy
              <select value={policyName} onChange={(e) => setPolicyName(e.target.value)}>
                {agents.map((agent) => (
                  <option key={agent} value={agent}>
                    {agent}
                  </option>
                ))}
              </select>
            </label>
          ) : null}
          {agentMode === "trained_rl" ? (
            <>
              <label>
                Model
                <select value={modelPath} onChange={(e) => setModelPath(e.target.value)}>
                  {models
                    .filter((m) => m.exists)
                    .map((m) => (
                      <option key={m.path} value={m.path}>
                        {m.label}
                      </option>
                    ))}
                </select>
              </label>
              <label>
                Model Type
                <select value={modelType} onChange={(e) => setModelType(e.target.value)}>
                  <option value="maskable">maskable</option>
                  <option value="recurrent">recurrent</option>
                </select>
              </label>
            </>
          ) : null}
        </div>
        <div className="row">
          <button onClick={startSimulation} disabled={starting || running}>
            {starting ? "Starting..." : running ? "Running..." : "Start Live Simulation"}
          </button>
          <button className="ghost" onClick={stopSimulation} disabled={!running}>
            Stop
          </button>
        </div>
        {routePlan.length ? (
          <div style={{ marginTop: 12 }}>
            <div className="muted" style={{ marginBottom: 6 }}>
              Active fallback route
            </div>
            <div className="tag-wrap">
              {routePlan.map((route, idx) => (
                <span key={`${route}-${idx}`} className="tag">
                  {idx + 1}. {route}
                </span>
              ))}
            </div>
          </div>
        ) : null}
      </article>

      {trace.length ? (
        <>
          <article className="panel">
            <h3>Live Trajectory</h3>
            <LineChart seriesA={rewardSeries} seriesB={backlogSeries} labelA="Reward" labelB="Backlog" />
          </article>

          <article className="panel">
            <h3>Current Step</h3>
            {current ? (
              <div className="step-card animate-in">
                <div className="step-head">
                  <strong>Step {current.step}</strong>
                  <span>Day {current.day}</span>
                </div>
                <div className="step-meta">
                  <span>Action: {current.action_type}</span>
                  <span>Reward: {fmt(current.reward)}</span>
                  <span>Backlog: {current.backlog}</span>
                  <span>Source: {current.decision_source || "-"}</span>
                  <span>Provider: {current.provider || "-"}</span>
                  <span>Model: {current.model_used || "-"}</span>
                  {current.llm_attempts != null ? <span>LLM Attempts: {current.llm_attempts}</span> : null}
                  {current.repair_note ? <span>Repair: {current.repair_note}</span> : null}
                  {current.llm_error ? <span>Error: {current.llm_error}</span> : null}
                </div>
                <div className="queue-list">
                  {current.queue_rows.map((q) => (
                    <div key={q.service} className="queue-row">
                      <div className="queue-label">{q.service}</div>
                      <div className="queue-bar-wrap">
                        <div className="queue-bar" style={{ width: `${Math.min(100, q.active_cases * 4)}%` }} />
                      </div>
                      <div className="queue-val">{q.active_cases}</div>
                    </div>
                  ))}
                </div>
              </div>
            ) : null}
          </article>

          <article className="panel">
            <h3>Live Logs</h3>
            <pre className="terminal-log">{logs.join("\n")}</pre>
          </article>
        </>
      ) : null}

      {result?.summary ? (
        <article className="panel">
          <h3>Run Summary</h3>
          <div className="metric-grid">
            <div className="metric-card">
              <span>Score</span>
              <strong>{fmt(result.score, 3)}</strong>
            </div>
            <div className="metric-card">
              <span>Total Reward</span>
              <strong>{fmt(result.total_reward, 2)}</strong>
            </div>
            <div className="metric-card">
              <span>Total Steps</span>
              <strong>{result.summary?.total_steps ?? "-"}</strong>
            </div>
            <div className="metric-card">
              <span>Completed</span>
              <strong>{result.summary?.total_completed ?? "-"}</strong>
            </div>
            <div className="metric-card">
              <span>SLA Breaches</span>
              <strong>{result.summary?.total_sla_breaches ?? "-"}</strong>
            </div>
            <div className="metric-card">
              <span>Fairness Gap</span>
              <strong>{fmt(result.summary?.fairness_gap, 3)}</strong>
            </div>
            <div className="metric-card">
              <span>LLM Steps</span>
              <strong>{result.summary?.llm_steps ?? "-"}</strong>
            </div>
            <div className="metric-card">
              <span>Heuristic Fallback Steps</span>
              <strong>{result.summary?.heuristic_fallback_steps ?? "-"}</strong>
            </div>
            <div className="metric-card">
              <span>LLM Repaired Steps</span>
              <strong>{result.summary?.llm_repaired_steps ?? "-"}</strong>
            </div>
          </div>
          {Array.isArray(result.summary?.llm_route) && result.summary.llm_route.length ? (
            <div style={{ marginTop: 12 }}>
              <div className="muted" style={{ marginBottom: 6 }}>
                Final route used
              </div>
              <div className="tag-wrap">
                {result.summary.llm_route.map((route, idx) => (
                  <span key={`${route}-${idx}`} className="tag">
                    {idx + 1}. {route}
                  </span>
                ))}
              </div>
            </div>
          ) : null}
        </article>
      ) : null}

      {historyRuns.length ? (
        <article className="panel">
          <h3>Saved Simulation History</h3>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Task</th>
                  <th>Mode</th>
                  <th>Status</th>
                  <th>Score</th>
                  <th>Reward</th>
                  <th>Steps</th>
                  <th>Load</th>
                </tr>
              </thead>
              <tbody>
                {historyRuns.map((row) => (
                  <tr key={row.run_id}>
                    <td>{row.task_id}</td>
                    <td>{row.agent_mode}</td>
                    <td>{row.status}</td>
                    <td>{fmt(row.score, 3)}</td>
                    <td>{fmt(row.total_reward, 2)}</td>
                    <td>{row.summary?.total_steps ?? row.trace_len ?? "-"}</td>
                    <td>
                      <button
                        className="ghost"
                        onClick={() => loadSavedRun(row)}
                      >
                        Load
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </article>
      ) : null}
    </section>
  );
}
