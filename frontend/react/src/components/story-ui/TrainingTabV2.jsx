import React, { useEffect, useMemo, useRef, useState } from "react";
import { api, fmt } from "../../api/client";

function backendBaseUrl() {
  if (typeof window === "undefined") return "http://127.0.0.1:7860";
  const host = window.location.hostname;
  const port = window.location.port;
  if ((host === "127.0.0.1" || host === "localhost") && port === "5173") {
    return `http://${host}:7860`;
  }
  return window.location.origin;
}

function normalizePath(path) {
  return String(path || "").replace(/\\/g, "/").toLowerCase();
}

function toNumberOrNull(value) {
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
}

function timestampToDate(value) {
  const n = Number(value);
  if (!Number.isFinite(n) || n <= 0) return null;
  return new Date(n * 1000);
}

function metricRowKV(line) {
  const m = String(line || "").match(/\|\s*([a-zA-Z0-9_ ]+?)\s*\|\s*([-]?\d+(?:\.\d+)?)\s*\|/);
  if (!m) return null;
  return {
    key: String(m[1]).trim().toLowerCase().replace(/\s+/g, "_"),
    value: parseFloat(m[2]),
  };
}

function parseLogMetrics(lines) {
  const rewards = [];
  const scores = [];
  let latestTableReward = null;
  let latestTableScore = null;
  let latestProgressRatio = null;
  let latestLoggedTimesteps = null;

  for (const line of lines || []) {
    if (!line) continue;

    const ratioMatch = line.match(/(\d[\d,]*)\/(\d[\d,]*)/);
    if (ratioMatch) {
      const done = parseInt(String(ratioMatch[1]).replace(/,/g, ""), 10);
      const total = parseInt(String(ratioMatch[2]).replace(/,/g, ""), 10);
      if (Number.isFinite(done) && Number.isFinite(total) && total > 0) {
        latestProgressRatio = done / total;
      }
    }

    const metric = metricRowKV(line);
    if (metric) {
      if (metric.key === "ep_rew_mean" || metric.key === "mean_reward") {
        latestTableReward = metric.value;
      }
      if (metric.key === "grader_score" || metric.key === "avg_grader_score") {
        latestTableScore = metric.value;
      }
      if (metric.key === "total_timesteps") {
        const ts = parseInt(String(metric.value), 10);
        if (Number.isFinite(ts)) {
          latestLoggedTimesteps = ts;
          if (Number.isFinite(latestTableReward)) {
            rewards.push({ t: ts, value: Number(latestTableReward) });
            latestTableReward = null;
          }
          if (Number.isFinite(latestTableScore)) {
            scores.push({ t: ts, value: Number(latestTableScore) });
            latestTableScore = null;
          }
        }
      }
    }

    const evalReward = line.match(/Eval\s+num_timesteps=(\d[\d,]*),\s*episode_reward=([-]?\d+(?:\.\d+)?)/i);
    if (evalReward) {
      const ts = parseInt(String(evalReward[1]).replace(/,/g, ""), 10);
      const rew = parseFloat(evalReward[2]);
      if (Number.isFinite(ts) && Number.isFinite(rew)) {
        latestLoggedTimesteps = ts;
        rewards.push({ t: ts, value: rew });
      }
    }

    const evalScore = line.match(/\[Eval\]\s+Average grader score:\s+([0-9.]+)/i);
    if (evalScore) {
      const score = parseFloat(evalScore[1]);
      if (Number.isFinite(score)) {
        const ts = latestLoggedTimesteps || (scores.length > 0 ? scores[scores.length - 1].t + 1 : 1);
        scores.push({ t: ts, value: score });
      }
    }

    const bestScore = line.match(/\[Eval\]\s+New best(?: recurrent)? grader score:\s+([0-9.]+)/i);
    if (bestScore) {
      const score = parseFloat(bestScore[1]);
      if (Number.isFinite(score)) {
        const ts = latestLoggedTimesteps || (scores.length > 0 ? scores[scores.length - 1].t + 1 : 1);
        scores.push({ t: ts, value: score });
      }
    }
  }

  const dedupe = (rows) => {
    const map = new Map();
    for (const row of rows) {
      if (!Number.isFinite(row.t) || !Number.isFinite(row.value)) continue;
      map.set(row.t, row);
    }
    return Array.from(map.values()).sort((a, b) => a.t - b.t);
  };

  return {
    rewardPoints: dedupe(rewards),
    scorePoints: dedupe(scores),
    logProgressRatio: Number.isFinite(latestProgressRatio) ? latestProgressRatio : null,
    lastLoggedTimesteps: Number.isFinite(latestLoggedTimesteps) ? latestLoggedTimesteps : null,
  };
}

function seriesSpread(rows) {
  if (!Array.isArray(rows) || rows.length === 0) return 0;
  const vals = rows.map((r) => Number(r?.value)).filter(Number.isFinite);
  if (vals.length === 0) return 0;
  return Math.max(...vals) - Math.min(...vals);
}

function payloadHighlights(payload) {
  const src = payload && typeof payload === "object" ? payload : {};
  const keys = [
    "task_id",
    "step",
    "reward",
    "score",
    "done",
    "backlog",
    "completed",
    "total_backlog",
    "total_completed",
    "total_sla_breaches",
    "total_valid",
    "total_actions",
    "passed",
    "action_history_len",
  ];
  const out = [];
  for (const key of keys) {
    if (!(key in src)) continue;
    const value = src[key];
    if (value == null) continue;
    if (typeof value === "number") {
      out.push([key, Number.isFinite(value) ? Number(value).toFixed(Math.abs(value) >= 10 ? 1 : 3) : String(value)]);
    } else {
      out.push([key, String(value)]);
    }
  }
  return out;
}

function toPolyline(points, { minY, maxY, width, height }) {
  if (!points || points.length === 0) return "";
  return points
    .map((p, idx) => {
      const x = (idx / Math.max(points.length - 1, 1)) * width;
      const y = height - ((p.value - minY) / (maxY - minY || 1)) * height;
      return `${x},${y}`;
    })
    .join(" ");
}

function normalizeSeries(points) {
  const map = new Map();
  for (const row of points || []) {
    const t = Number(row?.t);
    const value = Number(row?.value);
    if (!Number.isFinite(t) || !Number.isFinite(value)) continue;
    map.set(t, { t, value });
  }
  return Array.from(map.values()).sort((a, b) => a.t - b.t);
}

function toPolylineByT(points, { minX, maxX, minY, maxY, width, height }) {
  if (!points || points.length === 0) return "";
  const xDen = maxX - minX || 1;
  const yDen = maxY - minY || 1;
  return points
    .map((p) => {
      const x = ((p.t - minX) / xDen) * width;
      const y = height - ((p.value - minY) / yDen) * height;
      return `${x},${y}`;
    })
    .join(" ");
}

function toStairPolylineByT(points, { minX, maxX, minY, maxY, width, height }) {
  if (!points || points.length === 0) return "";
  const xDen = maxX - minX || 1;
  const yDen = maxY - minY || 1;
  const xOf = (t) => ((t - minX) / xDen) * width;
  const yOf = (v) => height - ((v - minY) / yDen) * height;

  const sorted = normalizeSeries(points);
  if (sorted.length === 0) return "";

  const out = [];
  const first = sorted[0];
  out.push(`${xOf(minX)},${yOf(first.value)}`);
  out.push(`${xOf(first.t)},${yOf(first.value)}`);

  for (let i = 1; i < sorted.length; i += 1) {
    const prev = sorted[i - 1];
    const curr = sorted[i];
    const x = xOf(curr.t);
    out.push(`${x},${yOf(prev.value)}`);
    out.push(`${x},${yOf(curr.value)}`);
  }

  const last = sorted[sorted.length - 1];
  out.push(`${xOf(maxX)},${yOf(last.value)}`);
  return out.join(" ");
}

function summarizeLogLine(line) {
  const raw = String(line || "").trim();
  if (!raw) return { title: "Info", text: "Empty line", tone: "slate" };
  const lower = raw.toLowerCase();

  const evalReward = raw.match(/Eval\s+num_timesteps=(\d[\d,]*),\s*episode_reward=([-]?\d+(?:\.\d+)?)/i);
  if (evalReward) {
    const ts = Number(String(evalReward[1]).replace(/,/g, ""));
    const rew = Number(evalReward[2]);
    return {
      title: "Eval Checkpoint",
      text: `Timesteps ${Number.isFinite(ts) ? ts.toLocaleString() : "-"} | Reward ${Number.isFinite(rew) ? rew.toFixed(2) : "-"}`,
      tone: "emerald",
    };
  }

  const bestScore = raw.match(/\[Eval\]\s+New best(?: recurrent)? grader score:\s+([0-9.]+)/i);
  if (bestScore) {
    const score = Number(bestScore[1]);
    return {
      title: "Best Score Improved",
      text: `Grader score improved to ${Number.isFinite(score) ? score.toFixed(4) : "-"}.`,
      tone: "emerald",
    };
  }

  const avgScore = raw.match(/\[Eval\]\s+Average grader score:\s+([0-9.]+)/i);
  if (avgScore) {
    const score = Number(avgScore[1]);
    return {
      title: "Evaluation Summary",
      text: `Average grader score ${Number.isFinite(score) ? score.toFixed(4) : "-"}.`,
      tone: "emerald",
    };
  }

  const metric = metricRowKV(raw);
  if (metric) {
    const key = String(metric.key || "").replace(/_/g, " ");
    return {
      title: "Metric Update",
      text: `${key}: ${Number.isFinite(metric.value) ? metric.value : "-"}`,
      tone: "indigo",
    };
  }

  if (lower.includes("traceback") || lower.includes("exception") || lower.includes("error")) {
    return { title: "Error", text: "A runtime error was reported by the training process. Review backend logs for the exact stack trace.", tone: "rose" };
  }
  if (lower.includes("[eval]")) {
    return { title: "Evaluation", text: "Evaluation cycle completed and scores were updated.", tone: "emerald" };
  }
  if (lower.includes("[training_jobs]")) {
    if (lower.includes("started pid=")) {
      return { title: "Job Started", text: "Training worker started successfully and began consuming timesteps.", tone: "cyan" };
    }
    if (lower.includes("command:")) {
      return { title: "Runtime Config", text: "Training command was prepared with current phase and environment settings.", tone: "cyan" };
    }
    return { title: "System", text: "Background training service published a runtime status update.", tone: "cyan" };
  }
  if (lower.includes("[phase 1]")) {
    return { title: "Phase 1 Update", text: "Phase 1 PPO training is actively optimizing policy behavior.", tone: "indigo" };
  }
  if (lower.includes("[phase 2]")) {
    return { title: "Phase 2 Update", text: "Phase 2 curriculum training is active for harder scenario generalization.", tone: "indigo" };
  }
  if (lower.includes("[costmonitor]")) {
    return { title: "Constraint Monitor", text: "SLA/fairness penalty monitor updated policy constraint feedback.", tone: "amber" };
  }
  return { title: "Runtime Update", text: "The trainer reported a new runtime event and internal state progressed.", tone: "amber" };
}

function summarizeEnvEvent(event) {
  const stage = String(event?.stage || "");
  const payload = event?.payload || {};
  const task = payload?.task_id ? ` [${payload.task_id}]` : "";
  if (stage === "reset") {
    return `Task${task}: session created. Day ${payload?.day ?? "-"}, starting backlog ${payload?.backlog ?? "-"}.`;
  }
  if (stage === "state:initial") {
    return `Task${task}: initial snapshot captured. Completed ${payload?.total_completed ?? "-"}, backlog ${payload?.total_backlog ?? "-"}.`;
  }
  if (stage === "action-masks") {
    return `Task${task}: step ${payload?.step ?? "-"} validated actions (${payload?.total_valid ?? "-"} valid of ${payload?.total_actions ?? "-"}).`;
  }
  if (stage === "auto_step") {
    return `Task${task}: step ${payload?.step ?? "-"} executed. Reward ${fmt(payload?.reward, 3)}, backlog ${payload?.backlog ?? "-"}, completed ${payload?.completed ?? "-"}.`;
  }
  if (stage === "state:post_step") {
    return `Task${task}: post-step state updated. Completed ${payload?.total_completed ?? "-"}, backlog ${payload?.total_backlog ?? "-"}, SLA breaches ${payload?.total_sla_breaches ?? "-"}.`;
  }
  if (stage === "grade") {
    return `Task${task}: grading finished. Score ${fmt(payload?.score, 3)}, pass ${String(payload?.passed)}.`;
  }
  if (stage === "session:closed") {
    return `Task${task}: session closed successfully.`;
  }
  if (stage === "task:error") {
    return `Task${task}: run failed - ${payload?.error || "unknown error"}.`;
  }
  return `Task${task}: ${stage}.`;
}

function workflowStageLabel(stage) {
  const key = String(stage || "").toLowerCase();
  if (key === "reset") return "Reset";
  if (key === "state:initial") return "Initial State";
  if (key === "action-masks") return "Action Validation";
  if (key === "auto_step") return "Auto Step";
  if (key === "state:post_step") return "Post-Step State";
  if (key === "grade") return "Grade";
  if (key === "session:closed") return "Session Closed";
  if (key === "task:error") return "Task Error";
  return stage;
}

function jsonPretty(value) {
  try {
    return JSON.stringify(value, null, 2);
  } catch (_err) {
    return String(value);
  }
}

function toneClasses(tone) {
  if (tone === "rose") return "bg-rose-500/5 border-rose-500/20";
  if (tone === "emerald") return "bg-emerald-500/5 border-emerald-500/20";
  if (tone === "indigo") return "bg-indigo-500/5 border-indigo-500/20";
  if (tone === "cyan") return "bg-cyan-500/5 border-cyan-500/20";
  if (tone === "amber") return "bg-amber-500/5 border-amber-500/20";
  return "bg-slate-700/10 border-slate-500/20";
}

function statusClasses(status) {
  const s = String(status || "").toLowerCase();
  if (s === "running") return "text-emerald-300 bg-emerald-500/10 border-emerald-500/30";
  if (s === "queued") return "text-amber-300 bg-amber-500/10 border-amber-500/30";
  if (s === "completed") return "text-indigo-300 bg-indigo-500/10 border-indigo-500/30";
  if (s === "failed") return "text-rose-300 bg-rose-500/10 border-rose-500/30";
  if (s === "stopped") return "text-slate-300 bg-slate-600/20 border-slate-500/30";
  return "text-slate-300 bg-slate-700/20 border-slate-500/30";
}

function normalizeJob(raw, index) {
  const jobId = String(raw?.job_id || raw?.id || `job-${index}`);
  const status = String(raw?.status || "unknown");
  const timesteps = Number(raw?.timesteps || 0);
  const latestMetrics = raw?.latest_metrics && typeof raw.latest_metrics === "object" ? raw.latest_metrics : {};

  const progressRaw = toNumberOrNull(raw?.progress);
  const ts = toNumberOrNull(latestMetrics.total_timesteps);
  const progressFromMetrics =
    Number.isFinite(ts) && Number.isFinite(timesteps) && timesteps > 0
      ? Math.max(0, Math.min(1, Number(ts) / Number(timesteps)))
      : null;
  const progress = Number.isFinite(progressRaw)
    ? Math.max(0, Math.min(1, Number(progressRaw)))
    : Number.isFinite(progressFromMetrics)
      ? Number(progressFromMetrics)
      : 0;

  return {
    ...raw,
    job_id: jobId,
    status,
    timesteps: Number.isFinite(timesteps) ? timesteps : 0,
    phase: Number(raw?.phase || 0),
    n_envs: Number(raw?.n_envs || 0),
    progress,
    latest_metrics: latestMetrics,
    logs_tail: Array.isArray(raw?.logs_tail) ? raw.logs_tail : [],
    created_at: toNumberOrNull(raw?.created_at),
    updated_at: toNumberOrNull(raw?.updated_at),
  };
}

export function TrainingTabV2({ tasks = [] }) {
  const [endpointRows, setEndpointRows] = useState([]);
  const [endpointError, setEndpointError] = useState("");

  const [agents, setAgents] = useState([]);
  const [modelRows, setModelRows] = useState([]);
  const [modelError, setModelError] = useState("");

  const [jobs, setJobs] = useState([]);
  const [jobsLoading, setJobsLoading] = useState(false);
  const [jobsError, setJobsError] = useState("");
  const [activeJobId, setActiveJobId] = useState("");
  const [activeJob, setActiveJob] = useState(null);
  const [deletingJobId, setDeletingJobId] = useState("");
  const [jobError, setJobError] = useState("");
  const [pollIntervalMs, setPollIntervalMs] = useState(1500);
  const pollFailuresRef = useRef(0);

  const [rewardPoints, setRewardPoints] = useState([]);
  const [scorePoints, setScorePoints] = useState([]);
  const [scoreSignalMeta, setScoreSignalMeta] = useState({
    key: "grader_score",
    label: "Grader Score",
    fallback: false,
  });
  const [logLines, setLogLines] = useState([]);
  const [logProgressRatio, setLogProgressRatio] = useState(null);
  const [lastLoggedTimesteps, setLastLoggedTimesteps] = useState(null);

  const [jobForm, setJobForm] = useState({
    phase: 1,
    timesteps: 80000,
    n_envs: 4,
    seed: "",
  });

  const [envTaskId, setEnvTaskId] = useState(tasks[0] || "district_backlog_easy");
  const [envSeed, setEnvSeed] = useState("");
  const [envPolicyName, setEnvPolicyName] = useState("backlog_clearance");
  const [envMaxSteps, setEnvMaxSteps] = useState(6);
  const [envBusy, setEnvBusy] = useState(false);
  const [envError, setEnvError] = useState("");
  const [envFlowEvents, setEnvFlowEvents] = useState([]);
  const [envFlowSummary, setEnvFlowSummary] = useState(null);
  const [envFlowRuns, setEnvFlowRuns] = useState([]);
  const envEventSeqRef = useRef(0);

  useEffect(() => {
    if (tasks.length > 0 && !tasks.includes(envTaskId)) {
      setEnvTaskId(tasks[0]);
    }
  }, [tasks, envTaskId]);

  useEffect(() => {
    if (agents.length > 0 && !agents.includes(envPolicyName)) {
      setEnvPolicyName(agents[0]);
    }
  }, [agents, envPolicyName]);

  const refreshEndpointHealth = async () => {
    setEndpointError("");

    const directGet = async (path) => {
      const res = await fetch(`${backendBaseUrl()}${path}`, { method: "GET" });
      if (!res.ok) {
        throw new Error(`${path} -> ${res.status}`);
      }
      try {
        return await res.json();
      } catch (_err) {
        return { ok: true };
      }
    };

    const checks = [
      { key: "health", label: "Health", fn: () => api("/health") },
      { key: "tasks", label: "Tasks", fn: () => api("/tasks") },
      { key: "agents", label: "Agents", fn: () => api("/agents") },
      { key: "training_jobs", label: "Training Jobs", fn: () => api("/training_jobs") },
      { key: "actions_schema", label: "Action Schema", fn: () => api("/actions/schema") },
      { key: "rl_models", label: "RL Models", fn: () => api("/rl_models") },
      { key: "rl_models_v2", label: "RL Models V2", fn: () => api("/rl/models") },
      { key: "v1_agents", label: "V1 Agents", fn: () => directGet("/api/v1/agents") },
      { key: "v1_rl_models", label: "V1 RL Models", fn: () => directGet("/api/v1/rl_models") },
    ];

    const settled = await Promise.allSettled(
      checks.map(async (chk) => {
        const start = Date.now();
        await chk.fn();
        return { key: chk.key, label: chk.label, ok: true, ms: Date.now() - start };
      })
    );

    const rows = settled.map((res, idx) => {
      const meta = checks[idx];
      if (res.status === "fulfilled") return res.value;
      return {
        key: meta.key,
        label: meta.label,
        ok: false,
        ms: null,
        error: res.reason?.message || String(res.reason),
      };
    });

    setEndpointRows(rows);
    if (rows.some((r) => !r.ok)) {
      setEndpointError("Some endpoints are down. Retries remain active.");
    }
  };

  const refreshCatalog = async () => {
    setModelError("");
    try {
      const [agentRes, rlV1Res, rlV2Res] = await Promise.allSettled([
        api("/agents"),
        api("/rl_models"),
        api("/rl/models"),
      ]);

      if (agentRes.status === "fulfilled") {
        setAgents(Array.isArray(agentRes.value) ? agentRes.value : []);
      }

      const unified = [];
      if (rlV1Res.status === "fulfilled") {
        const rows = Array.isArray(rlV1Res.value?.models) ? rlV1Res.value.models : [];
        for (const row of rows) {
          unified.push({
            source: "api/rl_models",
            label: row.label || row.path || "unnamed",
            path: row.path || "",
            exists: Boolean(row.exists),
            phase: normalizePath(row.path).includes("/phase2/") ? 2 : normalizePath(row.path).includes("/phase1/") ? 1 : 0,
          });
        }
      }
      if (rlV2Res.status === "fulfilled") {
        const rows = Array.isArray(rlV2Res.value) ? rlV2Res.value : [];
        for (const row of rows) {
          const path = row.model_path
            ? (String(row.model_path).toLowerCase().endsWith(".zip") ? row.model_path : `${row.model_path}.zip`)
            : "";
          unified.push({
            source: "api/rl/models",
            label: path.split(/[\\/]/).pop() || row.model_path || "unnamed",
            path,
            exists: Boolean(row.exists),
            phase: Number(row.phase || 0),
          });
        }
      }

      const dedupe = new Map();
      for (const row of unified) {
        const key = normalizePath(row.path);
        if (!key) continue;
        if (!dedupe.has(key)) dedupe.set(key, row);
      }
      const rows = Array.from(dedupe.values()).sort((a, b) => {
        if (a.phase !== b.phase) return b.phase - a.phase;
        return String(a.label).localeCompare(String(b.label));
      });
      setModelRows(rows);
      if (rows.length === 0) {
        setModelError("No models discovered from dynamic model endpoints.");
      }
    } catch (err) {
      setModelError(err?.message || "Failed to load model registry.");
    }
  };

  const refreshJobs = async () => {
    setJobsLoading(true);
    try {
      const data = await api("/training_jobs");
      const rowsRaw = Array.isArray(data?.jobs) ? data.jobs : [];
      const rows = rowsRaw.map(normalizeJob).sort((a, b) => Number(b.created_at || 0) - Number(a.created_at || 0));
      setJobs(rows);
      setJobsError("");

      const running = rows.find((j) => j.status === "running" || j.status === "queued");
      const current = rows.find((j) => j.job_id === activeJobId);

      if (running?.job_id) {
        if (!current || (current.status !== "running" && current.status !== "queued")) {
          setActiveJobId(running.job_id);
        }
      } else if (!activeJobId && rows[0]?.job_id) {
        setActiveJobId(rows[0].job_id);
      }
    } catch (err) {
      setJobsError(err?.message || "Failed to load training jobs.");
    } finally {
      setJobsLoading(false);
    }
  };

  const parseAndSetPoints = (jobSnapshot) => {
    const lines = Array.isArray(jobSnapshot?.logs_tail) ? jobSnapshot.logs_tail : [];
    setLogLines(lines);

    const parsed = parseLogMetrics(lines);
    setLogProgressRatio(parsed.logProgressRatio);
    setLastLoggedTimesteps(parsed.lastLoggedTimesteps);

    const nextRewards = [];
    const nextScores = [];
    const nextSignals = {
      explained_variance: [],
      ep_len_mean: [],
      approx_kl: [],
    };

    const history = Array.isArray(jobSnapshot?.metric_history) ? jobSnapshot.metric_history : [];
    for (const row of history) {
      const t = Number(row?.t ?? row?.total_timesteps ?? NaN);
      if (!Number.isFinite(t)) continue;
      const rew = Number(row?.ep_rew_mean ?? row?.mean_reward ?? NaN);
      const score = Number(row?.grader_score ?? row?.avg_grader_score ?? NaN);
      if (Number.isFinite(rew)) nextRewards.push({ t, value: rew });
      if (Number.isFinite(score)) nextScores.push({ t, value: score });
      for (const key of Object.keys(nextSignals)) {
        const vv = Number(row?.[key] ?? NaN);
        if (Number.isFinite(vv)) nextSignals[key].push({ t, value: vv });
      }
    }
    nextRewards.push(...parsed.rewardPoints);
    nextScores.push(...parsed.scorePoints);

    const lm = jobSnapshot?.latest_metrics || {};
    const metricTs = Number(lm.total_timesteps ?? NaN);
    const metricReward = Number(lm.ep_rew_mean ?? lm.mean_reward ?? NaN);
    const metricScore = Number(lm.grader_score ?? lm.avg_grader_score ?? NaN);

    if (Number.isFinite(metricTs) && Number.isFinite(metricReward)) {
      nextRewards.push({ t: metricTs, value: metricReward });
    }
    if (Number.isFinite(metricTs) && Number.isFinite(metricScore)) {
      nextScores.push({ t: metricTs, value: metricScore });
    }
    for (const key of Object.keys(nextSignals)) {
      const vv = Number(lm[key] ?? NaN);
      if (Number.isFinite(metricTs) && Number.isFinite(vv)) {
        nextSignals[key].push({ t: metricTs, value: vv });
      }
    }

    const dedupe = (rows) => {
      const map = new Map();
      for (const row of rows) {
        if (!Number.isFinite(row.t) || !Number.isFinite(row.value)) continue;
        map.set(row.t, row);
      }
      return Array.from(map.values()).sort((a, b) => a.t - b.t);
    };

    const dedupedRewards = dedupe(nextRewards);
    const dedupedScores = dedupe(nextScores);
    const dedupedSignals = Object.fromEntries(
      Object.entries(nextSignals).map(([key, rows]) => [key, dedupe(rows)])
    );

    let chosenScores = dedupedScores;
    let chosenMeta = { key: "grader_score", label: "Grader Score", fallback: false };

    if (dedupedScores.length < 2 || seriesSpread(dedupedScores) < 1e-6) {
      const fallbackCandidates = [
        { key: "explained_variance", label: "Explained Variance" },
        { key: "ep_len_mean", label: "Episode Length Mean" },
        { key: "approx_kl", label: "Approx KL" },
      ];
      for (const candidate of fallbackCandidates) {
        const rows = dedupedSignals[candidate.key] || [];
        if (rows.length >= 2 && seriesSpread(rows) >= 1e-6) {
          chosenScores = rows;
          chosenMeta = { key: candidate.key, label: candidate.label, fallback: true };
          break;
        }
      }
    }

    setRewardPoints(dedupedRewards);
    setScorePoints(chosenScores);
    setScoreSignalMeta(chosenMeta);
  };

  const startTrainingJob = async () => {
    setJobError("");
    try {
      const payload = {
        phase: Number(jobForm.phase) || 1,
        timesteps: Number(jobForm.timesteps) || 80000,
        n_envs: Number(jobForm.n_envs) || 4,
      };
      const seedNum = Number(jobForm.seed);
      if (jobForm.seed !== "" && Number.isFinite(seedNum)) payload.seed = seedNum;

      const res = await api("/training_jobs", {
        method: "POST",
        body: JSON.stringify(payload),
      });
      if (res?.job_id) {
        setActiveJobId(res.job_id);
        const norm = normalizeJob(res, 0);
        setActiveJob(norm);
        parseAndSetPoints(norm);
      }
      await refreshJobs();
    } catch (err) {
      setJobError(err?.message || "Failed to start training job.");
    }
  };

  const stopTrainingJob = async () => {
    if (!activeJobId) return;
    setJobError("");
    try {
      await api(`/training_jobs/${activeJobId}/stop`, { method: "POST" });
      await refreshJobs();
      const stopped = await api(`/training_jobs/${activeJobId}`);
      const norm = normalizeJob(stopped, 0);
      setActiveJob(norm);
      parseAndSetPoints(norm);
    } catch (err) {
      setJobError(err?.message || "Failed to stop training job.");
    }
  };

  const clearTrainingHistory = async () => {
    setJobError("");
    try {
      await api("/training_jobs?clear_artifacts=false", { method: "DELETE" });
      setJobs([]);
      setActiveJob(null);
      setActiveJobId("");
      setRewardPoints([]);
      setScorePoints([]);
      setScoreSignalMeta({ key: "grader_score", label: "Grader Score", fallback: false });
      setLogLines([]);
      setLogProgressRatio(null);
      setLastLoggedTimesteps(null);
    } catch (err) {
      setJobError(err?.message || "Failed to clear training history.");
    }
  };

  const deleteTrainingJob = async (jobId) => {
    if (!jobId) return;
    setJobError("");
    setDeletingJobId(jobId);
    try {
      await api(`/training_jobs/${jobId}?clear_artifacts=false`, { method: "DELETE" });
      if (activeJobId === jobId) {
        setActiveJobId("");
        setActiveJob(null);
        setRewardPoints([]);
        setScorePoints([]);
        setScoreSignalMeta({ key: "grader_score", label: "Grader Score", fallback: false });
        setLogLines([]);
      }
      await refreshJobs();
    } catch (err) {
      setJobError(err?.message || "Failed to delete training job.");
    } finally {
      setDeletingJobId("");
    }
  };

  const pushEnvEvent = (stage, payload, tone = "indigo") => {
    const seq = envEventSeqRef.current + 1;
    envEventSeqRef.current = seq;
    setEnvFlowEvents((prev) => [
      ...prev,
      { id: `${Date.now()}-${Math.random()}`, seq, ts: Date.now(), stage, payload, tone },
    ].slice(-400));
  };

  const runAutomatedOpenEnvFlow = async () => {
    setEnvBusy(true);
    setEnvError("");
    setEnvFlowSummary(null);
    setEnvFlowEvents([]);
    setEnvFlowRuns([]);
    envEventSeqRef.current = 0;

    try {
      const seedNum = Number(envSeed);
      const taskScope = Array.isArray(tasks) && tasks.length > 0 ? tasks : [envTaskId];
      const runTaskIds = Array.from(new Set(taskScope.filter(Boolean)));
      const maxSteps = Math.max(1, Number(envMaxSteps) || 6);
      const taskResults = [];

      for (const taskId of runTaskIds) {
        let sessionId = "";
        let stepsExecuted = 0;
        let finalState = null;
        try {
          const resetPayload = { task_id: taskId };
          if (envSeed !== "" && Number.isFinite(seedNum)) {
            resetPayload.seed = seedNum;
          }

          const resetRes = await api("/reset", {
            method: "POST",
            body: JSON.stringify(resetPayload),
          });
          sessionId = String(resetRes?.session_id || "");
          if (!sessionId) throw new Error(`reset() did not return session_id for task ${taskId}`);

          pushEnvEvent(
            "reset",
            {
              task_id: taskId,
              day: resetRes?.observation?.day,
              backlog: resetRes?.observation?.total_backlog,
              completed: resetRes?.observation?.total_completed,
            },
            "emerald"
          );

          const initialState = await api("/state", {
            method: "POST",
            body: JSON.stringify({ session_id: sessionId, include_action_history: false }),
          });
          pushEnvEvent(
            "state:initial",
            {
              task_id: taskId,
              total_completed: initialState?.state?.total_completed,
              total_backlog: initialState?.state?.total_backlog,
              fairness_gap: initialState?.state?.fairness_gap,
            },
            "cyan"
          );

          let done = false;
          for (let idx = 0; idx < maxSteps; idx += 1) {
            if (done) break;

            const masks = await api("/action-masks", {
              method: "POST",
              body: JSON.stringify({ session_id: sessionId }),
            });
            pushEnvEvent(
              "action-masks",
              {
                task_id: taskId,
                step: idx + 1,
                total_valid: masks?.total_valid,
                total_actions: masks?.total_actions,
              },
              "amber"
            );

            const stepRes = await api("/auto_step", {
              method: "POST",
              body: JSON.stringify({
                session_id: sessionId,
                agent_policy: envPolicyName || "backlog_clearance",
              }),
            });
            done = Boolean(stepRes?.done);
            stepsExecuted += 1;
            pushEnvEvent(
              "auto_step",
              {
                task_id: taskId,
                step: idx + 1,
                reward: stepRes?.reward,
                done: stepRes?.done,
                day: stepRes?.observation?.day,
                backlog: stepRes?.observation?.total_backlog,
                completed: stepRes?.observation?.total_completed,
              },
              "indigo"
            );

            const stateRes = await api("/state", {
              method: "POST",
              body: JSON.stringify({ session_id: sessionId, include_action_history: true }),
            });
            finalState = stateRes;
            pushEnvEvent(
              "state:post_step",
              {
                task_id: taskId,
                step: idx + 1,
                total_completed: stateRes?.state?.total_completed,
                total_backlog: stateRes?.state?.total_backlog,
                total_sla_breaches: stateRes?.state?.total_sla_breaches,
                action_history_len: Array.isArray(stateRes?.state?.action_history) ? stateRes.state.action_history.length : 0,
              },
              "cyan"
            );
          }

          const gradeRes = await api("/grade", {
            method: "POST",
            body: JSON.stringify({ session_id: sessionId }),
          });
          const scoreValue = Number(gradeRes?.score);
          const dynamicPassed =
            typeof gradeRes?.passed === "boolean"
              ? gradeRes.passed
              : (Number.isFinite(scoreValue) ? scoreValue >= 0.5 : null);
          pushEnvEvent(
            "grade",
            {
              task_id: taskId,
              score: gradeRes?.score,
              passed: dynamicPassed,
            },
            "emerald"
          );

          taskResults.push({
            task_id: taskId,
            steps_executed: stepsExecuted,
            score: gradeRes?.score ?? null,
            passed: dynamicPassed,
            final_completed: finalState?.state?.total_completed ?? null,
            final_backlog: finalState?.state?.total_backlog ?? null,
            final_sla_breaches: finalState?.state?.total_sla_breaches ?? null,
          });
        } catch (taskErr) {
          const msg = taskErr?.message || String(taskErr);
          pushEnvEvent("task:error", { task_id: taskId, error: msg }, "rose");
          taskResults.push({
            task_id: taskId,
            steps_executed: stepsExecuted,
            score: null,
            passed: null,
            error: msg,
          });
        } finally {
          if (sessionId) {
            try {
              await api(`/sessions/${sessionId}`, { method: "DELETE" });
              pushEnvEvent("session:closed", { task_id: taskId }, "slate");
            } catch (_err) {
              // no-op
            }
          }
        }
      }

      setEnvFlowRuns(taskResults);
      const validScores = taskResults
        .map((row) => Number(row.score))
        .filter((v) => Number.isFinite(v));
      const passedCount = taskResults.filter((row) => row.passed === true).length;
      setEnvFlowSummary({
        tasks_executed: taskResults.length,
        total_steps_executed: taskResults.reduce((acc, row) => acc + Number(row.steps_executed || 0), 0),
        avg_score:
          validScores.length > 0
            ? validScores.reduce((acc, score) => acc + Number(score), 0) / validScores.length
            : null,
        passed_tasks: passedCount,
      });
    } catch (err) {
      setEnvError(err?.message || "Automated OpenEnv workflow failed.");
    } finally {
      setEnvBusy(false);
    }
  };

  useEffect(() => {
    refreshEndpointHealth();
    refreshCatalog();
    refreshJobs();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    const t = setInterval(() => {
      refreshJobs();
    }, 5000);
    return () => clearInterval(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    const t = setInterval(() => {
      refreshEndpointHealth();
    }, 15000);
    return () => clearInterval(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (!activeJobId) return undefined;
    let cancelled = false;

    const t = setInterval(async () => {
      if (cancelled) return;
      try {
        const snapshotRaw = await api(`/training_jobs/${activeJobId}`);
        if (cancelled) return;
        const snapshot = normalizeJob(snapshotRaw, 0);
        setActiveJob(snapshot);
        parseAndSetPoints(snapshot);
        setJobError("");
        pollFailuresRef.current = 0;
        if (pollIntervalMs !== 1500) setPollIntervalMs(1500);
      } catch (err) {
        pollFailuresRef.current += 1;
        if (pollFailuresRef.current >= 3) {
          setPollIntervalMs(4000);
          setJobError(err?.message || "Polling failed repeatedly, switched to fallback polling.");
        }
      }
    }, pollIntervalMs);

    return () => {
      cancelled = true;
      clearInterval(t);
    };
  }, [activeJobId, pollIntervalMs]);

  useEffect(() => {
    if (!activeJobId) return;
    const row = jobs.find((j) => j.job_id === activeJobId);
    if (!row) return;
    setActiveJob(row);
    parseAndSetPoints(row);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeJobId, jobs]);

  const progressA = useMemo(() => {
    if (!activeJob) return null;
    const p = toNumberOrNull(activeJob.progress);
    return Number.isFinite(p) ? Math.max(0, Math.min(1, Number(p))) : null;
  }, [activeJob]);

  const progressB = useMemo(() => {
    if (!activeJob) return null;
    const history = Array.isArray(activeJob?.metric_history) ? activeJob.metric_history : [];
    const historyTs = history.length > 0 ? toNumberOrNull(history[history.length - 1]?.t ?? history[history.length - 1]?.total_timesteps) : null;
    const ts = toNumberOrNull(activeJob?.latest_metrics?.total_timesteps) ?? historyTs;
    const total = toNumberOrNull(activeJob?.timesteps);
    if (!Number.isFinite(ts) || !Number.isFinite(total) || total <= 0) return null;
    return Math.max(0, Math.min(1, Number(ts) / Number(total)));
  }, [activeJob]);

  const progressC = useMemo(() => {
    if (!activeJob) return null;
    const total = toNumberOrNull(activeJob?.timesteps);
    if (!Number.isFinite(total) || total <= 0) {
      return Number.isFinite(logProgressRatio) ? Number(logProgressRatio) : null;
    }

    const fromLogTs =
      Number.isFinite(lastLoggedTimesteps) && Number(lastLoggedTimesteps) > 0
        ? Math.max(0, Math.min(1, Number(lastLoggedTimesteps) / Number(total)))
        : null;
    if (Number.isFinite(fromLogTs) && Number.isFinite(logProgressRatio)) {
      return Math.max(Number(fromLogTs), Number(logProgressRatio));
    }
    if (Number.isFinite(fromLogTs)) return Number(fromLogTs);
    if (Number.isFinite(logProgressRatio)) return Number(logProgressRatio);
    return null;
  }, [activeJob, lastLoggedTimesteps, logProgressRatio]);

  const effectiveProgress = useMemo(() => {
    const values = [progressA, progressB, progressC].filter((v) => Number.isFinite(v));
    return values.length > 0 ? Math.max(...values) : null;
  }, [progressA, progressB, progressC]);

  const rewardLatest = rewardPoints.length ? rewardPoints[rewardPoints.length - 1].value : null;
  const rewardBest = rewardPoints.length ? Math.max(...rewardPoints.map((p) => p.value)) : null;
  const scoreLatest = scorePoints.length ? scorePoints[scorePoints.length - 1].value : null;
  const scoreBest = scorePoints.length ? Math.max(...scorePoints.map((p) => p.value)) : null;

  const rewardSeries = useMemo(() => normalizeSeries(rewardPoints), [rewardPoints]);
  const scoreSeries = useMemo(() => normalizeSeries(scorePoints), [scorePoints]);

  const graphXMin = useMemo(() => {
    const allTs = [...rewardSeries, ...scoreSeries].map((p) => Number(p.t)).filter(Number.isFinite);
    if (allTs.length === 0) return 0;
    return Math.min(...allTs);
  }, [rewardSeries, scoreSeries]);
  const graphXMax = useMemo(() => {
    const allTs = [...rewardSeries, ...scoreSeries].map((p) => Number(p.t)).filter(Number.isFinite);
    if (allTs.length === 0) return 1;
    const mx = Math.max(...allTs);
    return mx > graphXMin ? mx : graphXMin + 1;
  }, [rewardSeries, scoreSeries, graphXMin]);

  const rewardMin = rewardPoints.length ? Math.min(...rewardPoints.map((p) => p.value), -10) : -10;
  const rewardMax = rewardPoints.length ? Math.max(...rewardPoints.map((p) => p.value), 10) : 10;
  const scoreMin = scorePoints.length ? Math.min(...scorePoints.map((p) => p.value), 0) : 0;
  const scoreMax = scorePoints.length ? Math.max(...scorePoints.map((p) => p.value), 1) : 1;

  const rewardPolyline = useMemo(
    () =>
      toPolylineByT(rewardSeries, {
        minX: graphXMin,
        maxX: graphXMax,
        minY: rewardMin,
        maxY: rewardMax,
        width: 700,
        height: 260,
      }),
    [rewardSeries, graphXMin, graphXMax, rewardMin, rewardMax]
  );
  const scoreStairPolyline = useMemo(
    () =>
      toStairPolylineByT(scoreSeries, {
        minX: graphXMin,
        maxX: graphXMax,
        minY: scoreMin,
        maxY: scoreMax,
        width: 700,
        height: 260,
      }),
    [scoreSeries, graphXMin, graphXMax, scoreMin, scoreMax]
  );

  const llmStoryCards = useMemo(() => {
    const cards = [];
    let seq = 1;

    if (activeJob) {
      cards.push({
        id: `story-${seq}`,
        seq: seq++,
        title: "Training Context",
        text: `Phase ${activeJob?.phase || "-"} job ${String(activeJob?.job_id || "").slice(0, 8)} is ${activeJob?.status || "unknown"} at ${fmt((Number(activeJob?.progress || 0) * 100), 1)}%.`,
        tone: "cyan",
      });
      if (rewardSeries.length >= 2 || scoreSeries.length >= 2) {
        const rewardStart = rewardSeries.length > 0 ? rewardSeries[0].value : null;
        const rewardEnd = rewardSeries.length > 0 ? rewardSeries[rewardSeries.length - 1].value : null;
        const scoreStart = scoreSeries.length > 0 ? scoreSeries[0].value : null;
        const scoreEnd = scoreSeries.length > 0 ? scoreSeries[scoreSeries.length - 1].value : null;
        cards.push({
          id: `story-${seq}`,
          seq: seq++,
          title: "Learning Trend",
          text: `Reward ${rewardStart != null ? fmt(rewardStart, 2) : "-"} -> ${rewardEnd != null ? fmt(rewardEnd, 2) : "-"}; ${scoreSignalMeta.label.toLowerCase()} ${scoreStart != null ? fmt(scoreStart, 3) : "-"} -> ${scoreEnd != null ? fmt(scoreEnd, 3) : "-"}.`,
          tone: "indigo",
        });
      }
    }

    for (const line of (logLines || []).slice(-14)) {
      const row = summarizeLogLine(line);
      cards.push({
        id: `log-${seq}-${line.slice(0, 8)}`,
        seq: seq++,
        title: row.title,
        text: row.text,
        tone: row.tone,
      });
    }

    const evalRows = Array.isArray(activeJob?.evaluation_rows) ? activeJob.evaluation_rows : [];
    for (const row of evalRows) {
      cards.push({
        id: `eval-${seq}-${row.task_id}`,
        seq: seq++,
        title: "Evaluation Replay",
        text: `${row.task_id}: score ${fmt(row.grader_score, 3)}, reward ${fmt(row.total_reward, 2)}, completed ${row.total_completed}, breaches ${row.total_sla_breaches}.`,
        tone: "emerald",
      });
    }
    if (toNumberOrNull(activeJob?.evaluation_avg_score) != null) {
      cards.push({
        id: `eval-avg-${seq}`,
        seq: seq++,
        title: "Evaluation Summary",
        text: `Average grader score ${fmt(activeJob.evaluation_avg_score, 3)} across evaluated tasks.`,
        tone: "emerald",
      });
    }

    for (const event of (envFlowEvents || []).slice(-10)) {
      cards.push({
        id: `replay-${seq}-${event.id}`,
        seq: seq++,
        title: "OpenEnv Replay",
        text: summarizeEnvEvent(event),
        tone: event?.tone || "cyan",
      });
    }

    return cards.slice(-32);
  }, [activeJob, rewardSeries, scoreSeries, logLines, envFlowEvents, scoreSignalMeta.label]);

  const progressText = (v) => (Number.isFinite(v) ? `${fmt(Number(v) * 100, 1)}%` : "-");
  const currentTs = useMemo(() => {
    const history = Array.isArray(activeJob?.metric_history) ? activeJob.metric_history : [];
    const histTs = history.length > 0 ? toNumberOrNull(history[history.length - 1]?.t ?? history[history.length - 1]?.total_timesteps) : null;
    return toNumberOrNull(activeJob?.latest_metrics?.total_timesteps) ?? histTs ?? lastLoggedTimesteps;
  }, [activeJob, lastLoggedTimesteps]);
  const currentReward = useMemo(() => {
    const history = Array.isArray(activeJob?.metric_history) ? activeJob.metric_history : [];
    const histReward = history.length > 0 ? toNumberOrNull(history[history.length - 1]?.ep_rew_mean ?? history[history.length - 1]?.mean_reward) : null;
    return toNumberOrNull(activeJob?.latest_metrics?.ep_rew_mean)
      ?? toNumberOrNull(activeJob?.latest_metrics?.mean_reward)
      ?? histReward;
  }, [activeJob]);
  const currentScore = scoreLatest;

  return (
    <div className="space-y-6">
      <div className="bg-slate-900/70 border border-white/5 rounded-xl p-5">
        <div className="flex items-center justify-between gap-3 mb-3">
          <h2 className="text-lg font-black text-white flex items-center gap-2">
            <span className="material-symbols-outlined text-indigo-400">hub</span>
            Endpoint Connectivity Matrix
          </h2>
          <button
            onClick={refreshEndpointHealth}
            className="text-xs font-bold px-3 py-1.5 rounded-lg bg-indigo-600/70 hover:bg-indigo-500 text-white"
          >
            Refresh Endpoints
          </button>
        </div>
        {endpointError && (
          <div className="mb-3 text-xs font-semibold text-amber-300 bg-amber-500/10 border border-amber-500/20 rounded p-2">
            {endpointError}
          </div>
        )}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
          {endpointRows.map((row) => (
            <div
              key={row.key}
              className={`border rounded-lg p-3 ${
                row.ok ? "border-emerald-500/25 bg-emerald-500/5" : "border-rose-500/25 bg-rose-500/5"
              }`}
            >
              <div className="flex items-center justify-between">
                <div className="text-sm font-bold text-white">{row.label}</div>
                <span className={`text-[10px] font-black ${row.ok ? "text-emerald-400" : "text-rose-400"}`}>
                  {row.ok ? "UP" : "DOWN"}
                </span>
              </div>
              <div className="text-xs text-slate-400 mt-1">
                {row.ok ? `${row.ms} ms` : row.error || "unreachable"}
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="bg-slate-900/70 border border-white/5 rounded-xl p-5">
        <div className="flex flex-wrap items-center justify-between gap-3 mb-4">
          <h2 className="text-lg font-black text-white flex items-center gap-2">
            <span className="material-symbols-outlined text-violet-400">tune</span>
            Live Training Control
          </h2>
          <div className="flex items-center gap-2">
            <button
              onClick={startTrainingJob}
              className="text-sm font-bold px-4 py-2 rounded-lg bg-violet-600 hover:bg-violet-500 text-white"
            >
              Start Training Job
            </button>
            <button
              onClick={stopTrainingJob}
              disabled={!activeJobId}
              className="text-sm font-bold px-4 py-2 rounded-lg bg-rose-600 hover:bg-rose-500 text-white disabled:opacity-50"
            >
              Stop Active Job
            </button>
            <button
              onClick={clearTrainingHistory}
              className="text-sm font-bold px-4 py-2 rounded-lg bg-slate-700 hover:bg-slate-600 text-white"
            >
              Clear Job History
            </button>
          </div>
        </div>

        {jobError && (
          <div className="mb-3 text-xs font-semibold text-rose-300 bg-rose-500/10 border border-rose-500/20 rounded p-2">
            {jobError}
          </div>
        )}

        <div className="grid grid-cols-1 md:grid-cols-4 gap-3 mb-3">
          <label className="text-xs text-slate-300">
            Phase
            <select
              value={jobForm.phase}
              onChange={(e) => setJobForm((prev) => ({ ...prev, phase: Number(e.target.value) }))}
              className="mt-1 w-full bg-slate-800 border border-white/10 rounded px-2 py-2 text-sm text-white"
            >
              <option value={1}>Phase 1</option>
              <option value={2}>Phase 2</option>
            </select>
          </label>
          <label className="text-xs text-slate-300">
            Timesteps
            <input
              value={jobForm.timesteps}
              onChange={(e) => setJobForm((prev) => ({ ...prev, timesteps: e.target.value }))}
              className="mt-1 w-full bg-slate-800 border border-white/10 rounded px-2 py-2 text-sm text-white"
            />
          </label>
          <label className="text-xs text-slate-300">
            N Envs
            <input
              value={jobForm.n_envs}
              onChange={(e) => setJobForm((prev) => ({ ...prev, n_envs: e.target.value }))}
              className="mt-1 w-full bg-slate-800 border border-white/10 rounded px-2 py-2 text-sm text-white"
            />
          </label>
          <label className="text-xs text-slate-300">
            Seed (optional)
            <input
              value={jobForm.seed}
              onChange={(e) => setJobForm((prev) => ({ ...prev, seed: e.target.value }))}
              className="mt-1 w-full bg-slate-800 border border-white/10 rounded px-2 py-2 text-sm text-white"
            />
          </label>
        </div>

        <div className="flex flex-wrap gap-2">
          <button
            onClick={() => setJobForm((prev) => ({ ...prev, timesteps: 30000, n_envs: Math.max(4, Number(prev.n_envs || 4)) }))}
            className="text-xs font-bold px-3 py-1.5 rounded bg-indigo-600/70 hover:bg-indigo-500 text-white"
          >
            Quick Demo Preset
          </button>
          <button
            onClick={() => setJobForm((prev) => ({ ...prev, timesteps: 120000, n_envs: 4 }))}
            className="text-xs font-bold px-3 py-1.5 rounded bg-slate-700 hover:bg-slate-600 text-white"
          >
            Default Preset
          </button>
        </div>
      </div>

      <div className="bg-slate-900/70 border border-white/5 rounded-xl p-5">
        <h2 className="text-lg font-black text-white flex items-center gap-2 mb-4">
          <span className="material-symbols-outlined text-indigo-400">monitoring</span>
          Live Metrics and Storytelling Timeline
        </h2>

        <div className="grid grid-cols-1 lg:grid-cols-4 gap-3 mb-4">
          <div className="bg-slate-950/50 border border-white/5 rounded p-3">
            <div className="text-[11px] uppercase text-slate-400">Active Job Status</div>
            <div className={`mt-2 inline-flex px-2 py-1 rounded border text-xs font-bold ${statusClasses(activeJob?.status)}`}>
              {activeJob?.status || "idle"}
            </div>
          </div>
          <div className="bg-slate-950/50 border border-white/5 rounded p-3">
            <div className="text-[11px] uppercase text-slate-400">Current Timesteps</div>
            <div className="mt-2 text-xl font-black text-indigo-300">{currentTs != null ? Number(currentTs).toLocaleString() : "-"}</div>
          </div>
          <div className="bg-slate-950/50 border border-white/5 rounded p-3">
            <div className="text-[11px] uppercase text-slate-400">Current Reward</div>
            <div className="mt-2 text-xl font-black text-amber-300">{currentReward != null ? fmt(currentReward, 3) : "-"}</div>
          </div>
          <div className="bg-slate-950/50 border border-white/5 rounded p-3">
            <div className="text-[11px] uppercase text-slate-400">Current {scoreSignalMeta.label}</div>
            <div className="mt-2 text-xl font-black text-emerald-300">{currentScore != null ? fmt(currentScore, 3) : "-"}</div>
          </div>
        </div>

        <div className="mb-4 flex flex-wrap items-center gap-3">
          <label className="text-xs text-slate-300">
            Story Job (active + history)
            <select
              value={activeJobId}
              onChange={(e) => setActiveJobId(e.target.value)}
              className="mt-1 min-w-[260px] bg-slate-800 border border-white/10 rounded px-2 py-2 text-sm text-white"
            >
              {jobs.map((job) => (
                <option key={job.job_id} value={job.job_id}>
                  {String(job.job_id).slice(0, 8)} | phase {job.phase || "-"} | {job.status}
                </option>
              ))}
            </select>
          </label>
          <div className="text-[11px] text-slate-400">
            Reward line (left axis) + {scoreSignalMeta.label} stair-step line (right axis), updated from live backend metrics.
          </div>
        </div>

        <div className="bg-slate-950/50 border border-white/5 rounded p-3 mb-4">
          <div className="flex items-center justify-between mb-2">
            <div className="text-xs uppercase tracking-widest text-slate-400">Combined Reward and Score (Dual Axis)</div>
            <div className="text-[11px] text-slate-500">
              timesteps {Number.isFinite(graphXMin) ? Number(graphXMin).toLocaleString() : "-"} - {Number.isFinite(graphXMax) ? Number(graphXMax).toLocaleString() : "-"}
            </div>
          </div>
          {rewardSeries.length === 0 && scoreSeries.length === 0 ? (
            <div className="h-[260px] flex items-center justify-center text-slate-500 text-sm">
              Waiting for live metric history from training logs...
            </div>
          ) : (
            <div className="relative">
              <svg viewBox="0 0 700 260" className="w-full h-[260px]">
                {[0, 1, 2, 3, 4].map((i) => (
                  <line
                    key={`grid-${i}`}
                    x1="0"
                    x2="700"
                    y1={String((260 / 4) * i)}
                    y2={String((260 / 4) * i)}
                    stroke="#334155"
                    strokeOpacity="0.35"
                    strokeWidth="1"
                  />
                ))}
                {rewardPolyline ? (
                  <polyline
                    points={rewardPolyline}
                    fill="none"
                    stroke="#818cf8"
                    strokeWidth="2.2"
                    strokeLinejoin="round"
                    strokeLinecap="round"
                  />
                ) : null}
                {scoreStairPolyline ? (
                  <polyline
                    points={scoreStairPolyline}
                    fill="none"
                    stroke="#34d399"
                    strokeWidth="2.2"
                    strokeLinejoin="round"
                    strokeLinecap="round"
                  />
                ) : null}
              </svg>
              <div className="absolute top-1 left-2 text-[10px] text-indigo-300">
                Reward min {rewardMin.toFixed(2)} | max {rewardMax.toFixed(2)}
              </div>
              <div className="absolute top-1 right-2 text-[10px] text-emerald-300">
                {scoreSignalMeta.label} min {scoreMin.toFixed(3)} | max {scoreMax.toFixed(3)}
              </div>
            </div>
          )}
          <div className="mt-2 text-xs text-slate-300">
            reward current: {rewardLatest != null ? rewardLatest.toFixed(3) : "-"} | reward best: {rewardBest != null ? rewardBest.toFixed(3) : "-"} | {scoreSignalMeta.label.toLowerCase()} current: {scoreLatest != null ? scoreLatest.toFixed(3) : "-"} | {scoreSignalMeta.label.toLowerCase()} best: {scoreBest != null ? scoreBest.toFixed(3) : "-"}
          </div>
          <div className="mt-1 text-[11px] text-slate-500">
            Legend: <span className="text-indigo-300">Reward (line)</span> - <span className="text-emerald-300">{scoreSignalMeta.label} (stair-step hold-last-value)</span>{scoreSignalMeta.fallback ? " - fallback metric used because grader score has no live movement yet." : ""}
          </div>
        </div>

        <div className="bg-slate-950/50 border border-white/5 rounded p-3">
          <div className="flex items-center justify-between mb-3">
            <div className="text-xs uppercase tracking-widest text-slate-400">LLM Story Feed (logs + replay + evaluation)</div>
            <div className="text-[11px] text-slate-500">Sequential order - {llmStoryCards.length} cards</div>
          </div>
          {llmStoryCards.length === 0 ? (
            <div className="text-slate-500 text-sm">No storyline events yet.</div>
          ) : (
            <div className="space-y-2 max-h-[340px] overflow-auto pr-1">
              {llmStoryCards.map((card) => (
                <div key={card.id} className={`border rounded p-2.5 ${toneClasses(card.tone)}`}>
                  <div className="flex items-center justify-between mb-1">
                    <div className="text-[11px] font-bold text-white">{card.title}</div>
                    <div className="text-[10px] text-slate-400">#{card.seq}</div>
                  </div>
                  <div className="text-[11px] text-slate-300 font-mono leading-relaxed break-words">{card.text}</div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <div className="bg-slate-900/70 border border-white/5 rounded-xl p-5">
          <div className="flex items-center justify-between mb-3">
            <h2 className="text-lg font-black text-white flex items-center gap-2">
              <span className="material-symbols-outlined text-amber-400">history</span>
              Training Job History
            </h2>
            <div className="flex items-center gap-2">
              <button
                onClick={() => deleteTrainingJob(activeJobId)}
                disabled={!activeJobId || !!deletingJobId}
                className="text-xs font-bold px-3 py-1.5 rounded bg-rose-600/70 hover:bg-rose-500 text-white disabled:opacity-50"
              >
                {deletingJobId && deletingJobId === activeJobId ? "Deleting..." : "Delete Selected"}
              </button>
              <button
                onClick={refreshJobs}
                className="text-xs font-bold px-3 py-1.5 rounded bg-amber-600/70 hover:bg-amber-500 text-white"
              >
                Refresh Jobs
              </button>
            </div>
          </div>
          {jobsError && <div className="text-xs text-rose-300 mb-2">{jobsError}</div>}
          {jobsLoading ? (
            <div className="text-sm text-slate-400">Loading jobs...</div>
          ) : (
            <div className="max-h-80 overflow-auto border border-white/5 rounded">
              <table className="w-full text-xs">
                <thead className="bg-slate-800/70 text-slate-300 sticky top-0">
                  <tr>
                    <th className="px-2 py-2 text-left">Job</th>
                    <th className="px-2 py-2 text-left">Status</th>
                    <th className="px-2 py-2 text-left">Phase</th>
                    <th className="px-2 py-2 text-left">Progress</th>
                    <th className="px-2 py-2 text-left">Updated</th>
                    <th className="px-2 py-2 text-left">Action</th>
                  </tr>
                </thead>
                <tbody>
                  {jobs.map((job) => {
                    const updated = timestampToDate(job.updated_at);
                    return (
                      <tr
                        key={job.job_id}
                        className={`border-t border-white/5 cursor-pointer hover:bg-white/5 ${
                          activeJobId === job.job_id ? "bg-indigo-500/10" : ""
                        }`}
                        onClick={() => setActiveJobId(job.job_id)}
                      >
                        <td className="px-2 py-2 text-indigo-300 font-mono">{String(job.job_id || "").slice(0, 8)}</td>
                        <td className="px-2 py-2">
                          <span className={`px-2 py-0.5 rounded border text-[11px] font-bold ${statusClasses(job.status)}`}>
                            {job.status}
                          </span>
                        </td>
                        <td className="px-2 py-2 text-slate-300">{job.phase || "-"}</td>
                        <td className="px-2 py-2 text-slate-300">{fmt((Number(job.progress || 0) * 100), 1)}%</td>
                        <td className="px-2 py-2 text-slate-400">{updated ? updated.toLocaleTimeString() : "-"}</td>
                        <td className="px-2 py-2">
                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              deleteTrainingJob(job.job_id);
                            }}
                            disabled={!!deletingJobId}
                            className="text-[11px] font-bold px-2 py-1 rounded bg-rose-600/70 hover:bg-rose-500 text-white disabled:opacity-50"
                          >
                            {deletingJobId === job.job_id ? "Deleting..." : "Delete"}
                          </button>
                        </td>
                      </tr>
                    );
                  })}
                  {jobs.length === 0 && (
                    <tr>
                      <td className="px-2 py-3 text-slate-500" colSpan={6}>
                        No training jobs found.
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          )}
        </div>

        <div className="bg-slate-900/70 border border-white/5 rounded-xl p-5">
          <div className="flex items-center justify-between mb-3">
            <h2 className="text-lg font-black text-white flex items-center gap-2">
              <span className="material-symbols-outlined text-emerald-400">database</span>
              Model Registry (Dynamic)
            </h2>
            <button
              onClick={refreshCatalog}
              className="text-xs font-bold px-3 py-1.5 rounded bg-emerald-600/70 hover:bg-emerald-500 text-white"
            >
              Refresh Models
            </button>
          </div>
          {modelError && <div className="text-xs text-amber-300 mb-2">{modelError}</div>}
          <div className="max-h-80 overflow-auto border border-white/5 rounded">
            <table className="w-full text-xs">
              <thead className="bg-slate-800/70 text-slate-300 sticky top-0">
                <tr>
                  <th className="px-2 py-2 text-left">Label</th>
                  <th className="px-2 py-2 text-left">Phase</th>
                  <th className="px-2 py-2 text-left">Source</th>
                  <th className="px-2 py-2 text-left">Exists</th>
                </tr>
              </thead>
              <tbody>
                {modelRows.map((m) => (
                  <tr key={`${m.path}-${m.source}`} className="border-t border-white/5">
                    <td className="px-2 py-2 text-slate-200">
                      <div>{m.label}</div>
                      <div className="text-[11px] text-slate-500 truncate max-w-[280px]">{m.path || "-"}</div>
                    </td>
                    <td className="px-2 py-2 text-slate-300">{m.phase || "-"}</td>
                    <td className="px-2 py-2 text-slate-300">{m.source || "-"}</td>
                    <td className={`px-2 py-2 ${m.exists ? "text-emerald-300" : "text-rose-300"}`}>
                      {m.exists ? "yes" : "no"}
                    </td>
                  </tr>
                ))}
                {modelRows.length === 0 && (
                  <tr>
                    <td className="px-2 py-3 text-slate-500" colSpan={4}>
                      No models discovered.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      <div className="bg-slate-900/70 border border-white/5 rounded-xl p-5">
        <h2 className="text-lg font-black text-white flex items-center gap-2 mb-4">
          <span className="material-symbols-outlined text-fuchsia-400">api</span>
          Automated OpenEnv Workflow (`reset`, `step`, `state`, `grade`)
        </h2>
        <div className="text-xs text-slate-400 mb-3">
          Runs sequentially across all available tasks and records each stage in chronological order.
        </div>

        {envError && (
          <div className="mb-3 text-xs font-semibold text-rose-300 bg-rose-500/10 border border-rose-500/20 rounded p-2">
            {envError}
          </div>
        )}

        <div className="grid grid-cols-1 md:grid-cols-4 gap-3 mb-3">
          <label className="text-xs text-slate-300">
            Task Scope
            <input
              value={`${(Array.isArray(tasks) && tasks.length > 0 ? tasks.length : 1)} task(s) automatic`}
              readOnly
              className="mt-1 w-full bg-slate-800 border border-white/10 rounded px-2 py-2 text-sm text-white"
            />
          </label>
          <label className="text-xs text-slate-300">
            Seed (optional)
            <input
              value={envSeed}
              onChange={(e) => setEnvSeed(e.target.value)}
              className="mt-1 w-full bg-slate-800 border border-white/10 rounded px-2 py-2 text-sm text-white"
            />
          </label>
          <label className="text-xs text-slate-300">
            Auto-Step Policy
            <select
              value={envPolicyName}
              onChange={(e) => setEnvPolicyName(e.target.value)}
              className="mt-1 w-full bg-slate-800 border border-white/10 rounded px-2 py-2 text-sm text-white"
            >
              {(agents.length > 0 ? agents : ["backlog_clearance"]).map((p) => (
                <option key={p} value={p}>
                  {p}
                </option>
              ))}
            </select>
          </label>
          <label className="text-xs text-slate-300">
            Max Automated Steps
            <input
              value={envMaxSteps}
              onChange={(e) => setEnvMaxSteps(e.target.value)}
              className="mt-1 w-full bg-slate-800 border border-white/10 rounded px-2 py-2 text-sm text-white"
            />
          </label>
        </div>

        <div className="flex gap-2 mb-4">
          <button
            onClick={runAutomatedOpenEnvFlow}
            disabled={envBusy}
            className="text-sm font-bold px-4 py-2 rounded-lg bg-fuchsia-600 hover:bg-fuchsia-500 text-white disabled:opacity-50"
          >
            {envBusy ? "Running Workflow..." : "Proceed"}
          </button>
        </div>

        {envFlowSummary && (
          <div className="mb-3 bg-slate-950/50 border border-white/5 rounded p-3 text-xs">
            <div className="text-slate-300">Tasks Executed: <span className="font-bold text-white">{envFlowSummary.tasks_executed}</span></div>
            <div className="text-slate-300">Total Steps Executed: <span className="font-bold text-white">{envFlowSummary.total_steps_executed}</span></div>
            <div className="text-slate-300">Average Score: <span className="font-bold text-emerald-300">{envFlowSummary.avg_score != null ? fmt(envFlowSummary.avg_score, 3) : "-"}</span></div>
            <div className="text-slate-300">Passed Tasks: <span className="font-bold text-cyan-300">{envFlowSummary.passed_tasks}</span></div>
          </div>
        )}

        {envFlowRuns.length > 0 && (
          <div className="mb-3 border border-white/5 rounded overflow-auto">
            <table className="w-full text-xs">
              <thead className="bg-slate-800/70 text-slate-300">
                <tr>
                  <th className="px-2 py-2 text-left">Task</th>
                  <th className="px-2 py-2 text-left">Steps</th>
                  <th className="px-2 py-2 text-left">Score</th>
                  <th className="px-2 py-2 text-left">Completed</th>
                  <th className="px-2 py-2 text-left">Backlog</th>
                  <th className="px-2 py-2 text-left">SLA Breaches</th>
                  <th className="px-2 py-2 text-left">Passed</th>
                </tr>
              </thead>
              <tbody>
                {envFlowRuns.map((row) => (
                  <tr key={`run-${row.task_id}`} className="border-t border-white/5">
                    <td className="px-2 py-2 text-slate-200">{row.task_id}</td>
                    <td className="px-2 py-2 text-slate-300">{row.steps_executed}</td>
                    <td className="px-2 py-2 text-emerald-300">{row.score != null ? fmt(row.score, 3) : "-"}</td>
                    <td className="px-2 py-2 text-slate-300">{row.final_completed ?? "-"}</td>
                    <td className="px-2 py-2 text-slate-300">{row.final_backlog ?? "-"}</td>
                    <td className="px-2 py-2 text-slate-300">{row.final_sla_breaches ?? "-"}</td>
                    <td className={`px-2 py-2 ${row.passed === true ? "text-emerald-300" : row.passed === false ? "text-rose-300" : "text-slate-400"}`}>
                      {row.passed === true ? "true" : row.passed === false ? "false" : "-"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        <div className="space-y-2 max-h-[380px] overflow-auto pr-1">
          {envFlowEvents.length === 0 ? (
            <div className="text-slate-500 text-sm">No automated workflow events yet.</div>
          ) : (
            envFlowEvents.map((event) => (
              <div key={event.id} className={`border rounded p-3 ${toneClasses(event.tone)}`}>
                <div className="flex items-center justify-between mb-1">
                  <div className="text-xs uppercase tracking-widest text-slate-400">{workflowStageLabel(event.stage)}</div>
                  <div className="text-[10px] text-slate-400">
                    #{event.seq} | {new Date(event.ts).toLocaleTimeString()}
                  </div>
                </div>
                <div className="text-xs text-slate-200 leading-relaxed">
                  {summarizeEnvEvent(event)}
                </div>
                {payloadHighlights(event.payload).length > 0 && (
                  <div className="mt-2 flex flex-wrap gap-1">
                    {payloadHighlights(event.payload).map(([k, v]) => (
                      <span
                        key={`${event.id}-${k}`}
                        className="text-[10px] bg-slate-800/70 border border-white/10 rounded px-1.5 py-0.5 text-slate-300"
                      >
                        {k}: {v}
                      </span>
                    ))}
                  </div>
                )}
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
}


