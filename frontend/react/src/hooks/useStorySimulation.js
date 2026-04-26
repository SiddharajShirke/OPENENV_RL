import { useState, useRef, useCallback, useEffect } from "react";
import { api } from "../api/client";

// ─────────────────────────────────────────────────────────────────────────────
// Narrative translator: maps raw action → human-readable cause→effect story
// ─────────────────────────────────────────────────────────────────────────────
function mapActionToStory(actionType, payload, reward, backlogDelta, slaDelta, fairnessDelta) {
  let title = "Standard Processing Cycle";
  let desc = "The system advanced one cycle and continued normal queue processing.";
  let reason = "No override was required, so routine processing continued.";
  let icon = "schedule";
  let type = reward > 0 ? "success" : "info";

  const changes = [];
  if (backlogDelta < 0) changes.push(`backlog improved by ${Math.abs(backlogDelta)} case(s)`);
  else if (backlogDelta > 0) changes.push(`backlog increased by ${backlogDelta} case(s)`);
  else changes.push("backlog stayed stable");

  if (slaDelta > 0) changes.push(`${slaDelta} new SLA breach(es) occurred`);
  else if (slaDelta < 0) changes.push(`${Math.abs(slaDelta)} SLA breach(es) recovered`);

  if (Number.isFinite(Number(fairnessDelta)) && Number(fairnessDelta) !== 0) {
    const v = Number(fairnessDelta);
    changes.push(`fairness gap ${v > 0 ? "worsened" : "improved"} by ${Math.abs(v).toFixed(3)}`);
  }

  const effectClause = `${changes.join(", ")}.`;
  if (slaDelta > 0) type = "error";

  switch (actionType) {
    case "assign_capacity":
      title = "Capacity Assigned";
      desc = `Officers were assigned to '${payload.service_target ?? payload.service ?? "target queue"}'; ${effectClause}`;
      reason = "The agent detected staffing pressure and increased capacity where it could reduce delay.";
      icon = "group_add";
      break;
    case "reallocate_officers":
      title = "Staff Reallocated";
      desc = `Officers were reallocated toward higher-pressure services; ${effectClause}`;
      reason = `The agent shifted staffing to reduce bottlenecks in '${payload.service_target ?? "priority"}' services.`;
      icon = "compare_arrows";
      break;
    case "request_missing_documents":
      title = "Documents Requested";
      desc = `Missing documents were requested to unblock pending files; ${effectClause}`;
      reason = "The agent prioritized document blockers to avoid queue stagnation.";
      icon = "rule_folder";
      type = type !== "error" ? "success" : type;
      break;
    case "escalate_service":
      title = "Service Escalated";
      desc = `At-risk services were escalated for faster handling; ${effectClause}`;
      reason = "Escalation was used to protect SLA-critical cases.";
      icon = "warning";
      type = "warning";
      break;
    case "set_priority_mode":
      title = "Priority Mode Updated";
      desc = `Priority mode switched to '${payload.priority_mode ?? "balanced"}'; ${effectClause}`;
      reason = "The agent changed queue strategy to better match current workload pressure.";
      icon = "model_training";
      break;
    default:
      desc = `Routine processing executed; ${effectClause}`;
      break;
  }

  if (reward < 0 && type === "info") type = "warning";

  const isHighReward = reward >= 1.0;
  const isHugeImpact = backlogDelta <= -5;
  return { title, desc, reason, icon, type, isHighReward, isHugeImpact };
}

// Determines the simulation phase label from step index and total
function getPhase(step, maxSteps) {
  const pct = step / Math.max(maxSteps, 1);
  if (pct < 0.33) return "early";
  if (pct < 0.67) return "middle";
  return "late";
}

// Detect if a step is a "key decision" turning point
function isKeyDecision(s, backlogDelta) {
  return (
    Math.abs(Number(s.reward)) >= 1.0 || // high reward magnitude
    (backlogDelta !== 0 && Math.abs(backlogDelta) >= 5) || // large backlog swing
    Boolean(s.invalid_action) // failed action = notable event
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Hook
// ─────────────────────────────────────────────────────────────────────────────
export function useStorySimulation({ defaultTask }) {
  const [taskId, setTaskId] = useState(defaultTask || "district_backlog_easy");
  const [maxSteps, setMaxSteps] = useState(40);
  const [agentMode, setAgentMode] = useState("trained_rl");
  const [policyName, setPolicyName] = useState("backlog_clearance");
  const [modelPath, setModelPath] = useState("");
  const [modelType, setModelType] = useState("maskable");
  const [availablePolicies, setAvailablePolicies] = useState([]);
  const [availableModels, setAvailableModels] = useState([]);
  const [configError, setConfigError] = useState("");
  const [running, setRunning] = useState(false);
  const [starting, setStarting] = useState(false);
  const [runId, setRunId] = useState("");

  const [kpis, setKpis] = useState({
    backlog: 0, backlogDelta: 0,
    slaBreaches: 0, slaDelta: 0,
    fairness: 0, fairnessDelta: 0,
  });

  const [timeline, setTimeline] = useState([]);
  const [resources, setResources] = useState([]);

  // Progress tracking
  const [currentStep, setCurrentStep] = useState(0);

  // Before vs after journey stats
  const [journeyStats, setJourneyStats] = useState(null); // null = not yet done

  // Internal refs
  const lastState = useRef({ backlog: 0, sla: 0, fairness: 0 });
  const initialSnapshot = useRef(null); // captured on first real step
  const stepCount = useRef(0);
  const maxStepsRef = useRef(40);

  useEffect(() => {
    let mounted = true;
    (async () => {
      try {
        const [policiesRes, modelsV1Res, modelsV2Res] = await Promise.allSettled([
          api("/agents"),
          api("/rl_models"),
          api("/rl/models"),
        ]);
        if (!mounted) return;

        const policyRows = policiesRes.status === "fulfilled" && Array.isArray(policiesRes.value) ? policiesRes.value : [];
        setAvailablePolicies(policyRows);
        if (policyRows.length > 0 && !policyRows.includes(policyName)) {
          setPolicyName(policyRows[0]);
        }

        const modelRowsV1 = modelsV1Res.status === "fulfilled" && Array.isArray(modelsV1Res.value?.models)
          ? modelsV1Res.value.models
          : [];
        const modelRowsV2 = modelsV2Res.status === "fulfilled" && Array.isArray(modelsV2Res.value)
          ? modelsV2Res.value.map((row) => ({
            label: row?.model_path ? String(row.model_path).split(/[\\/]/).pop() : "model",
            path: row?.model_path ? (String(row.model_path).toLowerCase().endsWith(".zip") ? row.model_path : `${row.model_path}.zip`) : "",
            exists: Boolean(row?.exists),
            model_type: "maskable",
          }))
          : [];

        const dedupe = new Map();
        for (const m of [...modelRowsV1, ...modelRowsV2]) {
          const key = String(m?.path || "").replace(/\\/g, "/").toLowerCase();
          if (!key || dedupe.has(key)) continue;
          dedupe.set(key, m);
        }
        const existingModels = Array.from(dedupe.values()).filter((m) => Boolean(m?.exists));
        setAvailableModels(existingModels);
        const preferred =
          existingModels.find((m) => String(m.path || "").toLowerCase().includes("phase2_final")) ||
          existingModels[0];
        if (preferred?.path) {
          setModelPath(preferred.path);
          setModelType(preferred.model_type || "maskable");
          setAgentMode((prev) => (prev === "baseline_policy" ? "trained_rl" : prev));
        }
      } catch (err) {
        if (!mounted) return;
        setConfigError(err?.message || "Failed to load simulation options.");
      }
    })();
    return () => {
      mounted = false;
    };
  }, []);

  const startSimulation = async () => {
    setStarting(true);
    setConfigError("");
    setJourneyStats(null);
    setCurrentStep(0);
    initialSnapshot.current = null;
    stepCount.current = 0;
    maxStepsRef.current = maxSteps;
    try {
      const payload = {
        task_id: taskId,
        agent_mode: agentMode,
        max_steps: maxSteps,
        policy_name: policyName,
        model_path: modelPath || null,
        model_type: modelType,
      };

      const started = await api("/simulation/live/start", {
        method: "POST",
        body: JSON.stringify(payload),
      });

      setRunId(started.run_id);
      setTimeline([{
        id: "start",
        time: "Step 0",
        title: "Simulation Initialized",
        desc: `Scenario locked: ${taskId.replace(/_/g, " ")}. Agent mode '${agentMode}' engaged — agent begins resolving backlog.`,
        impact: 0,
        type: "info",
        icon: "rocket_launch",
        phase: "early",
        key: false,
      }]);
      setResources([]);
      lastState.current = { backlog: 0, sla: 0, fairness: 0 };
      setRunning(true);
    } catch (err) {
      console.error("Start failed:", err);
      setTimeline([{
        id: "error",
        time: "—",
        title: "Initialization Failed",
        desc: `Backend error: ${err.message || "Cannot start simulation."}`,
        impact: 0,
        type: "error",
        icon: "error",
        phase: "early",
        key: false,
      }]);
      setConfigError(err?.message || "Cannot start simulation.");
    } finally {
      setStarting(false);
    }
  };

  const stopSimulation = async () => {
    if (!runId) return;
    try {
      await api(`/simulation/live/${runId}/stop`, { method: "POST" });
    } catch (err) {
      console.error(err);
    } finally {
      setRunning(false);
    }
  };

  // Polling loop — runs while running=true
  const runLoop = useCallback(async (rid, cancelled) => {
    if (cancelled.v) return;
    try {
      const res = await api("/simulation/live/step", {
        method: "POST",
        body: JSON.stringify({ run_id: rid }),
      });

      if (cancelled.v) return;

      if (res.step) {
        const s = res.step;
        stepCount.current += 1;
        const stepNum = Number(s.step ?? stepCount.current);
        setCurrentStep(stepNum);

        const currentBacklog = Number(s.backlog ?? 0);
        const currentSla = Number(s.sla_breaches ?? 0);
        const currentFairness = Number(s.fairness_gap ?? 0);

        // Capture initial snapshot from step 1
        if (initialSnapshot.current === null) {
          initialSnapshot.current = {
            backlog: currentBacklog,
            sla: currentSla,
            fairness: currentFairness,
          };
        }

        const backlogDelta = currentBacklog - lastState.current.backlog;
        const slaDelta = currentSla - lastState.current.sla;
        const fairnessDelta = currentFairness - lastState.current.fairness;

        setKpis({
          backlog: currentBacklog,
          backlogDelta,
          slaBreaches: currentSla,
          slaDelta,
          fairness: currentFairness,
          fairnessDelta,
        });

        lastState.current = { backlog: currentBacklog, sla: currentSla, fairness: currentFairness };

        const payload = typeof s.action_payload === "string"
          ? (() => { try { return JSON.parse(s.action_payload); } catch { return {}; } })()
          : (s.action_payload || {});

        const story = mapActionToStory(
          s.action_type || "advance_time",
          payload,
          Number(s.reward),
          backlogDelta,
          slaDelta,
          fairnessDelta
        );

        const phase = getPhase(stepNum, maxStepsRef.current);
        const key = isKeyDecision(s, backlogDelta);
        const improvesBacklog = backlogDelta < 0;
        const worsensBacklog = backlogDelta > 0;
        const worsensSla = slaDelta > 0;
        const improvesSla = slaDelta < 0;
        const outcomeLabel = improvesBacklog || improvesSla
          ? "Improvement"
          : worsensBacklog || worsensSla
            ? "Degradation"
            : "Stable";
        const outcomeType = outcomeLabel === "Improvement" ? "success" : outcomeLabel === "Degradation" ? "warning" : "info";

        const newEvent = {
          id: `step-${stepNum}`,
          time: `Step ${stepNum}`,
          title: s.invalid_action ? "Action Blocked" : story.title,
          desc: s.invalid_action
            ? "This action was blocked by environment constraints; the agent adapts on the next step."
            : story.desc,
          reason: s.invalid_action ? "The attempted operation violated environment constraints (e.g. over-assignment)." : story.reason,
          impact: Number(s.reward),
          type: s.invalid_action ? "error" : story.type,
          icon: s.invalid_action ? "block" : story.icon,
          isHighReward: story.isHighReward && !s.invalid_action,
          isHugeImpact: story.isHugeImpact && !s.invalid_action,
          phase,
          key,
          outcomeLabel,
          outcomeType,
          backlogDelta, // Used for phase summary
        };

        // Collapse consecutive identical titles (deduplication for repeated events)
        setTimeline((prev) => {
          const [top, ...rest] = prev;
          if (
            top &&
            top.title === newEvent.title &&
            top.phase === newEvent.phase &&
            !top.key &&
            !newEvent.key
          ) {
            // Merge: bump count, accumulate reward and backlog diff
            const merged = {
              ...top,
              id: newEvent.id,
              time: `${top.time?.split("–")[0]?.trim()}–${newEvent.time}`,
              desc: top.desc,
              impact: Number(top.impact) + Number(newEvent.impact),
              backlogDelta: (top.backlogDelta || 0) + backlogDelta,
              _count: (top._count || 1) + 1,
            };
            return [merged, ...rest].slice(0, 30);
          }
          return [newEvent, ...prev].slice(0, 30);
        });

        // Update queue monitors
        if (Array.isArray(s.queue_rows) && s.queue_rows.length > 0) {
          const maxCases = Math.max(...s.queue_rows.map((q) => q.active_cases ?? 0), 1);
          setResources(s.queue_rows.map((q) => ({
            name: (q.service ?? q.service_type ?? "unknown").replace(/_/g, " ").toUpperCase(),
            activeCases: q.active_cases ?? 0,
            percentage: Math.min(100, Math.floor(((q.active_cases ?? 0) / maxCases) * 100)),
          })));
        }
      }

      // Episode done
      if (res.done || res.step?.done) {
        const finalBacklog = lastState.current.backlog;
        const initSnap = initialSnapshot.current ?? { backlog: finalBacklog, sla: 0, fairness: 0 };

        const backlogImprovement = initSnap.backlog > 0
          ? Math.round(((initSnap.backlog - finalBacklog) / initSnap.backlog) * 100)
          : 0;

        setJourneyStats({
          initialBacklog: initSnap.backlog,
          finalBacklog,
          backlogImprovement,
          initialSla: initSnap.sla,
          finalSla: lastState.current.sla,
          totalSteps: stepCount.current,
          finalScore: res.score ?? null,
          totalReward: res.total_reward ?? null,
        });

        setTimeline((prev) => [{
          id: "end",
          time: "Final",
          title: "Episode Complete",
          desc: `Resolution finished in ${stepCount.current} steps. Final score: ${res.score != null ? (res.score * 100).toFixed(1) + "%" : "N/A"}. Backlog ${finalBacklog < initSnap.backlog ? "reduced" : "unchanged"} — SLAs verified.`,
          impact: res.total_reward ?? 0,
          type: "success",
          icon: "verified",
          phase: "late",
          key: true,
        }, ...prev]);

        setRunning(false);
        return;
      }

      setTimeout(() => runLoop(rid, cancelled), 1000);
    } catch (err) {
      if (!cancelled.v) {
        setRunning(false);
        setTimeline((prev) => [{
          id: `error-${Date.now()}`,
          time: "Halted",
          title: "System Error Detected",
          desc: `Backend synchronization failed: ${err.message}`,
          impact: 0,
          type: "error",
          icon: "warning",
          phase: "late",
          key: false,
        }, ...prev]);
      }
    }
  }, []);

  // Start/stop the polling loop reactively
  const cancelRef = useRef({ v: false });
  useEffect(() => {
    if (!running || !runId) {
      cancelRef.current.v = true;
      return undefined;
    }
    cancelRef.current = { v: false };
    const boot = setTimeout(() => {
      if (!cancelRef.current.v) {
        runLoop(runId, cancelRef.current);
      }
    }, 100);
    return () => {
      clearTimeout(boot);
      cancelRef.current.v = true;
    };
  }, [running, runId, runLoop]);

  return {
    taskId, setTaskId,
    maxSteps, setMaxSteps,
    agentMode, setAgentMode,
    policyName, setPolicyName,
    modelPath, setModelPath,
    modelType, setModelType,
    availablePolicies,
    availableModels,
    configError,
    running, starting,
    currentStep,
    kpis, timeline, resources,
    journeyStats,
    startSimulation, stopSimulation,
  };
}




