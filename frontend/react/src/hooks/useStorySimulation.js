import { useState, useRef, useCallback } from "react";
import { api, fmt } from "../api/client";

// ─────────────────────────────────────────────────────────────────────────────
// Narrative translator: maps raw action → human-readable cause→effect story
// ─────────────────────────────────────────────────────────────────────────────
function mapActionToStory(actionType, payload, reward, backlogDelta, slaDelta) {
  let title = "System Advanced Time";
  let desc = "Simulation progressed standard queue processing.";
  let reason = "Agents continued with standard assignments to let active services burn down queues.";
  let icon = "schedule";
  let type = reward > 0 ? "success" : "info";

  const effectClause =
    backlogDelta < 0
      ? `→ backlog reduced by ${Math.abs(backlogDelta)}.`
      : slaDelta > 0
      ? `→ ${slaDelta} cases hit critical SLA breach.`
      : "→ workload stabilized.";

  if (slaDelta > 0) type = "error";

  switch (actionType) {
    case "assign_capacity":
      title = "Resources Allocated";
      desc = `Assigned ${payload.capacity_assignment ?? payload.officer_delta ?? 1} officers to '${payload.service_target ?? payload.service ?? "queue"}'. ${effectClause}`;
      reason = "Detected capacity shortage and dynamically routed officers to prevent processing delays.";
      icon = "group_add";
      break;
    case "reallocate_officers":
      title = "Staff Redeployed";
      desc = `Shifted ${Math.abs(payload.reallocation_delta ?? payload.officer_delta ?? 1)} officers to prioritize critical backlog. ${effectClause}`;
      reason = `Rebalanced resources to specifically target overloaded '${payload.service_target ?? "high-risk"}' queues avoiding imminent breaches.`;
      icon = "compare_arrows";
      break;
    case "request_missing_documents":
      title = "Bottleneck Cleared";
      desc = `Requested missing documentation for blocked cases. ${effectClause}`;
      reason = "Identified process starvation due to missing constraints, clearing them proactively.";
      icon = "rule_folder";
      type = type !== "error" ? "success" : type;
      break;
    case "escalate_service":
      title = "Priority Override Applied";
      desc = `Used emergency escalation to fast-track at-risk cases. ${effectClause}`;
      reason = "Urgency threshold exceeded; applying forced escalation to preserve SLAs.";
      icon = "warning";
      type = "warning";
      break;
    case "set_priority_mode":
      title = "Queue Strategy Updated";
      desc = `System automatically shifted to '${payload.priority_mode}' strategy to resolve load imbalances.`;
      reason = "Global policy adjustment to counteract widespread performance degradation.";
      icon = "model_training";
      break;
    default:
      break;
  }

  if (reward < 0 && type === "info") type = "warning";

  const isHighReward = reward >= 1.0;
  const isHugeImpact = backlogDelta <= -5; // Huge backlog drop (negative delta is good)

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

  const startSimulation = async () => {
    setStarting(true);
    setJourneyStats(null);
    setCurrentStep(0);
    initialSnapshot.current = null;
    stepCount.current = 0;
    maxStepsRef.current = maxSteps;
    try {
      const payload = {
        task_id: taskId,
        agent_mode: "baseline_policy",
        max_steps: maxSteps,
        policy_name: "backlog_clearance",
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
        desc: `Scenario locked: ${taskId.replace(/_/g, " ")}. Baseline policy engaged — agent begins resolving backlog.`,
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
          slaDelta
        );

        const phase = getPhase(stepNum, maxStepsRef.current);
        const key = isKeyDecision(s, backlogDelta);

        const newEvent = {
          id: `step-${stepNum}`,
          time: `Step ${stepNum}`,
          title: s.invalid_action ? "Action Blocked" : story.title,
          desc: s.invalid_action
            ? `Technical constraints prevented this action. The agent will adapt. ${story.desc.includes("→") ? story.desc.split("→")[1]?.trim() ?? "" : ""}`
            : story.desc,
          reason: s.invalid_action ? "The attempted operation violated environment constraints (e.g. over-assignment)." : story.reason,
          impact: Number(s.reward),
          type: s.invalid_action ? "error" : story.type,
          icon: s.invalid_action ? "block" : story.icon,
          isHighReward: story.isHighReward && !s.invalid_action,
          isHugeImpact: story.isHugeImpact && !s.invalid_action,
          phase,
          key,
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
  const prevRunning = useRef(false);

  if (running && !prevRunning.current) {
    prevRunning.current = true;
    cancelRef.current = { v: false };
    // Kick off after a tiny delay so runId is set
    setTimeout(() => {
      if (!cancelRef.current.v) {
        runLoop(runId, cancelRef.current);
      }
    }, 100);
  }
  if (!running && prevRunning.current) {
    prevRunning.current = false;
    cancelRef.current.v = true;
  }

  return {
    taskId, setTaskId,
    maxSteps, setMaxSteps,
    running, starting,
    currentStep,
    kpis, timeline, resources,
    journeyStats,
    startSimulation, stopSimulation,
  };
}
