import React, { useState, useEffect } from "react";
import { api, fmt } from "../../api/client";
import { useStorySimulation } from "../../hooks/useStorySimulation";
import { TrainingTabV2 } from "./TrainingTabV2";

// --- Timeline Tab -------------------------------------------------------------
const PHASE_LABELS = {
  early: { label: "Early Phase", color: "indigo", icon: "flag", desc: "Agent explores the environment and initial decisions are made." },
  middle: { label: "Mid-Phase", color: "amber", icon: "timeline", desc: "Policy adapts as patterns emerge in the backlog." },
  late: { label: "Final Phase", color: "violet", icon: "sports_score", desc: "Agent converges toward optimal resolution strategy." },
};

function TimelineTab({ tasks }) {
  const {
    taskId, setTaskId, maxSteps, setMaxSteps,
    agentMode,
    policyName, setPolicyName,
    modelPath, setModelPath,
    modelType, setModelType,
    availablePolicies,
    availableModels,
    configError,
    running, starting, currentStep,
    kpis, timeline, resources, journeyStats,
    startSimulation, stopSimulation,
  } = useStorySimulation({ defaultTask: tasks[0] || "district_backlog_easy" });

  const isIdle = !starting && !running;
  const startBlocked = agentMode === "trained_rl" && !modelPath;
  const progressPct = maxSteps > 0 ? Math.min(100, Math.round((currentStep / maxSteps) * 100)) : 0;
  const fmt2 = (n) => new Intl.NumberFormat().format(n ?? 0);
  const fmtDelta = (n) => { const v = Number(n ?? 0); return v > 0 ? `+${v.toFixed(1)}` : v.toFixed(1); };

  // Local string buffer so the user can freely type without the field snapping back
  const [stepsInput, setStepsInput] = useState(String(maxSteps));
  // Keep buffer in sync if maxSteps changes from outside
  React.useEffect(() => { setStepsInput(String(maxSteps)); }, [maxSteps]);

  // Build phase-annotated timeline: insert phase dividers between phase changes
  const annotatedTimeline = [];
  let lastPhase = null;
  let phaseStats = { drop: 0, keys: 0 };

  for (let i = 0; i < timeline.length; i++) {
    const ev = timeline[i];
    const ph = ev.phase;

    if (ph && ph !== lastPhase) {
      if (lastPhase && PHASE_LABELS[lastPhase]) {
        // We reached the end of the previous (newer) phase in the chronological timeline,
        // so insert its summary before starting the older phase.
        annotatedTimeline.push({
          _summary: true,
          phase: lastPhase,
          stats: { ...phaseStats },
          key: `sum-${lastPhase}-${i}`,
        });
      }
      if (PHASE_LABELS[ph]) {
        annotatedTimeline.push({ _divider: true, phase: ph, key: `div-${ph}-${i}` });
      }
      lastPhase = ph;
      phaseStats = { drop: 0, keys: 0 };
    }

    if (ev.key) phaseStats.keys += 1;
    if (ev.backlogDelta) phaseStats.drop += ev.backlogDelta;

    annotatedTimeline.push(ev);
  }

  // Handle the very last (oldest) phase summary at the bottom of the list
  if (lastPhase && PHASE_LABELS[lastPhase] && timeline.length > 0) {
    annotatedTimeline.push({
      _summary: true,
      phase: lastPhase,
      stats: { ...phaseStats },
      key: `sum-${lastPhase}-end`,
    });
  }

  return (
    <div className="space-y-5">
      {/* --- Controls bar --- */}
      <div className="flex flex-wrap gap-3 items-center justify-between bg-slate-900/60 border border-white/5 rounded-xl px-5 py-3">
        <div className="flex flex-wrap items-center gap-4">
          <div className="flex items-center gap-2">
            <span className="text-slate-400 text-sm font-medium">Scenario</span>
            <select
              value={taskId}
              onChange={(e) => setTaskId(e.target.value)}
              disabled={!isIdle}
              className="appearance-none bg-slate-800 border border-white/10 text-sm font-medium px-3 py-1.5 rounded-lg text-indigo-300 focus:outline-none focus:border-indigo-500 cursor-pointer"
            >
              {tasks.length > 0
                ? tasks.map((t) => <option key={t} value={t} className="bg-slate-900">{t.replace(/_/g, " ").toUpperCase()}</option>)
                : <option>Loading...</option>}
            </select>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-slate-400 text-sm font-medium">Steps</span>
            <input
              type="number"
              min={10}
              max={100}
              step={10}
              value={stepsInput}
              disabled={!isIdle}
              onChange={(e) => setStepsInput(e.target.value)}
              onBlur={() => {
                const v = parseInt(stepsInput, 10);
                const clamped = isNaN(v) ? 40 : Math.min(100, Math.max(10, v));
                setMaxSteps(clamped);
                setStepsInput(String(clamped));
              }}
              onKeyDown={(e) => {
                if (e.key === "Enter") e.currentTarget.blur();
              }}
              className="w-20 bg-slate-800 border border-white/10 text-sm font-medium px-3 py-1.5 rounded-lg text-indigo-300 focus:outline-none focus:border-indigo-500 text-center"
            />
          </div>
          {agentMode === "baseline_policy" && (
            <div className="flex items-center gap-2">
              <span className="text-slate-400 text-sm font-medium">Policy</span>
              <select
                value={policyName}
                onChange={(e) => setPolicyName(e.target.value)}
                disabled={!isIdle}
                className="appearance-none bg-slate-800 border border-white/10 text-sm font-medium px-3 py-1.5 rounded-lg text-indigo-300 focus:outline-none focus:border-indigo-500 cursor-pointer"
              >
                {(availablePolicies.length > 0 ? availablePolicies : ["backlog_clearance"]).map((p) => (
                  <option key={p} value={p} className="bg-slate-900">{String(p).replace(/_/g, " ")}</option>
                ))}
              </select>
            </div>
          )}
          {agentMode === "trained_rl" && (
            <>
              <div className="flex items-center gap-2">
                <span className="text-slate-400 text-sm font-medium">Model</span>
                <select
                  value={modelPath}
                  onChange={(e) => {
                    const selected = availableModels.find((m) => m.path === e.target.value);
                    setModelPath(e.target.value);
                    if (selected?.model_type) setModelType(selected.model_type);
                  }}
                  disabled={!isIdle}
                  className="max-w-[260px] appearance-none bg-slate-800 border border-white/10 text-sm font-medium px-3 py-1.5 rounded-lg text-indigo-300 focus:outline-none focus:border-indigo-500 cursor-pointer"
                >
                  {(availableModels.length > 0
                    ? availableModels
                    : [{ label: "No model found", path: "", model_type: "maskable" }]
                  ).map((m) => (
                    <option key={`${m.path}-${m.model_type}`} value={m.path} className="bg-slate-900">
                      {m.label || m.path || "Unknown model"}
                    </option>
                  ))}
                </select>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-slate-400 text-sm font-medium">Type</span>
                <select
                  value={modelType}
                  onChange={(e) => setModelType(e.target.value)}
                  disabled={!isIdle}
                  className="appearance-none bg-slate-800 border border-white/10 text-sm font-medium px-3 py-1.5 rounded-lg text-indigo-300 focus:outline-none focus:border-indigo-500 cursor-pointer"
                >
                  <option value="maskable" className="bg-slate-900">Maskable PPO</option>
                  <option value="recurrent" className="bg-slate-900">Recurrent PPO</option>
                </select>
              </div>
            </>
          )}
        </div>
        <button
          onClick={running ? stopSimulation : startSimulation}
          disabled={starting || (!running && startBlocked)}
          className={`text-white text-sm font-bold px-6 py-2 rounded-lg transition-all duration-300 ${
            running
              ? "bg-rose-500/80 hover:bg-rose-500 shadow-[0_0_15px_rgba(244,63,94,0.4)]"
              : "bg-gradient-to-r from-violet-600 to-indigo-500 shadow-[0_0_15px_rgba(99,102,241,0.4)] hover:shadow-[0_0_25px_rgba(99,102,241,0.7)]"
          }`}
        >
          {starting ? "Initializing..." : running ? "Stop Simulation" : "Start Auto-Resolution"}
        </button>
      </div>
      {configError && (
        <div className="bg-rose-500/10 border border-rose-500/30 rounded-xl px-4 py-3 text-xs font-semibold text-rose-300">
          {configError}
        </div>
      )}
      {startBlocked && !configError && (
        <div className="bg-amber-500/10 border border-amber-500/30 rounded-xl px-4 py-3 text-xs font-semibold text-amber-300">
          Select an available RL model checkpoint before starting `trained_rl` mode.
        </div>
      )}

      {/* --- Progress bar (only visible while running) --- */}
      {(running || currentStep > 0) && (
        <div className="bg-slate-900/60 border border-white/5 rounded-xl px-5 py-3">
          <div className="flex justify-between items-center mb-2">
            <span className="text-xs font-semibold text-slate-400 uppercase tracking-widest">
              {running ? "Simulation In Progress" : journeyStats ? "Episode Complete" : "Stopped"}
            </span>
            <span className="text-xs font-black text-white">
              Step {currentStep} / {maxSteps} - {progressPct}%
            </span>
          </div>
          <div className="w-full bg-slate-800 rounded-full h-2 overflow-hidden">
            <div
              className={`h-2 rounded-full transition-all duration-500 ${
                journeyStats ? "bg-emerald-500" : "bg-indigo-500"
              } ${running ? "animate-pulse" : ""}`}
              style={{ width: `${progressPct}%` }}
            />
          </div>
          {running && (
            <div className="flex items-center gap-1.5 mt-2">
              <div className="w-1.5 h-1.5 bg-indigo-400 rounded-full animate-bounce" style={{ animationDelay: "0ms" }} />
              <div className="w-1.5 h-1.5 bg-indigo-400 rounded-full animate-bounce" style={{ animationDelay: "150ms" }} />
              <div className="w-1.5 h-1.5 bg-indigo-400 rounded-full animate-bounce" style={{ animationDelay: "300ms" }} />
              <span className="text-xs text-slate-500 ml-1">Agent is making decisions...</span>
            </div>
          )}
        </div>
      )}

      {/* --- Journey Summary (Before -> After) - appears after episode completes --- */}
      {journeyStats && (
        <div className="bg-gradient-to-br from-slate-900 to-indigo-950/30 border border-indigo-500/20 rounded-xl p-5 shadow-[0_0_30px_rgba(99,102,241,0.08)]">
          <div className="flex items-center gap-2 mb-4">
            <span className="material-symbols-outlined text-indigo-400">auto_graph</span>
            <h3 className="text-base font-black text-white">Journey Summary - Start to End Transformation</h3>
          </div>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            {[
              {
                label: "Backlog Change",
                before: journeyStats.initialBacklog,
                after: journeyStats.finalBacklog,
                suffix: " cases",
                goodWhenDown: true,
              },
              {
                label: "SLA Breaches",
                before: journeyStats.initialSla,
                after: journeyStats.finalSla,
                suffix: "",
                goodWhenDown: true,
              },
              {
                label: "Steps Taken",
                before: null,
                after: journeyStats.totalSteps,
                suffix: "",
                goodWhenDown: false,
                singleValue: true,
              },
              {
                label: "Final Score",
                before: journeyStats.finalScore != null ? "No Agent (0.0%)" : "N/A",
                after: journeyStats.finalScore != null ? `${(journeyStats.finalScore * 100).toFixed(1)}%` : "N/A",
                suffix: "",
                goodWhenDown: false,
                isScore: true,
                isBaselineCmp: true,
              },
            ].map((stat) => {
              const delta = stat.singleValue ? null : stat.isBaselineCmp ? (journeyStats.finalScore * 100) : stat.after - stat.before;
              const trend =
                delta === null
                  ? "none"
                  : delta === 0
                    ? "stable"
                    : stat.goodWhenDown
                      ? (delta < 0 ? "improving" : "worsening")
                      : (delta > 0 ? "improving" : "worsening");
              const direction =
                delta === null || delta === 0
                  ? "stable"
                  : stat.goodWhenDown
                    ? (delta < 0 ? "down" : "up")
                    : (delta > 0 ? "up" : "down");
              const directionIcon =
                direction === "up"
                  ? "north"
                  : direction === "down"
                    ? "south"
                    : "horizontal_rule";
              const trendClass =
                trend === "improving"
                  ? "text-emerald-400"
                  : trend === "worsening"
                    ? "text-rose-400"
                    : "text-slate-300";
              return (
                <div key={stat.label} className="bg-slate-800/60 border border-white/5 rounded-lg p-3">
                  <div className="text-xs font-semibold text-slate-400 mb-2 tracking-wide">{stat.label}</div>
                  {stat.singleValue ? (
                    <div className={`text-2xl font-black ${stat.isScore ? "text-emerald-400" : "text-white"}`}>{stat.after}{stat.suffix}</div>
                  ) : (
                    <div className="flex items-center gap-2">
                      <span className="text-slate-500 text-sm font-bold truncate">
                        {stat.isBaselineCmp ? "Baseline" : stat.before}{stat.suffix}
                      </span>
                      <span className="material-symbols-outlined text-slate-600 text-base">arrow_forward</span>
                      <span className={`text-xl font-black ${trendClass}`}>
                        {stat.after}{stat.suffix}
                      </span>
                    </div>
                  )}
                  {delta !== null && (
                    <div className={`text-xs font-bold mt-1 ${trendClass} inline-flex items-center gap-1`}>
                      <span className="material-symbols-outlined text-[14px] leading-none">{directionIcon}</span>
                      <span>{Number(Math.abs(delta).toFixed(2))} {trend === "stable" ? "no change" : trend}</span>
                    </div>
                  )}
                  {stat.label === "Backlog Change" && journeyStats.backlogImprovement !== 0 && (
                    <div className="text-[10px] text-slate-500 mt-0.5">
                      {journeyStats.backlogImprovement > 0 ? `${journeyStats.backlogImprovement}% cleared` : `${Math.abs(journeyStats.backlogImprovement)}% grew`}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* --- KPI Row --- */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {[
          { label: "Total Backlog", value: fmt2(kpis.backlog), delta: kpis.backlogDelta, accent: "rose", icon: "inbox" },
          { label: "SLA Breaches", value: fmt2(kpis.slaBreaches), delta: kpis.slaDelta, accent: "amber", icon: "timer_off" },
          { label: "Fairness Gap", value: `${(Number(kpis.fairness) * 100).toFixed(1)}%`, delta: kpis.fairnessDelta, accent: "emerald", icon: "balance" },
        ].map((kpi) => {
          const delta = Number(kpi.delta ?? 0);
          const trend = delta < 0 ? "down" : delta > 0 ? "up" : "stable";
          const trendIcon = trend === "up" ? "north" : trend === "down" ? "south" : "horizontal_rule";
          const badgeClass =
            trend === "down"
              ? "bg-emerald-500/20 text-emerald-400"
              : trend === "up"
                ? "bg-rose-500/20 text-rose-400"
                : "bg-slate-500/20 text-slate-300";
          return (
            <div key={kpi.label} className="bg-slate-900/70 border border-white/5 backdrop-blur-md p-5 rounded-xl relative overflow-hidden group hover:border-white/10 transition-colors">
              <div className={`absolute -right-3 -top-3 w-20 h-20 bg-${kpi.accent}-500/10 rounded-full blur-2xl`} />
              <div className="flex justify-between items-start mb-2">
                <div className="flex items-center gap-1.5">
                  <span className={`material-symbols-outlined text-${kpi.accent}-400 text-base`}>{kpi.icon}</span>
                  <span className="text-xs font-semibold tracking-widest text-slate-400 uppercase">{kpi.label}</span>
                </div>
                <span className={`text-xs font-bold px-2 py-0.5 rounded-full ${badgeClass} inline-flex items-center gap-1`}>
                  <span className="material-symbols-outlined text-[14px] leading-none">{trendIcon}</span>
                  <span>{fmtDelta(delta)}</span>
                </span>
              </div>
              <div className="text-4xl font-black text-white">{kpi.value}</div>
              <div className="text-xs text-slate-500 mt-1">
                {trend === "down" ? "Trend improving" : trend === "stable" ? "Stable" : "Trend worsening"}
              </div>
            </div>
          );
        })}
      </div>

      {/* --- Story Timeline + Queue Monitors --- */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-4">
        {/* Story Timeline */}
        <div className="lg:col-span-7 bg-slate-900/70 border border-white/5 backdrop-blur-md rounded-xl p-6 min-h-[420px]">
          <h2 className="text-lg font-bold text-white mb-5 flex items-center gap-2">
            <span className="material-symbols-outlined text-indigo-400">auto_stories</span> Story Timeline
            {timeline.length > 1 && (
              <span className="ml-auto text-xs text-slate-500">{timeline.filter(e => e.key).length} key moments</span>
            )}
          </h2>

          {timeline.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-64 text-slate-500">
              <span className="material-symbols-outlined text-5xl mb-3 opacity-30">play_circle</span>
              <p className="text-center text-sm">
                Select a scenario, set the number of steps, and press{" "}
                <strong className="text-white">Start Auto-Resolution</strong> to begin.
              </p>
            </div>
          ) : (
            <div className="relative pl-8 space-y-4 before:absolute before:inset-0 before:ml-[1.125rem] before:-translate-x-px before:h-full before:w-0.5 before:bg-gradient-to-b before:from-indigo-500/60 before:to-transparent max-h-[520px] overflow-y-auto pr-1">
              {annotatedTimeline.map((ev, idx) => {
                // Phase divider
                if (ev._divider) {
                  const ph = PHASE_LABELS[ev.phase];
                  return (
                    <div key={ev.key} className="relative flex items-center gap-3 mt-6 mb-2">
                      <div className={`absolute left-[-2.2rem] w-9 h-9 bg-slate-900 rounded-full border border-${ph.color}-500/40 flex items-center justify-center z-10`}>
                        <span className={`material-symbols-outlined text-[14px] text-${ph.color}-400`}>{ph.icon}</span>
                      </div>
                      <div className={`ml-2 text-xs font-black text-${ph.color}-400 tracking-widest uppercase border-b border-${ph.color}-500/20 pb-1 flex-1`}>
                        {ph.label}
                        <span className="font-normal text-slate-500 normal-case tracking-normal ml-2">- {ph.desc}</span>
                      </div>
                    </div>
                  );
                }

                // Phase summary block
                if (ev._summary) {
                  const drop = Math.abs(ev.stats.drop || 0);
                  const isDrop = (ev.stats.drop || 0) < 0;
                  return (
                    <div key={ev.key} className="relative pl-12 py-2">
                      <div className="bg-slate-800/40 rounded-lg p-3 inline-flex items-center gap-6 border border-white/5">
                        <div>
                          <span className="text-[10px] text-slate-500 uppercase tracking-widest block mb-0.5">Phase Backlog Move</span>
                          <span className={`text-sm font-black ${isDrop ? "text-emerald-400" : ev.stats.drop > 0 ? "text-rose-400" : "text-slate-300"}`}>
                            {isDrop ? "down " : ev.stats.drop > 0 ? "up " : ""}{drop} cases
                          </span>
                        </div>
                        <div>
                          <span className="text-[10px] text-slate-500 uppercase tracking-widest block mb-0.5">Key Decisions</span>
                          <span className="text-sm font-black text-indigo-300">{ev.stats.keys}</span>
                        </div>
                      </div>
                    </div>
                  );
                }

                const color = ev.type === "error" ? "rose" : ev.type === "warning" ? "amber" : ev.type === "success" ? "emerald" : "indigo";
                return (
                  <div
                    key={`${ev.id}-${idx}`}
                    className="relative group"
                    style={{ animation: `fadeUp 0.25s ease-out ${Math.min(idx, 10) * 0.03}s both` }}
                  >
                    <div className={`absolute left-[-2.2rem] w-9 h-9 bg-slate-900 rounded-full border border-${color}-500/40 flex items-center justify-center z-10 group-hover:border-${color}-400 transition-colors ${ev.key ? `shadow-[0_0_10px_rgba(99,102,241,0.3)]` : ""}`}>
                      <span className={`material-symbols-outlined text-[16px] text-${color}-400`}>{ev.icon}</span>
                    </div>
                    <div className={`bg-slate-800/50 border rounded-lg p-3 hover:bg-white/5 transition-colors ${ev.key ? `border-${color}-500/30 shadow-[0_0_12px_rgba(99,102,241,0.08)]` : "border-white/5"}`}>
                      <div className="flex justify-between items-start gap-3">
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center gap-2 mb-0.5">
                            <span className={`text-xs font-bold text-${color}-400`}>{ev.time}</span>
                            {ev.outcomeLabel && (
                              <span
                                className={`text-[10px] font-bold px-1.5 py-0.5 rounded ${
                                  ev.outcomeType === "success"
                                    ? "bg-emerald-500/20 text-emerald-300"
                                    : ev.outcomeType === "warning"
                                      ? "bg-amber-500/20 text-amber-300"
                                      : "bg-slate-600/20 text-slate-300"
                                }`}
                              >
                                {ev.outcomeLabel}
                              </span>
                            )}
                            {ev.key && (
                              <span className="text-[10px] font-black bg-indigo-500/20 text-indigo-300 px-1.5 py-0.5 rounded tracking-wider">
                                KEY MOMENT
                              </span>
                            )}
                            {ev._count > 1 && (
                              <span className="text-[10px] font-bold bg-slate-700 text-slate-400 px-1.5 py-0.5 rounded">
                                x{ev._count}
                              </span>
                            )}
                          </div>
                          <h4 className="font-bold text-white text-sm flex items-center gap-1.5">
                            {ev.title}
                            {ev.isHugeImpact && <span title="Massive Improvement" className="text-sm">High Impact</span>}
                            {ev.isHighReward && <span title="High Reward Action" className="text-sm">Hot</span>}
                          </h4>
                          <p className="text-xs text-slate-400 mt-1 leading-relaxed">{ev.desc}</p>
                          {ev.reason && (
                            <div className="mt-2 bg-indigo-500/10 border-l-2 border-indigo-500/30 pl-2 py-1 text-xs text-indigo-200/80">
                              <span className="font-semibold text-indigo-300">Agent Reasoning:</span> {ev.reason}
                            </div>
                          )}
                        </div>
                        {ev.impact !== 0 && (
                          <div className={`shrink-0 bg-${color}-500/10 border border-${color}-500/20 px-2 py-1 rounded text-xs font-bold text-${color}-400 whitespace-nowrap`}>
                            {Number(ev.impact) >= 0 ? "+" : ""}{Number(ev.impact).toFixed(2)}
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>

        {/* Live Queue Monitors */}
        <div className="lg:col-span-5 bg-slate-900/70 border border-white/5 backdrop-blur-md rounded-xl p-6">
          <h2 className="text-lg font-bold text-white mb-5 flex items-center gap-2">
            <span className="material-symbols-outlined text-emerald-400">monitor_heart</span> Live Queue Monitors
          </h2>
          {resources.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-48 text-slate-500">
              <span className="material-symbols-outlined text-4xl mb-2 opacity-30">sensors</span>
              <p className="text-sm">Awaiting live telemetry...</p>
            </div>
          ) : (
            <div className="space-y-5">
              {resources.map((res, i) => {
                const color = res.percentage > 85 ? "rose" : res.percentage > 60 ? "amber" : "emerald";
                const tone = color === "rose"
                  ? {
                      text: "text-rose-400",
                      bar: "bg-rose-500",
                    }
                  : color === "amber"
                    ? {
                        text: "text-amber-400",
                        bar: "bg-amber-500",
                      }
                    : {
                        text: "text-emerald-400",
                        bar: "bg-emerald-500",
                      };
                return (
                  <div key={res.name || i}>
                    <div className="flex justify-between mb-1.5">
                      <span className="text-sm font-semibold text-white">{res.name}</span>
                      <div className="flex items-center gap-2">
                        <span className={`text-xs font-bold ${tone.text}`}>{res.activeCases} active</span>
                        {res.percentage > 85 && (
                          <span className="text-[10px] font-black text-rose-400 bg-rose-500/10 px-1.5 rounded">OVERLOADED</span>
                        )}
                      </div>
                    </div>
                    <div className="w-full bg-slate-800 rounded-full h-2.5 overflow-hidden">
                      <div
                        className={`${tone.bar} h-full rounded-full transition-all duration-700 ease-in-out`}
                        style={{ width: `${res.percentage}%` }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          )}

          {/* Reward cumulative tracker - shown after first step */}
          {currentStep > 0 && (
            <div className="mt-6 pt-5 border-t border-white/5">
              <div className="text-xs font-semibold text-slate-400 mb-3 uppercase tracking-widest">Impact Summary</div>
              <div className="grid grid-cols-2 gap-3">
                <div className="bg-slate-800/60 rounded-lg p-3 text-center">
                  <div className="text-xs text-slate-400 mb-1">Steps Elapsed</div>
                  <div className="text-xl font-black text-white">{currentStep}</div>
                </div>
                <div className="bg-slate-800/60 rounded-lg p-3 text-center">
                  <div className="text-xs text-slate-400 mb-1">Key Moments</div>
                  <div className="text-xl font-black text-indigo-300">
                    {timeline.filter((e) => e.key).length}
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}


// --- Resources Tab ------------------------------------------------------------
function BenchmarkResults({ results }) {
  const COLORS = { backlog_clearance: "#6366f1", urgent_first: "#10b981", oldest_first: "#f59e0b" };
  const sorted = [...results.agent_results].sort((a, b) => b.average_score - a.average_score);
  const winner = sorted[0];
  const maxScore = Math.max(...results.agent_results.map((a) => a.average_score), 0.001);
  const chartH = 140;

  return (
    <div className="space-y-5">
      {/* Winner callout */}
      <div className="bg-emerald-500/10 border border-emerald-500/30 rounded-xl p-5 flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-center gap-4">
          <span className="material-symbols-outlined text-emerald-400 text-4xl">emoji_events</span>
          <div>
            <div className="text-xs font-black text-emerald-400 tracking-widest mb-1">BEST PERFORMING POLICY</div>
            <div className="text-xl font-black text-white capitalize">{winner.agent_policy.replace(/_/g, " ")}</div>
            <div className="text-sm text-slate-400 mt-0.5">
              Avg score{" "}<span className="text-emerald-400 font-bold">{(winner.average_score * 100).toFixed(1)}%</span>
              {" | "}Range {(winner.min_score * 100).toFixed(0)}%-{(winner.max_score * 100).toFixed(0)}%
            </div>
          </div>
        </div>
        <div className="bg-emerald-500/10 border border-emerald-500/20 px-3 py-2 rounded-lg max-w-sm hidden lg:block">
          <div className="text-xs font-bold text-emerald-400 mb-1 flex items-center gap-1">
            <span className="material-symbols-outlined text-[14px]">psychology</span> Agent Intelligence
          </div>
          <p className="text-[10px] text-emerald-200/80 leading-relaxed font-medium">
            This policy performed best by maintaining fewer SLA breaches relative to its peers while securing steady backlog reduction across critical queues.
          </p>
        </div>
      </div>

      {/* Bar chart */}
      <div className="bg-slate-900/70 border border-white/5 rounded-xl p-6">
        <h3 className="text-sm font-bold text-white mb-6">Average Grader Score by Policy</h3>
        <div className="flex items-end justify-center gap-10">
          {sorted.map((agent) => {
            const pct = agent.average_score / maxScore;
            const barH = Math.max(Math.round(pct * chartH), 6);
            const color = COLORS[agent.agent_policy] || "#6366f1";
            const isWinner = agent.agent_policy === winner.agent_policy;
            return (
              <div key={agent.agent_policy} className="flex flex-col items-center gap-2 w-28">
                <div className="text-base font-black text-white">{(agent.average_score * 100).toFixed(1)}%</div>
                <div className="relative w-full flex items-end justify-center" style={{ height: chartH }}>
                  {isWinner && <div className="absolute -top-5 left-1/2 -translate-x-1/2 text-lg text-emerald-400">Top</div>}
                  <div
                    className="w-full rounded-t-lg transition-all duration-700"
                    style={{
                      height: barH,
                      background: `linear-gradient(to top, ${color}88, ${color})`,
                      boxShadow: isWinner ? `0 0 24px ${color}60` : "none",
                    }}
                  />
                </div>
                <div className="text-xs font-semibold text-center leading-tight" style={{ color }}>
                  {agent.agent_policy.replace(/_/g, " ")}
                </div>
                <div className="text-xs text-slate-500">{agent.runs.length} runs</div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Multi-metric comparison bars */}
      <div className="bg-slate-900/70 border border-white/5 rounded-xl p-6">
        <h3 className="text-sm font-bold text-white mb-5">Metric Comparison</h3>
        <div className="space-y-6">
          {[
            {
              label: "Score (higher is better)",
              vals: results.agent_results.map((a) => ({ key: a.agent_policy, v: a.average_score, display: `${(a.average_score * 100).toFixed(1)}%` })),
              higherGood: true,
            },
            {
              label: "Avg Completed Cases (higher is better)",
              vals: results.agent_results.map((a) => {
                const avg = a.runs.reduce((s, r) => s + (r.completed ?? 0), 0) / Math.max(a.runs.length, 1);
                return { key: a.agent_policy, v: avg, display: avg.toFixed(1) };
              }),
              higherGood: true,
            },
            {
              label: "Avg Remaining Backlog (lower is better)",
              vals: results.agent_results.map((a) => {
                const avg = a.runs.reduce((s, r) => s + (r.backlog ?? 0), 0) / Math.max(a.runs.length, 1);
                return { key: a.agent_policy, v: avg, display: avg.toFixed(1) };
              }),
              higherGood: false,
            },
          ].map(({ label, vals, higherGood }) => {
            const maxVal = Math.max(...vals.map((v) => v.v), 0.001);
            const best = higherGood
              ? vals.reduce((a, b) => (b.v > a.v ? b : a))
              : vals.reduce((a, b) => (b.v < a.v ? b : a));
            return (
              <div key={label}>
                <div className="text-xs font-bold text-slate-400 mb-3">{label}</div>
                <div className="space-y-2">
                  {vals.map((v) => {
                    const pct = Math.round((v.v / maxVal) * 100);
                    const color = (COLORS)[v.key] || "#6366f1";
                    return (
                      <div key={v.key} className="flex items-center gap-3">
                        <div className="w-36 text-xs text-slate-300 capitalize shrink-0 flex items-center gap-1">
                          {v.key.replace(/_/g, " ")}
                          {v.key === best.key && <span className="text-[10px] font-black text-emerald-400">Top</span>}
                        </div>
                        <div className="flex-1 bg-slate-800 rounded-full h-2.5 overflow-hidden">
                          <div className="h-2.5 rounded-full transition-all duration-700" style={{ width: `${pct}%`, backgroundColor: color }} />
                        </div>
                        <div className="w-14 text-right text-xs font-bold text-white">{v.display}</div>
                      </div>
                    );
                  })}
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Raw episode table */}
      <div className="bg-slate-900/70 border border-white/5 rounded-xl p-6">
        <h3 className="text-sm font-bold text-white mb-4">All Episodes - Raw Data</h3>
        <div className="overflow-x-auto">
          <table className="w-full text-xs text-left">
            <thead>
              <tr className="text-slate-400 border-b border-white/5">
                <th className="pb-2 pr-4">Policy</th>
                <th className="pb-2 pr-4">Run #</th>
                <th className="pb-2 pr-4">Score</th>
                <th className="pb-2 pr-4">Reward</th>
                <th className="pb-2 pr-4">Completed</th>
                <th className="pb-2 pr-4">Backlog</th>
                <th className="pb-2">Steps</th>
              </tr>
            </thead>
            <tbody>
              {results.agent_results.flatMap((agent) =>
                agent.runs.map((run) => (
                  <tr key={`${agent.agent_policy}-${run.run_index}`} className="border-b border-white/5 hover:bg-white/5">
                    <td className="py-2 pr-4 font-medium" style={{ color: (COLORS)[agent.agent_policy] || "#6366f1" }}>
                      {agent.agent_policy.replace(/_/g, " ")}
                    </td>
                    <td className="py-2 pr-4 text-slate-400">#{run.run_index}</td>
                    <td className="py-2 pr-4 font-bold text-white">{(run.score * 100).toFixed(1)}%</td>
                    <td className="py-2 pr-4 text-amber-400">{run.reward_sum?.toFixed(2) ?? "-"}</td>
                    <td className="py-2 pr-4 text-emerald-400">{run.completed ?? "-"}</td>
                    <td className="py-2 pr-4 text-rose-400">{run.backlog ?? "-"}</td>
                    <td className="py-2 text-slate-400">{run.steps ?? "-"}</td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

function ResourcesTab({ tasks }) {
  const [benchTask, setBenchTask] = useState(tasks[0] || "district_backlog_easy");
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState("");

  const runBenchmark = async () => {
    setLoading(true);
    setError("");
    setResults(null);
    try {
      const data = await api("/benchmark", {
        method: "POST",
        body: JSON.stringify({
          task_id: benchTask,
          agent_policies: ["backlog_clearance", "urgent_first", "oldest_first"],
          runs: 3,
          max_steps: 60,
        }),
      });
      setResults(data);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <div className="bg-slate-900/70 border border-white/5 rounded-xl p-6">
        <h2 className="text-lg font-bold text-white mb-1 flex items-center gap-2">
          <span className="material-symbols-outlined text-violet-400">leaderboard</span> Policy Benchmark Comparison
        </h2>
        <p className="text-sm text-slate-400 mb-5">
          Run all three baseline policies on the same scenario and compare their grader scores,
          completed cases, and remaining backlogs side-by-side with visual charts.
        </p>
        <div className="flex flex-wrap gap-3 items-center">
          <select
            value={benchTask}
            onChange={(e) => setBenchTask(e.target.value)}
            className="appearance-none bg-slate-800 border border-white/10 text-sm font-medium px-3 py-1.5 rounded-lg text-indigo-300 focus:outline-none focus:border-indigo-500"
          >
            {tasks.map((t) => (
              <option key={t} value={t} className="bg-slate-900">
                {t.replace(/_/g, " ").toUpperCase()}
              </option>
            ))}
          </select>
          <button
            onClick={runBenchmark}
            disabled={loading}
            className="bg-violet-600 hover:bg-violet-500 text-white text-sm font-bold px-5 py-2 rounded-lg transition-all disabled:opacity-50"
          >
            {loading ? "Simulating 9 episodes..." : "Run Benchmark"}
          </button>
        </div>
      </div>

      {error && (
        <div className="bg-rose-500/10 border border-rose-500/30 rounded-xl p-4 text-rose-400 text-sm">
          {error}
        </div>
      )}

      {loading && (
        <div className="bg-slate-900/70 border border-white/5 rounded-xl p-10 flex flex-col items-center gap-4">
          <div className="w-10 h-10 border-4 border-indigo-500 border-t-transparent rounded-full animate-spin" />
          <p className="text-slate-400 text-sm">Running 3 policies x 3 episodes each - takes ~20 seconds.</p>
        </div>
      )}

      {results && <BenchmarkResults results={results} />}
    </div>
  );
}

// --- Library Tab --------------------------------------------------------------
function LibraryTab({ tasks }) {
  const [compliance, setCompliance] = useState(null);
  const [workflows, setWorkflows] = useState(null);
  const [selected, setSelected] = useState(null);

  useEffect(() => {
    api("/openenv_compliance").then(setCompliance).catch(() => {});
    api("/workflows/components").then(setWorkflows).catch(() => {});
  }, []);

  const taskDetails = {
    district_backlog_easy: { diff: "Easy", desc: "Single-service district queue focused on income certificate flow.", services: 1 },
    mixed_urgency_medium: { diff: "Medium", desc: "Income, land, passport, driving license, and Aadhaar workloads with mixed urgency.", services: 5 },
    cross_department_hard: { diff: "Hard", desc: "Five-service crisis mode with high arrivals, fairness pressure, and event shocks.", services: 5 },
  };

  const systemTabGuide = [
    {
      id: "timeline",
      title: "Simulation (Timeline Tab)",
      icon: "timeline",
      summary: "Runs live step-by-step environment simulation and shows queue movement, KPI changes, and decision timeline in real time.",
      userFlow: "Choose scenario, steps, and model/policy, then start auto-resolution.",
      outputs: "Live backlog, SLA, fairness, key moments, queue pressure bars, and impact summary.",
      endpoints: ["/simulation/live/start", "/simulation/live/step", "/simulation/live/{run_id}/stop", "/tasks", "/agents", "/rl_models", "/rl/models"],
    },
    {
      id: "training",
      title: "Training Tab",
      icon: "fitness_center",
      summary: "Controls RL training jobs and tracks how the policy improves over timesteps.",
      userFlow: "Start/stop a training job and monitor live checkpoints and job history.",
      outputs: "Active job state, progress, reward/score checkpoints, sequential narrative feed, and OpenEnv contract replay results.",
      endpoints: ["/training_jobs", "/training_jobs/list", "/training_jobs/{job_id}", "/training_jobs/{job_id}/stop", "/reset", "/step", "/state", "/grade"],
    },
    {
      id: "analytics",
      title: "Analytics Tab",
      icon: "analytics",
      summary: "Shows endpoint-fed system analytics from historical simulation, jobs, models, sessions, and compliance health.",
      userFlow: "Open the tab; metrics auto-refresh from backend every few seconds.",
      outputs: "Task distributions, mode splits, training status mix, endpoint health, model inventory, and run history tables.",
      endpoints: ["/history/simulations", "/history/comparisons", "/training_jobs", "/rl_models", "/rl/models", "/tasks", "/agents", "/sessions", "/actions/schema", "/openenv_compliance", "/workflows/components"],
    },
    {
      id: "resources",
      title: "Resources Tab (Benchmark)",
      icon: "leaderboard",
      summary: "Compares baseline policies on the same task to identify which strategy performs best.",
      userFlow: "Select a scenario and run benchmark.",
      outputs: "Winner policy card, score bars, metric comparison bars, and raw run-level benchmark table.",
      endpoints: ["/compare_agents"],
    },
    {
      id: "library",
      title: "Library Tab",
      icon: "menu_book",
      summary: "Acts as the complete system overview and reference center for tasks, compliance, and workflow availability.",
      userFlow: "Explore scenarios, inspect OpenEnv checks, and verify available workflow components.",
      outputs: "Task cards with difficulty/service counts, compliance checklist, and component readiness matrix.",
      endpoints: ["/tasks", "/openenv_compliance", "/workflows/components"],
    },
  ];

  return (
    <div className="space-y-6">
      <div className="bg-slate-900/70 border border-white/5 rounded-xl p-6">
        <h2 className="text-lg font-bold text-white mb-2 flex items-center gap-2">
          <span className="material-symbols-outlined text-violet-400">hub</span> Complete System Overview
        </h2>
        <p className="text-sm text-slate-400 mb-5">
          This section explains how each product tab works, what backend APIs power it, and what outputs users can expect.
          Use it as a quick guide for judges and reviewers.
        </p>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          {systemTabGuide.map((tab) => (
            <div key={tab.id} className="bg-slate-800/50 border border-white/5 rounded-xl p-4">
              <div className="flex items-center gap-2 mb-2">
                <span className="material-symbols-outlined text-indigo-300">{tab.icon}</span>
                <h3 className="text-sm font-bold text-white">{tab.title}</h3>
              </div>
              <p className="text-xs text-slate-300 leading-relaxed mb-2">{tab.summary}</p>
              <div className="text-xs text-slate-400 mb-1">
                <span className="text-slate-300 font-semibold">User flow:</span> {tab.userFlow}
              </div>
              <div className="text-xs text-slate-400 mb-3">
                <span className="text-slate-300 font-semibold">Outputs:</span> {tab.outputs}
              </div>
              <div className="flex flex-wrap gap-1.5">
                {tab.endpoints.map((ep) => (
                  <code key={ep} className="text-[10px] text-cyan-300 bg-slate-900 px-1.5 py-0.5 rounded border border-cyan-500/20">
                    {ep}
                  </code>
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>

      <div>
        <h2 className="text-lg font-bold text-white mb-4 flex items-center gap-2">
          <span className="material-symbols-outlined text-amber-400">menu_book</span> Scenario Library
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {tasks.map((t) => {
            const info = taskDetails[t] || { diff: "-", desc: "Custom scenario.", services: "-" };
            const diffColor = info.diff === "Easy" ? "emerald" : info.diff === "Medium" ? "amber" : "rose";
            const isSelected = selected === t;
            return (
              <button
                key={t}
                onClick={() => setSelected(isSelected ? null : t)}
                className={`text-left bg-slate-900/70 border rounded-xl p-5 transition-all hover:border-indigo-500/40 ${isSelected ? "border-indigo-500/60 shadow-[0_0_20px_rgba(99,102,241,0.15)]" : "border-white/5"}`}
              >
                <div className="flex justify-between items-start mb-3">
                  <div className={`text-xs font-black tracking-widest text-${diffColor}-400`}>{info.diff.toUpperCase()}</div>
                  <span className="material-symbols-outlined text-slate-500 text-lg">{isSelected ? "expand_less" : "expand_more"}</span>
                </div>
                <h3 className="font-bold text-white text-sm mb-2">{t.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase())}</h3>
                <p className="text-xs text-slate-400 leading-relaxed">{info.desc}</p>
                {isSelected && (
                  <div className="mt-4 pt-4 border-t border-white/5 space-y-2">
                    <div className="flex justify-between text-xs"><span className="text-slate-400">Services</span><span className="text-white font-bold">{info.services}</span></div>
                    <div className="flex justify-between text-xs"><span className="text-slate-400">Difficulty</span><span className="text-white font-bold">{info.diff}</span></div>
                    <div className="flex justify-between text-xs"><span className="text-slate-400">Task ID</span><span className="text-indigo-300 font-mono">{t}</span></div>
                  </div>
                )}
              </button>
            );
          })}
        </div>
      </div>

      {compliance && (
        <div className="bg-slate-900/70 border border-white/5 rounded-xl p-6">
          <h2 className="text-lg font-bold text-white mb-4 flex items-center gap-2">
            <span className="material-symbols-outlined text-indigo-400">verified</span> OpenEnv Compliance Status
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {compliance.items?.map((item) => (
              <div key={item.key} className={`flex items-start gap-3 bg-slate-800/50 border rounded-lg p-3 ${item.status === "pass" ? "border-emerald-500/25" : "border-rose-500/25"}`}>
                <span className={`material-symbols-outlined text-lg shrink-0 ${item.status === "pass" ? "text-emerald-400" : item.status === "fail" ? "text-rose-400" : "text-amber-400"}`}>
                  {item.status === "pass" ? "check_circle" : item.status === "fail" ? "cancel" : "help"}
                </span>
                <div>
                  <div className="text-sm font-semibold text-white">{item.label}</div>
                  <div className="text-xs text-slate-400 mt-0.5">{item.detail}</div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {workflows && (
        <div className="bg-slate-900/70 border border-white/5 rounded-xl p-6">
          <h2 className="text-lg font-bold text-white mb-4 flex items-center gap-2">
            <span className="material-symbols-outlined text-cyan-400">account_tree</span> Workflow Components
          </h2>
          <div className="space-y-3">
            {workflows.components?.map((c) => (
              <div key={c.component} className={`flex items-center gap-4 bg-slate-800/50 border rounded-lg p-3 ${c.available ? "border-emerald-500/20" : "border-slate-700"}`}>
                <span className={`material-symbols-outlined text-lg ${c.available ? "text-emerald-400" : "text-slate-600"}`}>
                  {c.available ? "check_box" : "check_box_outline_blank"}
                </span>
                <div className="flex-1 min-w-0">
                  <div className="text-sm font-bold text-white">{c.component}</div>
                  <div className="text-xs text-slate-400 truncate">{c.description}</div>
                </div>
                {c.command && (
                  <code className="text-xs text-indigo-300 bg-slate-900 px-2 py-1 rounded font-mono hidden lg:block max-w-xs truncate">{c.command}</code>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

// --- Analytics Tab ------------------------------------------------------------
function AnalyticsTab() {
  const [history, setHistory] = useState([]);
  const [rlModels, setRlModels] = useState([]);
  const [rlModelsV2, setRlModelsV2] = useState([]);
  const [trainingJobs, setTrainingJobs] = useState([]);
  const [tasksList, setTasksList] = useState([]);
  const [agentsList, setAgentsList] = useState([]);
  const [sessionsInfo, setSessionsInfo] = useState({ active_sessions: 0, session_ids: [] });
  const [actionsSchema, setActionsSchema] = useState({});
  const [complianceInfo, setComplianceInfo] = useState({ items: [] });
  const [workflowInfo, setWorkflowInfo] = useState({ components: [] });
  const [comparisonsInfo, setComparisonsInfo] = useState({ comparisons: [] });
  const [endpointHealth, setEndpointHealth] = useState([]);
  const [loadingHistory, setLoadingHistory] = useState(true);
  const [loadingAll, setLoadingAll] = useState(true);

  useEffect(() => {
    let cancelled = false;

    const load = async () => {
      setLoadingHistory(true);
      setLoadingAll(true);
      try {
        const [
          historyRes,
          rlRes,
          rlResV2,
          jobsRes,
          tasksRes,
          agentsRes,
          sessionsRes,
          actionsRes,
          complianceRes,
          workflowsRes,
          comparisonsRes,
        ] = await Promise.allSettled([
          api("/history/simulations?limit=80"),
          api("/rl_models"),
          api("/rl/models"),
          api("/training_jobs"),
          api("/tasks"),
          api("/agents"),
          api("/sessions"),
          api("/actions/schema"),
          api("/openenv_compliance"),
          api("/workflows/components"),
          api("/history/comparisons?limit=30"),
        ]);

        if (cancelled) return;

        const checks = [
          { key: "history", label: "History", ok: historyRes.status === "fulfilled" },
          { key: "rl_models", label: "RL Models", ok: rlRes.status === "fulfilled" },
          { key: "rl_models_v2", label: "RL Models V2", ok: rlResV2.status === "fulfilled" },
          { key: "training_jobs", label: "Training Jobs", ok: jobsRes.status === "fulfilled" },
          { key: "tasks", label: "Tasks", ok: tasksRes.status === "fulfilled" },
          { key: "agents", label: "Agents", ok: agentsRes.status === "fulfilled" },
          { key: "sessions", label: "Sessions", ok: sessionsRes.status === "fulfilled" },
          { key: "actions_schema", label: "Action Schema", ok: actionsRes.status === "fulfilled" },
          { key: "openenv_compliance", label: "Compliance", ok: complianceRes.status === "fulfilled" },
          { key: "workflow_components", label: "Workflow Components", ok: workflowsRes.status === "fulfilled" },
          { key: "comparison_history", label: "Comparison History", ok: comparisonsRes.status === "fulfilled" },
        ];
        setEndpointHealth(checks);

        setHistory(historyRes.status === "fulfilled" ? (historyRes.value?.runs || []) : []);
        setRlModels(rlRes.status === "fulfilled" ? (rlRes.value?.models || []) : []);
        setRlModelsV2(rlResV2.status === "fulfilled" ? (Array.isArray(rlResV2.value) ? rlResV2.value : []) : []);
        setTrainingJobs(jobsRes.status === "fulfilled" ? (jobsRes.value?.jobs || []) : []);
        setTasksList(tasksRes.status === "fulfilled" ? (tasksRes.value?.tasks || []) : []);
        setAgentsList(agentsRes.status === "fulfilled" ? (Array.isArray(agentsRes.value) ? agentsRes.value : []) : []);
        setSessionsInfo(sessionsRes.status === "fulfilled" ? (sessionsRes.value || { active_sessions: 0, session_ids: [] }) : { active_sessions: 0, session_ids: [] });
        setActionsSchema(actionsRes.status === "fulfilled" ? (actionsRes.value || {}) : {});
        setComplianceInfo(complianceRes.status === "fulfilled" ? (complianceRes.value || { items: [] }) : { items: [] });
        setWorkflowInfo(workflowsRes.status === "fulfilled" ? (workflowsRes.value || { components: [] }) : { components: [] });
        setComparisonsInfo(comparisonsRes.status === "fulfilled" ? (comparisonsRes.value || { comparisons: [] }) : { comparisons: [] });
      } finally {
        if (!cancelled) {
          setLoadingHistory(false);
          setLoadingAll(false);
        }
      }
    };

    load();
    const timer = setInterval(load, 8000);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, []);

  const byTask = history.reduce((acc, run) => {
    const t = run.task_id || "unknown";
    if (!acc[t]) acc[t] = [];
    acc[t].push(run);
    return acc;
  }, {});

  const getRunScore = (run) => {
    const value = run?.score ?? run?.payload?.score;
    const num = Number(value);
    return Number.isFinite(num) ? num : null;
  };

  const getRunReward = (run) => {
    const value = run?.total_reward ?? run?.payload?.total_reward;
    const num = Number(value);
    return Number.isFinite(num) ? num : null;
  };

  const getJobProgress = (job) => {
    const p = Number(job?.progress);
    if (Number.isFinite(p)) return Math.max(0, Math.min(1, p));
    const ts = Number(job?.latest_metrics?.total_timesteps);
    const total = Number(job?.timesteps);
    if (Number.isFinite(ts) && Number.isFinite(total) && total > 0) {
      return Math.max(0, Math.min(1, ts / total));
    }
    return 0;
  };

  const scoreData = history.map(getRunScore).filter((v) => v != null);
  const avgScore = scoreData.length ? scoreData.reduce((s, v) => s + v, 0) / scoreData.length : null;
  const runningJobs = trainingJobs.filter((j) => String(j?.status || "").toLowerCase() === "running").length;
  const endpointCoverage = endpointHealth.length
    ? endpointHealth.filter((x) => x.ok).length / endpointHealth.length
    : null;

  const timelineTaskRows = Object.entries(byTask)
    .map(([label, runs]) => ({ label, value: runs.length }))
    .sort((a, b) => b.value - a.value);

  const timelineModeRows = Object.entries(
    history.reduce((acc, run) => {
      const mode = String(run?.agent_mode || "unknown");
      acc[mode] = (acc[mode] || 0) + 1;
      return acc;
    }, {})
  ).map(([label, value]) => ({ label, value }));

  const trainingStatusRows = Object.entries(
    trainingJobs.reduce((acc, job) => {
      const status = String(job?.status || "unknown").toLowerCase();
      acc[status] = (acc[status] || 0) + 1;
      return acc;
    }, {})
  ).map(([label, value]) => ({ label, value }));

  const trainingPhaseRows = [1, 2].map((phase) => {
    const rows = trainingJobs.filter((job) => Number(job?.phase || 0) === phase);
    const avgProgress = rows.length
      ? rows.reduce((sum, job) => sum + getJobProgress(job), 0) / rows.length
      : 0;
    return {
      label: `Phase ${phase}`,
      value: Number((avgProgress * 100).toFixed(1)),
      jobs: rows.length,
    };
  });

  const compliancePass = Array.isArray(complianceInfo?.items)
    ? complianceInfo.items.filter((x) => x?.status === "pass").length
    : 0;
  const complianceFail = Array.isArray(complianceInfo?.items)
    ? complianceInfo.items.filter((x) => x?.status === "fail").length
    : 0;
  const complianceUnknown = Array.isArray(complianceInfo?.items)
    ? complianceInfo.items.filter((x) => x?.status !== "pass" && x?.status !== "fail").length
    : 0;

  const systemMetricRows = [
    { label: "Tasks", value: tasksList.length },
    { label: "Agents", value: agentsList.length },
    { label: "Action Types", value: Number(actionsSchema?.total_action_types || 0) },
    { label: "Active Sessions", value: Number(sessionsInfo?.active_sessions || 0) },
    { label: "RL Models V1", value: rlModels.filter((m) => m.exists).length },
    { label: "RL Models V2", value: rlModelsV2.filter((m) => m.exists).length },
    {
      label: "Workflow Components",
      value: Array.isArray(workflowInfo?.components)
        ? workflowInfo.components.filter((x) => x?.available).length
        : 0,
    },
    { label: "Comparisons", value: Array.isArray(comparisonsInfo?.comparisons) ? comparisonsInfo.comparisons.length : 0 },
  ];

  const buildConicGradient = (rows, palette) => {
    const total = rows.reduce((sum, row) => sum + Number(row?.value || 0), 0);
    if (total <= 0) return null;
    let cursor = 0;
    const segments = [];
    rows.forEach((row, idx) => {
      const value = Number(row?.value || 0);
      if (value <= 0) return;
      const delta = (value / total) * 100;
      const start = cursor;
      const end = cursor + delta;
      segments.push(`${palette[idx % palette.length]} ${start.toFixed(2)}% ${end.toFixed(2)}%`);
      cursor = end;
    });
    if (cursor < 100) {
      segments.push(`#1e293b ${cursor.toFixed(2)}% 100%`);
    }
    return `conic-gradient(${segments.join(", ")})`;
  };

  const timelineModeGradient = buildConicGradient(
    timelineModeRows,
    ["#22d3ee", "#a78bfa", "#f59e0b", "#34d399", "#f472b6"]
  );
  const trainingStatusGradient = buildConicGradient(
    trainingStatusRows,
    ["#22c55e", "#eab308", "#6366f1", "#ef4444", "#64748b"]
  );
  const complianceGradient = buildConicGradient(
    [
      { label: "pass", value: compliancePass },
      { label: "fail", value: complianceFail },
      { label: "unknown", value: complianceUnknown },
    ],
    ["#22c55e", "#ef4444", "#f59e0b"]
  );

  const renderBars = (rows, color = "bg-indigo-500") => {
    const maxVal = Math.max(...rows.map((r) => Number(r?.value || 0)), 1);
    return (
      <div className="space-y-2">
        {rows.map((row) => {
          const widthPct = Math.max(0, Math.min(100, (Number(row.value || 0) / maxVal) * 100));
          return (
            <div key={row.label} className="space-y-1">
              <div className="flex justify-between text-xs">
                <span className="text-slate-300">{row.label.replace(/_/g, " ")}</span>
                <span className="text-white font-semibold">{Number(row.value || 0)}</span>
              </div>
              <div className="h-2 w-full rounded bg-slate-800 overflow-hidden">
                <div className={`h-full ${color}`} style={{ width: `${widthPct}%` }} />
              </div>
            </div>
          );
        })}
      </div>
    );
  };

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {[
          { label: "Total Runs", value: history.length, icon: "play_circle", color: "indigo" },
          { label: "Avg Score", value: avgScore != null ? `${(avgScore * 100).toFixed(1)}%` : "—", icon: "grade", color: "emerald" },
          { label: "Running Jobs", value: runningJobs, icon: "settings_slow_motion", color: "violet" },
          { label: "Endpoint Coverage", value: endpointCoverage != null ? `${(endpointCoverage * 100).toFixed(0)}%` : "—", icon: "hub", color: "amber" },
        ].map((s) => (
          <div key={s.label} className="bg-slate-900/70 border border-white/5 rounded-xl p-4">
            <div className="flex items-center gap-2 mb-2">
              <span className={`material-symbols-outlined text-${s.color}-400`}>{s.icon}</span>
              <span className="text-xs font-semibold text-slate-400 uppercase tracking-widest">{s.label}</span>
            </div>
            <div className="text-3xl font-black text-white">{s.value}</div>
          </div>
        ))}
      </div>

      {!loadingHistory && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="bg-slate-900/70 border border-white/5 rounded-xl p-6">
            <h2 className="text-base font-bold text-white mb-4 flex items-center gap-2">
              <span className="material-symbols-outlined text-cyan-400">bar_chart</span> Timeline Metric: Runs by Task
            </h2>
            {timelineTaskRows.length === 0 ? (
              <div className="text-xs text-slate-500">No timeline history yet.</div>
            ) : renderBars(timelineTaskRows, "bg-cyan-500")}
          </div>

          <div className="bg-slate-900/70 border border-white/5 rounded-xl p-6 h-full flex flex-col">
            <h2 className="text-base font-bold text-white mb-4 flex items-center gap-2">
              <span className="material-symbols-outlined text-violet-400">pie_chart</span> Timeline Metric: Agent Mode Mix
            </h2>
            {timelineModeGradient ? (
              <div className="grid grid-cols-[120px,1fr] gap-4 items-center">
                <div className="relative w-[120px] h-[120px] rounded-full" style={{ background: timelineModeGradient }}>
                  <div className="absolute inset-[18px] rounded-full bg-slate-950/95 border border-white/5" />
                </div>
                <div className="space-y-2">
                  {timelineModeRows.map((row, idx) => (
                    <div key={row.label} className="flex items-center justify-between text-xs">
                      <div className="flex items-center gap-2 text-slate-300">
                        <span
                          className="inline-block w-2.5 h-2.5 rounded-full"
                          style={{ backgroundColor: ["#22d3ee", "#a78bfa", "#f59e0b", "#34d399", "#f472b6"][idx % 5] }}
                        />
                        {row.label}
                      </div>
                      <span className="text-white font-semibold">{row.value}</span>
                    </div>
                  ))}
                </div>
              </div>
            ) : (
              <div className="text-xs text-slate-500">No timeline mode data yet.</div>
            )}
          </div>
        </div>
      )}

      {!loadingAll && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="bg-slate-900/70 border border-white/5 rounded-xl p-6">
            <h2 className="text-base font-bold text-white mb-4 flex items-center gap-2">
              <span className="material-symbols-outlined text-emerald-400">stacked_bar_chart</span> Training Metric: Job Status Mix
            </h2>
            {trainingStatusGradient ? (
              <div className="space-y-3">
                <div className="h-4 rounded bg-slate-800 overflow-hidden">
                  <div className="h-full" style={{ background: trainingStatusGradient }} />
                </div>
                <div className="grid grid-cols-2 gap-2">
                  {trainingStatusRows.map((row, idx) => (
                    <div key={row.label} className="flex items-center justify-between text-xs bg-slate-800/40 border border-white/5 rounded px-2 py-1">
                      <div className="flex items-center gap-2 text-slate-300">
                        <span
                          className="inline-block w-2.5 h-2.5 rounded-full"
                          style={{ backgroundColor: ["#22c55e", "#eab308", "#6366f1", "#ef4444", "#64748b"][idx % 5] }}
                        />
                        {row.label}
                      </div>
                      <span className="text-white font-semibold">{row.value}</span>
                    </div>
                  ))}
                </div>
              </div>
            ) : (
              <div className="text-xs text-slate-500">No training jobs available yet.</div>
            )}
          </div>

          <div className="bg-slate-900/70 border border-white/5 rounded-xl p-6">
            <h2 className="text-base font-bold text-white mb-4 flex items-center gap-2">
              <span className="material-symbols-outlined text-indigo-400">dataset</span> Training Metric: Phase Progress (%)
            </h2>
            <div className="space-y-3">
              {trainingPhaseRows.map((row) => (
                <div key={row.label} className="space-y-1">
                  <div className="flex justify-between text-xs">
                    <span className="text-slate-300">{row.label}</span>
                    <span className="text-white font-semibold">{row.value.toFixed(1)}% · {row.jobs} jobs</span>
                  </div>
                  <div className="h-2 w-full rounded bg-slate-800 overflow-hidden">
                    <div className="h-full bg-indigo-500" style={{ width: `${Math.max(0, Math.min(100, row.value))}%` }} />
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {!loadingAll && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="bg-slate-900/70 border border-white/5 rounded-xl p-6 h-full flex flex-col">
            <h2 className="text-base font-bold text-white mb-4 flex items-center gap-2">
              <span className="material-symbols-outlined text-cyan-400">analytics</span> System Metric: Endpoint-fed Counts
            </h2>
            {renderBars(systemMetricRows, "bg-cyan-500")}
          </div>

          <div className="bg-slate-900/70 border border-white/5 rounded-xl p-6 h-full flex flex-col">
            <h2 className="text-base font-bold text-white mb-4 flex items-center gap-2">
              <span className="material-symbols-outlined text-violet-400">policy</span>
              System Metric: Compliance + Endpoint Health
            </h2>
            <div className="grid grid-cols-[120px,1fr] gap-4 items-center mb-4">
              <div className="relative w-[120px] h-[120px] rounded-full" style={{ background: complianceGradient || "#1e293b" }}>
                <div className="absolute inset-[18px] rounded-full bg-slate-950/95 border border-white/5" />
              </div>
              <div className="space-y-1 text-xs">
                <div className="flex justify-between"><span className="text-slate-300">Pass</span><span className="text-emerald-400 font-semibold">{compliancePass}</span></div>
                <div className="flex justify-between"><span className="text-slate-300">Fail</span><span className="text-rose-400 font-semibold">{complianceFail}</span></div>
                <div className="flex justify-between"><span className="text-slate-300">Unknown</span><span className="text-amber-300 font-semibold">{complianceUnknown}</span></div>
              </div>
            </div>
            <h3 className="text-xs font-semibold uppercase tracking-widest text-slate-400 mb-2">Endpoint Health</h3>
            <div className="grid grid-cols-2 gap-2">
              {endpointHealth.map((row) => (
                <div
                  key={row.key}
                  className={`text-xs border rounded px-2 py-1 ${row.ok ? "border-emerald-500/30 text-emerald-300 bg-emerald-500/10" : "border-rose-500/30 text-rose-300 bg-rose-500/10"}`}
                >
                  {row.label}
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      <div className="bg-slate-900/70 border border-white/5 rounded-xl p-6">
        <h2 className="text-lg font-bold text-white mb-4 flex items-center gap-2">
          <span className="material-symbols-outlined text-indigo-400">history</span> Simulation Run History
        </h2>
        {loadingHistory ? (
          <div className="flex items-center gap-3 text-slate-400 text-sm p-6">
            <div className="w-5 h-5 border-2 border-indigo-500 border-t-transparent rounded-full animate-spin" />
            Loading history…
          </div>
        ) : history.length === 0 ? (
          <p className="text-slate-500 text-sm py-6 text-center">No simulation history yet. Run a simulation on the Timeline tab first.</p>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-xs text-left">
              <thead>
                <tr className="text-slate-400 border-b border-white/5">
                  <th className="pb-2 pr-4">Run ID</th>
                  <th className="pb-2 pr-4">Task</th>
                  <th className="pb-2 pr-4">Agent Mode</th>
                  <th className="pb-2 pr-4">Status</th>
                  <th className="pb-2 pr-4">Score</th>
                  <th className="pb-2">Reward</th>
                </tr>
              </thead>
              <tbody>
                {history.map((run) => {
                  const score = getRunScore(run);
                  const reward = getRunReward(run);
                  const status = run.status || "completed";
                  const statusColor = status === "completed" ? "emerald" : status === "running" ? "amber" : "slate";
                  return (
                    <tr key={run.run_id} className="border-b border-white/5 hover:bg-white/5">
                      <td className="py-2 pr-4 font-mono text-indigo-300">{run.run_id?.slice(0, 8)}…</td>
                      <td className="py-2 pr-4 text-white font-medium">{run.task_id?.replace(/_/g, " ")}</td>
                      <td className="py-2 pr-4 text-slate-400">{run.agent_mode}</td>
                      <td className="py-2 pr-4">
                        <span className={`bg-${statusColor}-500/20 text-${statusColor}-400 text-xs font-bold px-2 py-0.5 rounded-full`}>{status}</span>
                      </td>
                      <td className="py-2 pr-4 font-bold text-white">{score != null ? `${(score * 100).toFixed(1)}%` : "—"}</td>
                      <td className="py-2 text-amber-400">{reward != null ? reward.toFixed(2) : "—"}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>

      <div className="bg-slate-900/70 border border-white/5 rounded-xl p-6">
        <h2 className="text-lg font-bold text-white mb-4 flex items-center gap-2">
          <span className="material-symbols-outlined text-amber-400">model_training</span> Trained RL Model Checkpoints
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {rlModels.length === 0 && rlModelsV2.length === 0 ? (
            <p className="text-slate-500 text-sm col-span-3">No trained models found. Train a model via the RL pipeline first.</p>
          ) : (
            [...rlModels, ...rlModelsV2.map((m) => ({
              label: m.model_path ? String(m.model_path).split(/[\\/]/).pop() : "unnamed",
              path: m.model_path ? `${m.model_path}.zip` : "",
              exists: Boolean(m.exists),
              model_type: Number(m.phase) === 2 ? "phase2" : "phase1",
            }))].map((m) => (
              <div key={`${m.path}-${m.label}`} className={`border rounded-xl p-4 ${m.exists ? "border-amber-500/30 bg-amber-500/5" : "border-white/5 bg-slate-800/40"}`}>
                <div className="flex items-center gap-2 mb-2">
                  <span className={`material-symbols-outlined text-lg ${m.exists ? "text-amber-400" : "text-slate-600"}`}>
                    {m.exists ? "check_circle" : "radio_button_unchecked"}
                  </span>
                  <span className="text-sm font-bold text-white">{m.label}</span>
                </div>
                <div className="text-xs text-slate-400 font-mono truncate">{m.path?.split("\\").pop() || m.path?.split("/").pop()}</div>
                <div className="text-xs text-slate-500 mt-1">Type: {m.model_type}</div>
                {!m.exists && <div className="text-xs text-slate-600 mt-2">Not yet trained</div>}
              </div>
            ))
          )}
        </div>
      </div>

      {Object.keys(byTask).length > 0 && (
        <div className="bg-slate-900/70 border border-white/5 rounded-xl p-6">
          <h2 className="text-lg font-bold text-white mb-4 flex items-center gap-2">
            <span className="material-symbols-outlined text-violet-400">bar_chart</span> Score by Scenario
          </h2>
          <div className="space-y-4">
            {Object.entries(byTask).map(([task, runs]) => {
              const scores = runs.map((r) => r.score ?? r.payload?.score).filter((s) => s != null);
              const avg = scores.length ? scores.reduce((a, b) => a + b, 0) / scores.length : null;
              const avgPct = avg != null ? Number((avg * 100).toFixed(1)) : 0;
              return (
                <div key={task} className="space-y-1">
                  <div className="flex justify-between text-sm">
                    <span className="font-semibold text-white">{task.replace(/_/g, " ")}</span>
                    <span className="text-slate-400">{runs.length} runs · avg {avg != null ? `${avgPct}%` : "—"}</span>
                  </div>
                  <div className="h-2 w-full rounded bg-slate-800 overflow-hidden">
                    <div className="h-full bg-violet-500" style={{ width: `${Math.max(0, Math.min(100, avgPct))}%` }} />
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}

function TrainingTab({ tasks }) {
  return <TrainingTabV2 tasks={tasks} />;
}

const TABS = [
  { id: "timeline", label: "Timeline", icon: "timeline" },
  { id: "training", label: "Training", icon: "fitness_center" },
  { id: "resources", label: "Resources", icon: "leaderboard" },
  { id: "library", label: "Overview", icon: "menu_book" },
  { id: "analytics", label: "Analytics", icon: "analytics" },
];

export function Dashboard({ tasks = [] }) {
  const [activeTab, setActiveTab] = useState("library");

  return (
    <div className="font-body-base min-h-screen flex flex-col pt-16 bg-[#0a0b14] text-white">
      <nav className="fixed top-0 left-0 w-full z-50 flex items-center justify-between px-6 h-16 bg-slate-950/80 backdrop-blur-xl border-b border-white/5 shadow-2xl shadow-indigo-950/50">
        <div className="flex items-center space-x-8">
          <span className="text-lg font-black tracking-tighter text-white uppercase">
            <span className="text-indigo-400">OPEN</span>ENV
          </span>
          <div className="hidden md:flex space-x-1">
            {TABS.map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center gap-1.5 px-4 py-2 rounded-lg text-sm font-semibold transition-all duration-200 ${
                  activeTab === tab.id
                    ? "bg-indigo-600/30 text-indigo-300 border border-indigo-500/30"
                    : "text-slate-400 hover:text-white hover:bg-white/5"
                }`}
              >
                <span className="material-symbols-outlined text-[16px]">{tab.icon}</span>
                {tab.label}
              </button>
            ))}
          </div>
        </div>
        <div className="flex items-center gap-3">
          <div className="hidden md:flex items-center gap-1.5 bg-emerald-500/10 border border-emerald-500/20 px-3 py-1.5 rounded-full">
            <div className="w-2 h-2 bg-emerald-400 rounded-full animate-pulse" />
            <span className="text-xs font-bold text-emerald-400">LIVE</span>
          </div>
          <div className="text-xs text-slate-500 hidden md:block">Gov Workflow RL | OpenEnv v2.0</div>
        </div>
      </nav>

      <main className="flex-1 max-w-7xl w-full mx-auto px-6 py-8">
        <div className="flex md:hidden mb-6 bg-slate-900 rounded-xl p-1 space-x-1">
          {TABS.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`flex-1 py-2 text-xs font-bold rounded-lg transition-all ${activeTab === tab.id ? "bg-indigo-600 text-white" : "text-slate-400"}`}
            >
              {tab.label}
            </button>
          ))}
        </div>

        <div className="mb-6">
          {activeTab === "timeline" && <div><h1 className="text-2xl font-black text-white">Oversight Dashboard</h1><p className="text-sm text-slate-400 mt-1">Watch the AI agent resolve a government workflow backlog in real time - step by step, decision by decision.</p></div>}
          {activeTab === "training" && <div><h1 className="text-2xl font-black text-white">Reinforcement Learning</h1><p className="text-sm text-slate-400 mt-1">Visualize policy convergence and reward trends as the agent continuously improves.</p></div>}
          {activeTab === "resources" && <div><h1 className="text-2xl font-black text-white">Policy Benchmark</h1><p className="text-sm text-slate-400 mt-1">Compare all three baseline policies head-to-head on identical scenarios to see which strategy wins.</p></div>}
          {activeTab === "library" && <div><h1 className="text-2xl font-black text-white">Overview</h1><p className="text-sm text-slate-400 mt-1">Explore system behavior, task configurations, OpenEnv compliance status, and workflow architecture.</p></div>}
          {activeTab === "analytics" && <div><h1 className="text-2xl font-black text-white">Performance Analytics</h1><p className="text-sm text-slate-400 mt-1">Review historical simulation runs, trained model checkpoints, and reward improvement evidence.</p></div>}
        </div>

        {activeTab === "timeline" && <TimelineTab tasks={tasks} />}
        {activeTab === "training" && <TrainingTab tasks={tasks} />}
        {activeTab === "resources" && <ResourcesTab tasks={tasks} />}
        {activeTab === "library" && <LibraryTab tasks={tasks} />}
        {activeTab === "analytics" && <AnalyticsTab />}
      </main>

      <style>{`
        @keyframes fadeUp {
          from { opacity: 0; transform: translateY(8px); }
          to { opacity: 1; transform: translateY(0); }
        }
      `}</style>
    </div>
  );
}


