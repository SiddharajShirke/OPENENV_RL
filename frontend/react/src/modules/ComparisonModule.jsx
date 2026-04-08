import { useEffect, useMemo, useState } from "react";
import { api, fmt } from "../api/client";
import { CompareBars } from "../components/Charts";

function normalizeNumber(value, fallback = 0) {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

function synthesizeRunsFromSummary({
  runCount,
  seedBase,
  score,
  includeLlm = false,
}) {
  const count = Math.max(1, Number(runCount || 1));
  const baseSeed = Number(seedBase || 0);
  const rows = [];
  for (let i = 0; i < count; i += 1) {
    rows.push({
      run_index: i + 1,
      seed: baseSeed + i,
      score,
      reward_sum: null,
      completed: null,
      backlog: null,
      llm_steps: includeLlm ? null : undefined,
      heuristic_fallback_steps: includeLlm ? null : undefined,
      invalid_action_rate: includeLlm ? null : undefined,
      repaired_action_rate: includeLlm ? null : undefined,
      auto_switch_count: includeLlm ? null : undefined,
      _legacySynthetic: true,
    });
  }
  return rows;
}

function normalizeResultShape(result, historyMeta = {}) {
  if (!result || typeof result !== "object") {
    return {
      baselineScore: 0,
      trainedScore: 0,
      llmScore: null,
      llmError: null,
      deltaTrainedVsBaseline: 0,
      deltaLlmVsBaseline: null,
      baselineRuns: [],
      llmRuns: [],
    };
  }

  const baselineScore = normalizeNumber(result.baselineScore, 0);
  const trainedScore = normalizeNumber(result.trainedScore, 0);
  const llmScore =
    result.llmScore == null || Number.isNaN(Number(result.llmScore))
      ? null
      : Number(result.llmScore);
  const runCount = Number(historyMeta?.runs || 1);
  const seedBase = Number(historyMeta?.seed_base || 0);

  const baselineRuns = Array.isArray(result.baselineRuns)
    ? result.baselineRuns
    : synthesizeRunsFromSummary({
        runCount,
        seedBase,
        score: baselineScore,
      });
  const llmRuns = Array.isArray(result.llmRuns)
    ? result.llmRuns
    : llmScore == null
      ? []
      : synthesizeRunsFromSummary({
          runCount,
          seedBase,
          score: llmScore,
          includeLlm: true,
        });

  return {
    baselineScore,
    trainedScore,
    llmScore,
    llmError: result.llmError || null,
    deltaTrainedVsBaseline:
      result.deltaTrainedVsBaseline == null
        ? trainedScore - baselineScore
        : normalizeNumber(result.deltaTrainedVsBaseline, trainedScore - baselineScore),
    deltaLlmVsBaseline:
      result.deltaLlmVsBaseline == null
        ? llmScore == null
          ? null
          : llmScore - baselineScore
        : normalizeNumber(result.deltaLlmVsBaseline, llmScore == null ? 0 : llmScore - baselineScore),
    baselineRuns,
    llmRuns,
    legacySnapshotWithoutRunRows:
      !Array.isArray(result.baselineRuns) || (llmScore != null && !Array.isArray(result.llmRuns)),
  };
}

export function ComparisonModule({ tasks, agents, modelOptions, onStatus, defaultTask }) {
  const [taskId, setTaskId] = useState(defaultTask || tasks[0] || "district_backlog_easy");
  const [policyName, setPolicyName] = useState("backlog_clearance");
  const [modelPath, setModelPath] = useState(modelOptions[0]?.path || "");
  const [modelType, setModelType] = useState(modelOptions[0]?.model_type || "maskable");
  const [runs, setRuns] = useState(5);
  const [steps, setSteps] = useState(80);
  const [episodes, setEpisodes] = useState(5);
  const [seedBase, setSeedBase] = useState(100);
  const [includeLlm, setIncludeLlm] = useState(true);
  const [autoSelectBestModel, setAutoSelectBestModel] = useState(true);
  const [busy, setBusy] = useState(false);
  const [progressNote, setProgressNote] = useState("");
  const [result, setResult] = useState(null);
  const [historyRows, setHistoryRows] = useState([]);

  const optionMap = useMemo(() => {
    const map = new Map();
    modelOptions.forEach((m) => map.set(m.path, m));
    return map;
  }, [modelOptions]);

  const refreshHistory = async () => {
    try {
      const res = await api("/history/comparisons");
      setHistoryRows(res.comparisons || []);
    } catch (_err) {
      // Persistence may be disabled in some local/dev deployments.
      setHistoryRows([]);
    }
  };

  const clearComparisonHistory = async () => {
    try {
      const res = await api("/history/comparisons", { method: "DELETE" });
      setHistoryRows([]);
      onStatus(`Comparison history cleared (${res.deleted_rows || 0}).`);
    } catch (err) {
      onStatus(err.message);
    }
  };

  useEffect(() => {
    refreshHistory();
  }, []);

  useEffect(() => {
    if (taskId === "cross_department_hard" && Number(steps) < 70) setSteps(70);
    if (taskId === "mixed_urgency_medium" && Number(steps) < 60) setSteps(60);
  }, [taskId, steps]);

  const runLlmBatch = async ({ task, runCount, maxSteps, seedStart }) => {
    const rows = [];
    for (let i = 0; i < runCount; i += 1) {
      const seed = Number(seedStart) + i;
      onStatus(`Running LLM simulation ${i + 1}/${runCount}...`);
      setProgressNote(`LLM simulation run ${i + 1}/${runCount} in progress...`);
      const sim = await api("/simulation/run", {
        method: "POST",
        body: JSON.stringify({
          task_id: task,
          agent_mode: "llm_inference",
          max_steps: Number(maxSteps),
          seed,
        }),
      });
      rows.push({
        run_index: i + 1,
        seed: sim.seed,
        score: Number(sim.score ?? 0),
        reward_sum: Number(sim.total_reward ?? 0),
        completed: Number(sim.summary?.total_completed ?? 0),
        backlog: Number(sim.summary?.total_backlog ?? 0),
        llm_steps: Number(sim.summary?.llm_steps ?? 0),
        heuristic_fallback_steps: Number(sim.summary?.heuristic_fallback_steps ?? 0),
        invalid_action_rate: Number(sim.summary?.invalid_action_rate ?? 0),
        repaired_action_rate: Number(sim.summary?.repaired_action_rate ?? 0),
        auto_switch_count: Number(sim.summary?.auto_switch_count ?? 0),
      });
    }
    return rows;
  };

  const selectBestModelConfig = async (evalEpisodes) => {
    const candidates = modelOptions.filter((m) => m?.exists && m?.path);
    if (!candidates.length) return null;
    let best = null;
    for (let i = 0; i < candidates.length; i += 1) {
      const c = candidates[i];
      setProgressNote(`Evaluating trained model ${i + 1}/${candidates.length}: ${c.label}`);
      try {
        const res = await api("/rl/evaluate", {
          method: "POST",
          body: JSON.stringify({
            model_path: c.path,
            model_type: c.model_type || "auto",
            episodes: evalEpisodes,
            task_ids: [taskId],
          }),
        });
        const score = Number(res.results?.[0]?.grader_score ?? 0);
        if (best == null || score > best.score) {
          best = { path: c.path, model_type: c.model_type || "auto", score, label: c.label };
        }
      } catch (_err) {
        // Skip failing candidate and continue.
      }
    }
    return best;
  };

  const runCompare = async () => {
    setBusy(true);
    const seedRuns = Math.max(5, Math.min(10, Number(runs)));
    const evalEpisodes = Math.max(5, Math.min(10, Number(episodes)));
    const effectiveSteps =
      taskId === "cross_department_hard"
        ? Math.max(70, Number(steps))
        : taskId === "mixed_urgency_medium"
          ? Math.max(60, Number(steps))
          : Number(steps);
    setProgressNote("Preparing multi-seed comparison...");
    try {
      let chosenModelPath = modelPath;
      let chosenModelType = modelType;
      if (autoSelectBestModel) {
        const best = await selectBestModelConfig(evalEpisodes);
        if (best?.path) {
          chosenModelPath = best.path;
          chosenModelType = best.model_type === "auto" ? "maskable" : best.model_type;
          setModelPath(chosenModelPath);
          setModelType(chosenModelType);
          onStatus(`Auto-selected best trained config: ${best.label} (avg score ${fmt(best.score, 3)}).`);
        }
      }

      setProgressNote("Running baseline policy benchmark...");
      const [baseline, trained] = await Promise.all([
        api("/benchmark", {
          method: "POST",
          body: JSON.stringify({
            task_id: taskId,
            runs: seedRuns,
            max_steps: effectiveSteps,
            agent_policies: [policyName],
            seed_base: Number(seedBase),
          }),
        }),
        (async () => {
          const body = {
            model_path: chosenModelPath,
            model_type: chosenModelType,
            episodes: evalEpisodes,
            task_ids: [taskId],
          };
          try {
            return await api("/rl/evaluate", { method: "POST", body: JSON.stringify(body) });
          } catch (err) {
            const msg = String(err?.message || "");
            if (msg.includes("422")) {
              return await api("/rl/evaluate", {
                method: "POST",
                body: JSON.stringify({ ...body, model_type: "auto" }),
              });
            }
            throw err;
          }
        })(),
      ]);

      const baselineScore = Number(baseline.agent_results?.[0]?.average_score ?? 0);
      const trainedScore = Number(trained.results?.[0]?.grader_score ?? 0);
      setProgressNote("Baseline and trained-model evaluation complete. Processing optional LLM runs...");

      let llmRuns = [];
      let llmScore = null;
      let llmError = null;
      if (includeLlm) {
        try {
          llmRuns = await runLlmBatch({
            task: taskId,
            runCount: seedRuns,
            maxSteps: effectiveSteps,
            seedStart: Number(seedBase),
          });
          if (llmRuns.length) {
            llmScore = llmRuns.reduce((acc, row) => acc + Number(row.score || 0), 0) / llmRuns.length;
          }
        } catch (err) {
          llmError = err.message;
        }
      }

      setProgressNote("Finalizing comparison and saving history...");
      setResult(normalizeResultShape({
        baselineScore,
        trainedScore,
        llmScore,
        llmError,
        deltaTrainedVsBaseline: trainedScore - baselineScore,
        deltaLlmVsBaseline: llmScore == null ? null : llmScore - baselineScore,
        baselineRuns: baseline.agent_results?.[0]?.runs || [],
        llmRuns,
      }, {
        runs: seedRuns,
        seed_base: Number(seedBase),
      }));
      try {
        await api("/history/comparisons", {
          method: "POST",
          body: JSON.stringify({
            task_id: taskId,
            baseline_policy: policyName,
            model_path: chosenModelPath,
            model_type: chosenModelType,
            include_llm: includeLlm,
            runs: seedRuns,
            steps: effectiveSteps,
            episodes: evalEpisodes,
            seed_base: Number(seedBase),
            result: {
              baselineScore,
              trainedScore,
              llmScore,
              llmError,
              deltaTrainedVsBaseline: trainedScore - baselineScore,
              deltaLlmVsBaseline: llmScore == null ? null : llmScore - baselineScore,
              baselineRuns: baseline.agent_results?.[0]?.runs || [],
              llmRuns,
            },
          }),
        });
        await refreshHistory();
      } catch (_err) {
        // non-blocking for deployments without persistence
      }
      const llmPart = llmScore == null ? "LLM=n/a" : `LLM=${fmt(llmScore, 3)}`;
      onStatus(
        `Comparison completed (multi-seed=${seedRuns}). baseline=${fmt(baselineScore, 3)} | trained=${fmt(trainedScore, 3)} | ${llmPart}`
      );
    } catch (err) {
      onStatus(err.message);
    } finally {
      setProgressNote("");
      setBusy(false);
    }
  };

  const onModelChange = (val) => {
    setModelPath(val);
    const hit = optionMap.get(val);
    if (hit) setModelType(hit.model_type || "maskable");
  };

  return (
    <section className="module-grid">
      <article className="panel">
        <h2>Model Comparison</h2>
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
            Baseline Policy
            <select value={policyName} onChange={(e) => setPolicyName(e.target.value)}>
              {agents.map((agent) => (
                <option key={agent} value={agent}>
                  {agent}
                </option>
              ))}
            </select>
          </label>
          <label>
            Trained Model
            <select value={modelPath} onChange={(e) => onModelChange(e.target.value)}>
              {modelOptions.map((m) => (
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
          <label>
            Baseline Runs
            <input type="number" min={5} max={10} value={runs} onChange={(e) => setRuns(e.target.value)} />
          </label>
          <label>
            Seed Base
            <input type="number" value={seedBase} onChange={(e) => setSeedBase(e.target.value)} />
          </label>
          <label>
            Max Steps
            <input type="number" min={1} max={500} value={steps} onChange={(e) => setSteps(e.target.value)} />
          </label>
          <label>
            Eval Episodes
            <input type="number" min={5} max={10} value={episodes} onChange={(e) => setEpisodes(e.target.value)} />
          </label>
          <label>
            LLM Simulation
            <select value={includeLlm ? "on" : "off"} onChange={(e) => setIncludeLlm(e.target.value === "on")}>
              <option value="on">Include in comparison</option>
              <option value="off">Skip</option>
            </select>
          </label>
          <label>
            Trained Model Selection
            <select
              value={autoSelectBestModel ? "auto" : "manual"}
              onChange={(e) => setAutoSelectBestModel(e.target.value === "auto")}
            >
              <option value="auto">Auto-select best (avg score)</option>
              <option value="manual">Use selected model</option>
            </select>
          </label>
        </div>
        <div className="row">
          <button onClick={runCompare} disabled={busy || !modelPath}>
            {busy ? "Comparing..." : "Run Comparison"}
          </button>
          <button className="ghost" onClick={clearComparisonHistory}>
            Clear Saved History
          </button>
        </div>
        {busy ? (
          <div className="loading-inline">
            <span className="spinner-dot" />
            <span>{progressNote || "Running comparison..."}</span>
          </div>
        ) : null}
      </article>

      {result ? (
        <>
          <article className="panel">
            <h3>Score Comparison</h3>
            <CompareBars
              rows={[
                { label: `Baseline (${policyName})`, value: result.baselineScore },
                ...(result.llmScore == null ? [] : [{ label: "LLM Simulation", value: result.llmScore }]),
                { label: `Trained (${modelType})`, value: result.trainedScore },
              ]}
            />
            <p className="muted">
              Delta (trained - baseline): <strong>{fmt(result.deltaTrainedVsBaseline, 3)}</strong>
            </p>
            {result.deltaLlmVsBaseline != null ? (
              <p className="muted">
                Delta (llm - baseline): <strong>{fmt(result.deltaLlmVsBaseline, 3)}</strong>
              </p>
            ) : null}
            {result.legacySnapshotWithoutRunRows ? (
              <p className="muted">Legacy snapshot detected: per-run rows were not stored; table rows below are reconstructed placeholders from saved summary.</p>
            ) : null}
            {result.llmError ? <p className="muted">LLM simulation error: {result.llmError}</p> : null}
          </article>

          <article className="panel">
            <h3>Baseline Run Variability</h3>
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Run</th>
                    <th>Seed</th>
                    <th>Score</th>
                    <th>Reward</th>
                    <th>Completed</th>
                    <th>Backlog</th>
                  </tr>
                </thead>
                <tbody>
                  {(result.baselineRuns || []).map((r) => (
                    <tr key={r.run_index}>
                      <td>{r.run_index}</td>
                      <td>{r.seed}</td>
                      <td>{fmt(r.score, 3)}</td>
                      <td>{fmt(r.reward_sum, 2)}</td>
                      <td>{r.completed}</td>
                      <td>{r.backlog}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </article>

          {result.llmRuns?.length ? (
            <article className="panel">
              <h3>LLM Simulation Variability</h3>
              <div className="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>Run</th>
                      <th>Seed</th>
                      <th>Score</th>
                      <th>Reward</th>
                      <th>Completed</th>
                      <th>Backlog</th>
                      <th>LLM Steps</th>
                      <th>Fallback Steps</th>
                      <th>Invalid Rate</th>
                      <th>Repaired Rate</th>
                      <th>Auto Switches</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(result.llmRuns || []).map((r) => (
                      <tr key={`llm-${r.run_index}`}>
                        <td>{r.run_index}</td>
                        <td>{r.seed}</td>
                        <td>{fmt(r.score, 3)}</td>
                        <td>{fmt(r.reward_sum, 2)}</td>
                        <td>{r.completed}</td>
                        <td>{r.backlog}</td>
                        <td>{r.llm_steps}</td>
                        <td>{r.heuristic_fallback_steps}</td>
                        <td>{fmt((r.invalid_action_rate || 0) * 100, 1)}%</td>
                        <td>{fmt((r.repaired_action_rate || 0) * 100, 1)}%</td>
                        <td>{r.auto_switch_count || 0}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </article>
          ) : null}
        </>
      ) : null}

      {historyRows.length ? (
        <article className="panel">
          <h3>Saved Comparison History</h3>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Task</th>
                  <th>Baseline Policy</th>
                  <th>Trained Model Type</th>
                  <th>Baseline</th>
                  <th>LLM</th>
                  <th>Trained</th>
                  <th>Load</th>
                </tr>
              </thead>
              <tbody>
                {historyRows.map((row) => (
                  <tr key={row.comparison_id}>
                    <td>{row.task_id}</td>
                    <td>{row.baseline_policy}</td>
                    <td>{row.model_type}</td>
                    <td>{fmt(row.result?.baselineScore, 3)}</td>
                    <td>{row.result?.llmScore == null ? "-" : fmt(row.result?.llmScore, 3)}</td>
                    <td>{fmt(row.result?.trainedScore, 3)}</td>
                    <td>
                      <button
                        className="ghost"
                        onClick={() => {
                          setTaskId(row.task_id || taskId);
                          setPolicyName(row.baseline_policy || policyName);
                          setModelPath(row.model_path || modelPath);
                          setModelType(row.model_type || modelType);
                          setRuns(Number(row.runs || runs));
                          setSteps(Number(row.steps || steps));
                          setEpisodes(Number(row.episodes || episodes));
                          setSeedBase(Number(row.seed_base || seedBase));
                          setIncludeLlm(Boolean(row.include_llm));
                          const id = String(row.comparison_id || "");
                          const loadFromRow = () => {
                            setResult(normalizeResultShape(row.result, row));
                            onStatus(`Loaded saved comparison ${id.slice(0, 8)}.`);
                          };
                          if (id) {
                            api(`/history/comparisons/${id}`)
                              .then(async (detail) => {
                                const normalized = normalizeResultShape(detail?.result, detail || row);
                                if (normalized.legacySnapshotWithoutRunRows) {
                                  try {
                                    onStatus(`Repairing legacy comparison ${id.slice(0, 8)}...`);
                                    await api(`/history/comparisons/${id}/repair`, { method: "POST" });
                                    const repaired = await api(`/history/comparisons/${id}`);
                                    setResult(normalizeResultShape(repaired?.result, repaired || row));
                                    onStatus(`Loaded repaired comparison ${id.slice(0, 8)}.`);
                                    await refreshHistory();
                                    return;
                                  } catch (_err) {
                                    setResult(normalized);
                                    onStatus(`Loaded legacy comparison ${id.slice(0, 8)} (repair unavailable).`);
                                    return;
                                  }
                                }
                                setResult(normalized);
                                onStatus(`Loaded saved comparison ${id.slice(0, 8)}.`);
                              })
                              .catch(() => {
                                loadFromRow();
                              });
                          } else {
                            loadFromRow();
                          }
                        }}
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
