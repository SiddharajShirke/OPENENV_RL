import { useEffect, useMemo, useRef, useState } from "react";
import { api, fmt } from "../api/client";
import { CompareBars } from "../components/Charts";

export function TrainingModule({ onStatus, onModelReady }) {
  const [phase, setPhase] = useState(2);
  const [timesteps, setTimesteps] = useState(120000);
  const [nEnvs, setNEnvs] = useState(4);
  const [seed, setSeed] = useState("");
  const [configPath, setConfigPath] = useState("rl/configs/curriculum.yaml");
  const [jobs, setJobs] = useState([]);
  const [selectedJobId, setSelectedJobId] = useState("");
  const [busy, setBusy] = useState(false);
  const emittedModelPathsRef = useRef(new Set());

  const refreshJobs = async (silent = false) => {
    try {
      const res = await api("/training/jobs/list", { method: "GET" });
      const list = res.jobs || [];
      setJobs(list);
      if (!selectedJobId && list.length) {
        setSelectedJobId(list[0].job_id);
      }
      if (!silent) onStatus(`Training jobs refreshed (${list.length}).`);
    } catch (err) {
      onStatus(err.message);
    }
  };

  useEffect(() => {
    refreshJobs(true);
    const t = setInterval(() => refreshJobs(true), 2000);
    return () => clearInterval(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const selected = useMemo(() => jobs.find((j) => j.job_id === selectedJobId) || null, [jobs, selectedJobId]);

  const startTraining = async () => {
    setBusy(true);
    try {
      const payload = {
        phase: Number(phase),
        timesteps: Number(timesteps),
        n_envs: Number(nEnvs),
        seed: seed.trim() ? Number(seed) : null,
        config_path: configPath.trim() || null,
      };
      const res = await api("/training/jobs", {
        method: "POST",
        body: JSON.stringify(payload),
      });
      setSelectedJobId(res.job_id);
      onStatus(`Training started (job=${res.job_id}, seed=${res.seed}).`);
      await refreshJobs(true);
    } catch (err) {
      onStatus(err.message);
    } finally {
      setBusy(false);
    }
  };

  const stopTraining = async () => {
    if (!selected) return;
    try {
      await api(`/training/jobs/${selected.job_id}/stop`, { method: "POST" });
      onStatus(`Training stop requested for ${selected.job_id}.`);
      await refreshJobs(true);
    } catch (err) {
      onStatus(err.message);
    }
  };

  useEffect(() => {
    const outputPath = selected?.output_model_path;
    if (!outputPath) return;
    if (emittedModelPathsRef.current.has(outputPath)) return;
    emittedModelPathsRef.current.add(outputPath);
    onModelReady(outputPath, "maskable");
  }, [selected?.output_model_path, onModelReady]);

  return (
    <section className="module-grid">
      <article className="panel">
        <h2>Training Studio</h2>
        <div className="control-grid">
          <label>
            Phase
            <select value={phase} onChange={(e) => setPhase(Number(e.target.value))}>
              <option value={1}>Phase 1 (easy)</option>
              <option value={2}>Phase 2 (curriculum)</option>
            </select>
          </label>
          <label>
            Timesteps
            <input type="number" min={10000} max={2000000} value={timesteps} onChange={(e) => setTimesteps(e.target.value)} />
          </label>
          <label>
            Parallel Envs
            <input type="number" min={1} max={16} value={nEnvs} onChange={(e) => setNEnvs(e.target.value)} />
          </label>
          <label>
            Seed (optional)
            <input value={seed} onChange={(e) => setSeed(e.target.value)} placeholder="auto-random per run" />
          </label>
          <label>
            Config Path
            <input value={configPath} onChange={(e) => setConfigPath(e.target.value)} />
          </label>
        </div>
        <div className="row">
          <button onClick={startTraining} disabled={busy}>
            {busy ? "Starting..." : "Start New Training Run"}
          </button>
          <button className="ghost" onClick={stopTraining} disabled={!selected || selected.status !== "running"}>
            Stop Selected Run
          </button>
          <button className="ghost" onClick={() => refreshJobs(false)}>
            Refresh
          </button>
        </div>
      </article>

      <article className="panel">
        <h3>Training Jobs</h3>
        <div className="jobs-list">
          {jobs.map((job) => (
            <button
              key={job.job_id}
              className={`job-item ${selectedJobId === job.job_id ? "active" : ""}`}
              onClick={() => setSelectedJobId(job.job_id)}
            >
              <div>
                <strong>{job.job_id.slice(0, 8)}</strong>
                <div className="muted">phase {job.phase} | seed {job.seed}</div>
              </div>
              <span className={`job-status ${job.status}`}>{job.status}</span>
            </button>
          ))}
          {!jobs.length ? <p className="muted">No training jobs yet.</p> : null}
        </div>
      </article>

      {selected ? (
        <>
          <article className="panel">
            <h3>Selected Job Dashboard</h3>
            <div className="metric-grid">
              <div className="metric-card">
                <span>Status</span>
                <strong>{selected.status}</strong>
              </div>
              <div className="metric-card">
                <span>Progress</span>
                <strong>{fmt((selected.progress || 0) * 100, 1)}%</strong>
              </div>
              <div className="metric-card">
                <span>Grader Score</span>
                <strong>{fmt(selected.latest_metrics?.grader_score, 3)}</strong>
              </div>
              <div className="metric-card">
                <span>Mean Reward</span>
                <strong>{fmt(selected.latest_metrics?.mean_reward ?? selected.latest_metrics?.ep_rew_mean, 2)}</strong>
              </div>
              <div className="metric-card">
                <span>SLA Penalty</span>
                <strong>{fmt(selected.latest_metrics?.episode_mean_sla_penalty, 3)}</strong>
              </div>
              <div className="metric-card">
                <span>Fairness Penalty</span>
                <strong>{fmt(selected.latest_metrics?.episode_mean_fairness_penalty, 3)}</strong>
              </div>
            </div>
            <div className="progress-track">
              <div className="progress-fill" style={{ width: `${Math.max(2, (selected.progress || 0) * 100)}%` }} />
            </div>
            <p className="muted">
              Output Model: {selected.output_model_path || "pending"}
            </p>
          </article>

          <article className="panel">
            <h3>Evaluation After Training</h3>
            {selected.evaluation_rows?.length ? (
              <>
                <CompareBars rows={selected.evaluation_rows.map((r) => ({ label: r.task_id, value: r.grader_score }))} />
                <div className="table-wrap">
                  <table>
                    <thead>
                      <tr>
                        <th>Task</th>
                        <th>Score</th>
                        <th>Reward</th>
                        <th>Completed</th>
                        <th>SLA Breaches</th>
                      </tr>
                    </thead>
                    <tbody>
                      {selected.evaluation_rows.map((r) => (
                        <tr key={r.task_id}>
                          <td>{r.task_id}</td>
                          <td>{fmt(r.grader_score, 3)}</td>
                          <td>{fmt(r.total_reward, 2)}</td>
                          <td>{r.total_completed}</td>
                          <td>{r.total_sla_breaches}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <p className="muted">Average score: {fmt(selected.evaluation_avg_score, 3)}</p>
              </>
            ) : (
              <p className="muted">Evaluation will appear automatically after training completes.</p>
            )}
          </article>
        </>
      ) : null}
    </section>
  );
}
