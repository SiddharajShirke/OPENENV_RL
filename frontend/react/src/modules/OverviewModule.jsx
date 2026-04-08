export function OverviewModule({ health, tasks, agents, models }) {
  const availableModels = models.filter((m) => m.exists);
  return (
    <section className="module-grid">
      <article className="panel hero-panel">
        <h2>Real-world task simulation</h2>
        <p>
          This environment simulates a government service office. Agents must reduce backlog, SLA breaches,
          and fairness gaps across services by choosing staffing and priority actions through OpenEnv
          <code> step / reset / state </code> interactions.
        </p>
      </article>

      <article className="panel">
        <h3>Environment Snapshot</h3>
        <div className="metric-grid">
          <div className="metric-card">
            <span>Server</span>
            <strong>{health?.status || "-"}</strong>
          </div>
          <div className="metric-card">
            <span>Version</span>
            <strong>{health?.version || "-"}</strong>
          </div>
          <div className="metric-card">
            <span>Tasks</span>
            <strong>{tasks.length}</strong>
          </div>
          <div className="metric-card">
            <span>Baseline Agents</span>
            <strong>{agents.length}</strong>
          </div>
          <div className="metric-card">
            <span>RL Models</span>
            <strong>{availableModels.length}</strong>
          </div>
          <div className="metric-card">
            <span>Active Sessions</span>
            <strong>{health?.active_sessions ?? "-"}</strong>
          </div>
        </div>
      </article>

      <article className="panel">
        <h3>Workflow</h3>
        <ol className="flow-list">
          <li>Run simulation with baseline, inference-like LLM, or trained RL agent.</li>
          <li>Train a new Phase 2 PPO run in background with a fresh seed.</li>
          <li>Track training progress, rewards, grader score, and penalties live.</li>
          <li>Compare new run against baseline and existing Phase 2 checkpoint.</li>
        </ol>
      </article>

      <article className="panel">
        <h3>Benchmark Tasks</h3>
        <div className="tag-wrap">
          {tasks.map((task) => (
            <span key={task} className="tag">
              {task}
            </span>
          ))}
        </div>
      </article>
    </section>
  );
}

