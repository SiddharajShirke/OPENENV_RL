const NAV_ITEMS = [
  { id: "overview", title: "Overview" },
  { id: "simulation", title: "Simulation Lab" },
  { id: "training", title: "Training Studio" },
  { id: "comparison", title: "Model Comparison" },
];

export function Layout({ active, onChange, status, children }) {
  return (
    <div className="app-shell">
      <aside className="sidebar">
        <h1>OpenEnv RL Console</h1>
        <p className="sidebar-sub">Real-world government workflow simulation and RL training.</p>
        <nav>
          {NAV_ITEMS.map((item) => (
            <button
              key={item.id}
              className={`nav-btn ${active === item.id ? "active" : ""}`}
              onClick={() => onChange(item.id)}
            >
              {item.title}
            </button>
          ))}
        </nav>
      </aside>
      <main className="content">
        <div className="status-banner">{status}</div>
        {children}
      </main>
    </div>
  );
}

