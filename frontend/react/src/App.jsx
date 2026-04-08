import { useCallback, useEffect, useMemo, useState } from "react";
import { api } from "./api/client";
import { Layout } from "./components/Layout";
import { OverviewModule } from "./modules/OverviewModule";
import { SimulationModule } from "./modules/SimulationModule";
import { TrainingModule } from "./modules/TrainingModule";
import { ComparisonModule } from "./modules/ComparisonModule";

function uniqModels(models) {
  const map = new Map();
  models.forEach((m) => {
    if (m?.path) map.set(m.path, m);
  });
  return [...map.values()];
}

export default function App() {
  const [activeTab, setActiveTab] = useState("overview");
  const [status, setStatus] = useState("Initializing...");

  const [health, setHealth] = useState(null);
  const [tasks, setTasks] = useState([]);
  const [agents, setAgents] = useState([]);
  const [models, setModels] = useState([]);
  const [customModels, setCustomModels] = useState([]);

  const [preferredModelPath, setPreferredModelPath] = useState("");
  const [preferredModelType, setPreferredModelType] = useState("maskable");

  const allModels = useMemo(() => uniqModels([...models, ...customModels]), [models, customModels]);
  const defaultTask = tasks.includes("district_backlog_easy") ? "district_backlog_easy" : tasks[0];

  useEffect(() => {
    const boot = async () => {
      try {
        const [healthRes, taskRes, agentRes, modelRes] = await Promise.all([
          api("/health"),
          api("/tasks"),
          api("/agents"),
          api("/rl/models"),
        ]);
        setHealth(healthRes);
        setTasks(taskRes.tasks || []);
        setAgents(agentRes || []);
        const existingModels = (modelRes.models || []).filter((m) => m.exists);
        setModels(existingModels);

        const phase2 = existingModels.find((m) => String(m.path).toLowerCase().includes("phase2_final"));
        if (phase2) {
          setPreferredModelPath(phase2.path);
          setPreferredModelType(phase2.model_type || "maskable");
        } else if (existingModels.length) {
          setPreferredModelPath(existingModels[0].path);
          setPreferredModelType(existingModels[0].model_type || "maskable");
        }
        setStatus(`Backend ready (v${healthRes.version}).`);
      } catch (err) {
        setStatus(err.message);
      }
    };
    boot();
  }, []);

  const handleModelReady = useCallback((path, type, outputModelName = null) => {
    if (!path) return;
    const modelType = type || "maskable";
    const fileName = path.split("\\").pop() || path.split("/").pop();
    const shownName = outputModelName || fileName;
    const label = `New Training Run (${shownName})`;

    setCustomModels((prev) => {
      const exists = prev.some((m) => m.path === path);
      if (exists) return prev;
      return uniqModels([...prev, { label, path, exists: true, model_type: modelType }]);
    });

    setPreferredModelPath((prev) => (prev === path ? prev : path));
    setPreferredModelType((prev) => (prev === modelType ? prev : modelType));
    setStatus((prev) => {
      const next = `New model ready: ${path}`;
      return prev === next ? prev : next;
    });
  }, []);

  let content = null;
  if (activeTab === "overview") {
    content = <OverviewModule health={health} tasks={tasks} agents={agents} models={allModels} />;
  } else if (activeTab === "simulation") {
    content = (
      <SimulationModule
        tasks={tasks}
        agents={agents}
        models={allModels}
        defaultTask={defaultTask}
        preferredModelPath={preferredModelPath}
        preferredModelType={preferredModelType}
        onStatus={setStatus}
      />
    );
  } else if (activeTab === "training") {
    content = <TrainingModule onStatus={setStatus} onModelReady={handleModelReady} />;
  } else if (activeTab === "comparison") {
    content = (
      <ComparisonModule
        tasks={tasks}
        agents={agents}
        modelOptions={allModels}
        defaultTask={defaultTask}
        onStatus={setStatus}
      />
    );
  }

  return (
    <Layout active={activeTab} onChange={setActiveTab} status={status}>
      {content}
    </Layout>
  );
}
