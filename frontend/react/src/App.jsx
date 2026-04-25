import { useState, useEffect } from "react";
import { api } from "./api/client";
import { Dashboard } from "./components/story-ui/Dashboard";

export default function App() {
  const [tasks, setTasks] = useState([]);
  
  useEffect(() => {
    const boot = async () => {
      try {
        const taskRes = await api("/tasks");
        setTasks(taskRes.tasks || []);
      } catch (err) {
        console.error("Failed to load tasks", err);
      }
    };
    boot();
  }, []);

  return <Dashboard tasks={tasks} />;
}
