const DEFAULT_LOCAL_API = "http://127.0.0.1:7860";
const LOCAL_PORTS = ["7860"];
const LOCAL_HOSTS = ["127.0.0.1", "localhost"];

function candidates(path) {
  const urls = [];
  const rootOnlyPaths = path === "/rl/models";
  const compatNoApiPaths =
    path.startsWith("/simulation/") ||
    path.startsWith("/training/") ||
    path.startsWith("/rl/") ||
    path.startsWith("/openenv/") ||
    path.startsWith("/benchmark") ||
    path.startsWith("/history/");

  let isLocalDev5173 = false;
  if (typeof window !== "undefined") {
    const host = window.location.hostname;
    const isLocal = host === "localhost" || host === "127.0.0.1";
    isLocalDev5173 = isLocal && window.location.port === "5173";
  }

  // Training story endpoints are mounted at /training/* (not /api/training/*).
  // Avoid known-bad prefixes first to prevent noisy 404 logs in browser console.
  if (path.startsWith("/training/")) {
    if (isLocalDev5173) {
      for (const port of LOCAL_PORTS) {
        for (const lh of LOCAL_HOSTS) {
          urls.push(`http://${lh}:${port}${path}`);
        }
      }
    } else {
      urls.push(path);
    }
    return [...new Set(urls)];
  }

  if (isLocalDev5173) {
    // For local dev, prefer direct backend URLs first to avoid noisy Vite proxy
    // connection-refused spam when backend is temporarily down.
    for (const port of LOCAL_PORTS) {
      for (const lh of LOCAL_HOSTS) {
        if (rootOnlyPaths) {
          urls.push(`http://${lh}:${port}${path}`);
        } else {
          urls.push(`http://${lh}:${port}/api${path}`);
          urls.push(`http://${lh}:${port}/api/v1${path}`);
          if (compatNoApiPaths) {
            urls.push(`http://${lh}:${port}${path}`);
          }
        }
      }
    }
  }

  if (rootOnlyPaths) {
    urls.push(path);
  } else {
    urls.push(`/api${path}`, `/api/v1${path}`);
    if (compatNoApiPaths) {
      urls.push(path);
    }
  }

  if (isLocalDev5173 && !rootOnlyPaths) {
    for (const port of LOCAL_PORTS) {
      for (const lh of LOCAL_HOSTS) {
        // keep original ordering as fallback candidates
        urls.push(`http://${lh}:${port}/api${path}`);
        urls.push(`http://${lh}:${port}/api/v1${path}`);
      }
    }
  }

  return [...new Set(urls)];
}

export async function api(path, options = {}) {
  const method = String(options.method || "GET").toUpperCase();
  const headers = { ...(options.headers || {}) };
  if (method !== "GET" && method !== "HEAD" && !("Content-Type" in headers)) {
    headers["Content-Type"] = "application/json";
  }
  const requestOptions = {
    ...options,
    method,
    headers,
  };
  if (method === "GET" || method === "HEAD") {
    delete requestOptions.body;
  }

  const errors = [];
  for (const url of candidates(path)) {
    try {
      const res = await fetch(url, requestOptions);
      let payload = null;
      try {
        payload = await res.json();
      } catch (err) {
        payload = null;
      }
      if (!res.ok) {
        const detail = payload?.detail || `${res.status}`;
        throw new Error(`API ${path} failed on ${url}: ${detail}`);
      }
      return payload;
    } catch (err) {
      errors.push(err);
    }
  }

  const firstApiError = errors.find(
    (e) => e instanceof Error && e.message.startsWith(`API ${path} failed`)
  );
  if (firstApiError) {
    throw firstApiError;
  }
  const lastError = errors.length ? errors[errors.length - 1] : new Error("Unknown request failure.");

  throw new Error(
    `API ${path} connection failed. Start backend on ${DEFAULT_LOCAL_API}. Last error: ${
      lastError instanceof Error ? lastError.message : String(lastError)
    }`
  );
}

export function fmt(value, digits = 2) {
  if (value == null || Number.isNaN(Number(value))) return "-";
  return Number(value).toFixed(digits);
}
