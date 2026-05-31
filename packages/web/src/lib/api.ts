// Typed client for the Palm Guard API. No browser-storage hacks (CLAUDE.md):
// all state is real API state.

import type { Alert, Detection, StatusCounts, Tree } from "./types";

const BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
    cache: "no-store",
  });
  if (!res.ok) {
    throw new Error(`API ${res.status}: ${res.statusText}`);
  }
  return (await res.json()) as T;
}

export const api = {
  listTrees: () => request<Tree[]>("/api/v1/trees"),
  getTree: (id: string) => request<Tree>(`/api/v1/trees/${id}`),
  statusCounts: () => request<StatusCounts>("/api/v1/status/counts"),
  listDetections: (treeId: string, limit = 50) =>
    request<Detection[]>(`/api/v1/trees/${treeId}/detections?limit=${limit}`),
  markTreated: (id: string) =>
    request<Tree>(`/api/v1/trees/${id}/treat`, { method: "POST" }),
  listAlerts: (onlyOpen = false) =>
    request<Alert[]>(`/api/v1/alerts?only_open=${onlyOpen}`),
  acknowledgeAlert: (id: string) =>
    request<Alert>(`/api/v1/alerts/${id}/acknowledge`, { method: "POST" }),
};
