// Shared status → colour mapping for map pins and badges. Single source so the
// map, badges, and legend never drift.

import type { TreeStatus } from "./types";

export const STATUS_COLOR: Record<TreeStatus, string> = {
  clean: "#2e7d32", // green
  suspect: "#f9a825", // amber
  infested: "#c62828", // red
  treated: "#1565c0", // blue
};

export const STATUS_ORDER: TreeStatus[] = ["clean", "suspect", "infested", "treated"];
