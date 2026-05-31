// Domain types mirrored from the API (packages/api/app/models.py).

export type TreeStatus = "clean" | "suspect" | "infested" | "treated";
export type Label = "clean" | "infested";

export interface Tree {
  id: string;
  name_ar: string;
  name_en: string;
  lat: number;
  lon: number;
  status: TreeStatus;
  infested_streak: number;
  last_detection_at: string | null;
  updated_at: string;
}

export interface Detection {
  id: string;
  device_id: string;
  tree_id: string;
  label: Label;
  confidence: number;
  captured_at: string;
  received_at: string;
  audio_url: string | null;
  model_version: string | null;
}

export interface Alert {
  id: string;
  tree_id: string;
  created_at: string;
  acknowledged: boolean;
  acknowledged_at: string | null;
  message_ar: string;
  message_en: string;
}

export interface StatusCounts {
  clean: number;
  suspect: number;
  infested: number;
  treated: number;
}
