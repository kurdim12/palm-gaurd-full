"use client";

import { useI18n } from "@/lib/i18n";
import { STATUS_COLOR } from "@/lib/status";
import type { Detection } from "@/lib/types";

// A compact confidence timeline: one row per detection, newest first, bar width
// = confidence, colour keyed to the predicted label.
export function DetectionTimeline({ detections }: { detections: Detection[] }) {
  const { t, locale } = useI18n();
  if (detections.length === 0) {
    return <p className="muted">{t("alerts.empty")}</p>;
  }
  return (
    <div className="timeline">
      {detections.map((d) => {
        const color = d.label === "infested" ? STATUS_COLOR.infested : STATUS_COLOR.clean;
        const when = new Date(d.captured_at).toLocaleString(
          locale === "ar" ? "ar-SA" : "en-GB",
        );
        return (
          <div className="timeline-row" key={d.id}>
            <span className="badge" style={{ background: color }}>
              {t(`status.${d.label}`)}
            </span>
            <span
              className="bar"
              style={{
                background: `linear-gradient(90deg, ${color} ${Math.round(
                  d.confidence * 100,
                )}%, var(--border) ${Math.round(d.confidence * 100)}%)`,
              }}
              title={`${t("tree.confidence")}: ${(d.confidence * 100).toFixed(0)}%`}
            />
            <span className="muted" style={{ minWidth: "3.5rem", textAlign: "center" }}>
              {(d.confidence * 100).toFixed(0)}%
            </span>
            <span className="muted">{when}</span>
          </div>
        );
      })}
    </div>
  );
}
