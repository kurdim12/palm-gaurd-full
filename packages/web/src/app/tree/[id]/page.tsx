"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { DetectionTimeline } from "@/components/DetectionTimeline";
import { StatusBadge } from "@/components/StatusBadge";
import { api } from "@/lib/api";
import { localizedTreeName, useI18n } from "@/lib/i18n";
import type { Detection, Tree } from "@/lib/types";

export default function TreeDetailPage({ params }: { params: { id: string } }) {
  const { t, locale } = useI18n();
  const [tree, setTree] = useState<Tree | null>(null);
  const [detections, setDetections] = useState<Detection[]>([]);
  const [error, setError] = useState(false);
  const [treated, setTreated] = useState(false);
  const [busy, setBusy] = useState(false);

  const load = () => {
    Promise.all([api.getTree(params.id), api.listDetections(params.id)])
      .then(([tr, ds]) => {
        setTree(tr);
        setDetections(ds);
      })
      .catch(() => setError(true));
  };

  useEffect(load, [params.id]);

  const onTreat = async () => {
    setBusy(true);
    try {
      const updated = await api.markTreated(params.id);
      setTree(updated);
      setTreated(true);
    } catch {
      setError(true);
    } finally {
      setBusy(false);
    }
  };

  if (error) return <p className="muted">{t("common.error")}</p>;
  if (!tree) return <p className="muted">{t("common.loading")}</p>;

  const latestAudio = detections.find((d) => d.audio_url)?.audio_url ?? null;

  return (
    <>
      <Link href="/" className="muted">
        ← {t("tree.back")}
      </Link>
      <div style={{ display: "flex", alignItems: "center", gap: "1rem", marginTop: "0.5rem" }}>
        <h2 style={{ margin: 0 }}>{localizedTreeName(locale, tree)}</h2>
        <StatusBadge status={tree.status} />
      </div>

      <div className="cards">
        <div className="card">
          <div className="muted">{t("tree.last_detection")}</div>
          <div>
            {tree.last_detection_at
              ? new Date(tree.last_detection_at).toLocaleString(
                  locale === "ar" ? "ar-SA" : "en-GB",
                )
              : "—"}
          </div>
        </div>
        <div className="card">
          <div className="muted">{t("tree.infested_streak")}</div>
          <div className="count">{tree.infested_streak}</div>
        </div>
      </div>

      <h3>{t("tree.latest_audio")}</h3>
      {latestAudio ? (
        // eslint-disable-next-line jsx-a11y/media-has-caption
        <audio controls src={latestAudio} style={{ width: "100%" }} />
      ) : (
        <p className="muted">{t("tree.no_audio")}</p>
      )}

      <h3>{t("tree.timeline")}</h3>
      <DetectionTimeline detections={detections} />

      <div style={{ marginTop: "1.5rem" }}>
        <button
          className="btn btn-primary"
          onClick={onTreat}
          disabled={busy || tree.status === "treated"}
        >
          {t("tree.mark_treated")}
        </button>
        {treated && <span className="muted" style={{ marginInlineStart: "0.75rem" }}>
          {t("tree.treated_done")}
        </span>}
      </div>
    </>
  );
}
