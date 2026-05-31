"use client";

import Link from "next/link";
import { useCallback, useEffect, useState } from "react";
import { api } from "@/lib/api";
import { useI18n } from "@/lib/i18n";
import type { Alert } from "@/lib/types";

export default function AlertsPage() {
  const { t, locale } = useI18n();
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [onlyOpen, setOnlyOpen] = useState(true);
  const [error, setError] = useState(false);

  const load = useCallback(() => {
    api
      .listAlerts(onlyOpen)
      .then(setAlerts)
      .catch(() => setError(true));
  }, [onlyOpen]);

  useEffect(load, [load]);

  const onAck = async (id: string) => {
    try {
      await api.acknowledgeAlert(id);
      load();
    } catch {
      setError(true);
    }
  };

  if (error) return <p className="muted">{t("common.error")}</p>;

  return (
    <>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
        <h2>{t("alerts.inbox_title")}</h2>
        <label className="muted" style={{ display: "flex", gap: "0.4rem", alignItems: "center" }}>
          <input
            type="checkbox"
            checked={onlyOpen}
            onChange={(e) => setOnlyOpen(e.target.checked)}
          />
          {t("alerts.open_only")}
        </label>
      </div>

      {alerts.length === 0 ? (
        <p className="muted">{t("alerts.empty")}</p>
      ) : (
        alerts.map((alert) => (
          <div className="alert-row" key={alert.id}>
            <div>
              <div>{locale === "ar" ? alert.message_ar : alert.message_en}</div>
              <div className="muted">
                {t("alerts.created_at")}:{" "}
                {new Date(alert.created_at).toLocaleString(locale === "ar" ? "ar-SA" : "en-GB")}
              </div>
              <Link className="muted" href={`/tree/${alert.tree_id}`}>
                {t("alerts.view_tree")} →
              </Link>
            </div>
            <div>
              {alert.acknowledged ? (
                <span className="muted">{t("alerts.acknowledged")}</span>
              ) : (
                <button className="btn" onClick={() => onAck(alert.id)}>
                  {t("alerts.acknowledge")}
                </button>
              )}
            </div>
          </div>
        ))
      )}
    </>
  );
}
