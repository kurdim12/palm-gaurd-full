"use client";

import { useI18n } from "@/lib/i18n";
import { STATUS_COLOR, STATUS_ORDER } from "@/lib/status";
import type { StatusCounts } from "@/lib/types";

export function StatusCards({ counts }: { counts: StatusCounts }) {
  const { t } = useI18n();
  return (
    <div className="cards">
      {STATUS_ORDER.map((status) => (
        <div className="card" key={status}>
          <div className="muted">{t(`status.${status}`)}</div>
          <div className="count" style={{ color: STATUS_COLOR[status] }}>
            {counts[status]}
          </div>
        </div>
      ))}
    </div>
  );
}
