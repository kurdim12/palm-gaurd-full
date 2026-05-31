"use client";

import { useI18n } from "@/lib/i18n";
import { STATUS_COLOR } from "@/lib/status";
import type { TreeStatus } from "@/lib/types";

export function StatusBadge({ status }: { status: TreeStatus }) {
  const { t } = useI18n();
  return (
    <span className="badge" style={{ background: STATUS_COLOR[status] }}>
      {t(`status.${status}`)}
    </span>
  );
}
