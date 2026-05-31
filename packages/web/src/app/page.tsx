"use client";

import dynamic from "next/dynamic";
import { useEffect, useState } from "react";
import { StatusCards } from "@/components/StatusCards";
import { api } from "@/lib/api";
import { useI18n } from "@/lib/i18n";
import type { StatusCounts, Tree } from "@/lib/types";

// Leaflet touches `window`, so the map is client-only (no SSR).
const FarmMap = dynamic(() => import("@/components/FarmMap"), { ssr: false });

export default function HomePage() {
  const { t } = useI18n();
  const [trees, setTrees] = useState<Tree[]>([]);
  const [counts, setCounts] = useState<StatusCounts | null>(null);
  const [error, setError] = useState(false);

  useEffect(() => {
    Promise.all([api.listTrees(), api.statusCounts()])
      .then(([ts, cs]) => {
        setTrees(ts);
        setCounts(cs);
      })
      .catch(() => setError(true));
  }, []);

  if (error) return <p className="muted">{t("common.error")}</p>;
  if (!counts) return <p className="muted">{t("common.loading")}</p>;

  return (
    <>
      <StatusCards counts={counts} />
      <h2>{t("map.title")}</h2>
      {trees.length === 0 ? (
        <p className="muted">{t("map.no_trees")}</p>
      ) : (
        <FarmMap trees={trees} />
      )}
    </>
  );
}
