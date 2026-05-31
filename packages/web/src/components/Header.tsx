"use client";

import Link from "next/link";
import { useI18n } from "@/lib/i18n";

export function Header() {
  const { t, toggle } = useI18n();
  return (
    <header className="header">
      <div>
        <h1>{t("app.title")}</h1>
        <div className="sub">{t("app.subtitle")}</div>
      </div>
      <nav className="nav">
        <Link href="/">{t("app.nav_map")}</Link>
        <Link href="/alerts">{t("app.nav_alerts")}</Link>
        <button className="btn" onClick={toggle} aria-label="toggle language">
          {t("app.language")}
        </button>
      </nav>
    </header>
  );
}
