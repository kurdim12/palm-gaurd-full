"use client";

import { useEffect } from "react";
import { useI18n } from "@/lib/i18n";

// Keeps <html lang/dir> in sync with the active locale so RTL/LTR flips correctly.
export function HtmlDir() {
  const { locale, dir } = useI18n();
  useEffect(() => {
    document.documentElement.lang = locale;
    document.documentElement.dir = dir;
  }, [locale, dir]);
  return null;
}
