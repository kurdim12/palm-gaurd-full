"use client";

// Lightweight bilingual context (AR default, RTL). Kept dependency-free on
// purpose — no i18n library needed for two locales (CLAUDE.md: ask before adding
// deps; prefer boring/minimal).

import { createContext, useCallback, useContext, useMemo, useState } from "react";
import ar from "@/messages/ar.json";
import en from "@/messages/en.json";

export type Locale = "ar" | "en";

const MESSAGES: Record<Locale, Record<string, unknown>> = { ar, en };

interface I18nValue {
  locale: Locale;
  dir: "rtl" | "ltr";
  t: (key: string) => string;
  toggle: () => void;
}

const I18nContext = createContext<I18nValue | null>(null);

function lookup(messages: Record<string, unknown>, key: string): string {
  const parts = key.split(".");
  let node: unknown = messages;
  for (const part of parts) {
    if (typeof node === "object" && node !== null && part in node) {
      node = (node as Record<string, unknown>)[part];
    } else {
      return key;
    }
  }
  return typeof node === "string" ? node : key;
}

export function I18nProvider({ children }: { children: React.ReactNode }) {
  const [locale, setLocale] = useState<Locale>("ar");

  const t = useCallback((key: string) => lookup(MESSAGES[locale], key), [locale]);
  const toggle = useCallback(
    () => setLocale((prev) => (prev === "ar" ? "en" : "ar")),
    [],
  );

  const value = useMemo<I18nValue>(
    () => ({ locale, dir: locale === "ar" ? "rtl" : "ltr", t, toggle }),
    [locale, t, toggle],
  );

  return <I18nContext.Provider value={value}>{children}</I18nContext.Provider>;
}

export function useI18n(): I18nValue {
  const ctx = useContext(I18nContext);
  if (!ctx) {
    throw new Error("useI18n must be used within an I18nProvider");
  }
  return ctx;
}

export function localizedTreeName(
  locale: Locale,
  tree: { name_ar: string; name_en: string; id: string },
): string {
  const name = locale === "ar" ? tree.name_ar : tree.name_en;
  return name || tree.id;
}
