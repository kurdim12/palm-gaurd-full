import type { Metadata } from "next";
import "./globals.css";
import { I18nProvider } from "@/lib/i18n";
import { Header } from "@/components/Header";
import { HtmlDir } from "@/components/HtmlDir";

export const metadata: Metadata = {
  title: "Palm Guard | حارس النخيل",
  description: "Acoustic Red Palm Weevil detection dashboard",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  // Default document is Arabic / RTL; HtmlDir syncs <html lang/dir> with the
  // active locale on the client.
  return (
    <html lang="ar" dir="rtl">
      <body>
        <I18nProvider>
          <HtmlDir />
          <Header />
          <main className="container">{children}</main>
        </I18nProvider>
      </body>
    </html>
  );
}
