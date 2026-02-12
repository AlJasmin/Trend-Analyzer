import type { Metadata } from "next";
import "./globals.css";

const title = process.env.DASHBOARD_TITLE || "Atlas of Debate";

export const metadata: Metadata = {
  title,
  description: "Live dashboard for Reddit AI discourse analysis.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <div className="backdrop" aria-hidden="true" />
        <div className="grain" aria-hidden="true" />
        {children}
      </body>
    </html>
  );
}
