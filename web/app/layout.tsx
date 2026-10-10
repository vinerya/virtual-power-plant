import type { Metadata } from "next";
import "./globals.css";
import { Providers } from "./providers";

// Render every page per request: the Content-Security-Policy set by
// middleware.ts carries a fresh nonce, and Next.js can only stamp it on the
// inline scripts of a page rendered for that request (a prerendered page
// would ship without it and its scripts would be blocked).
export const dynamic = "force-dynamic";

export const metadata: Metadata = {
  title: "VPP Operator Console",
  description: "Operator console for the open-source Virtual Power Plant",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-background font-sans antialiased">
        <Providers>{children}</Providers>
      </body>
    </html>
  );
}
