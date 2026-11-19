"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState } from "react";
import { SearchDialog } from "./SearchDialog";
import { ThemeToggle } from "./ThemeToggle";

const NAV = [
  { href: "/learn", label: "Learn" },
  { href: "/labs", label: "Labs" },
  { href: "/code-lab", label: "Code Lab" },
  { href: "/projects", label: "Projects" },
];

export function SiteHeader() {
  const pathname = usePathname();
  const [menu, setMenu] = useState(false);
  const isActive = (href: string) => pathname === href || pathname.startsWith(href + "/");

  return (
    <header className="sticky top-0 z-40 border-b border-line bg-bg/90 backdrop-blur">
      <div className="mx-auto flex h-14 max-w-7xl items-center gap-6 px-4 sm:px-6">
        <Link href="/" className="flex items-center gap-2 font-serif text-[17px] font-semibold tracking-tight">
          <svg width="20" height="20" viewBox="0 0 24 24" aria-hidden className="text-accent">
            <path d="M3 20 L21 6" stroke="currentColor" strokeWidth="2" fill="none" />
            <circle cx="6" cy="14" r="1.8" fill="currentColor" />
            <circle cx="11" cy="15" r="1.8" fill="currentColor" />
            <circle cx="15" cy="9" r="1.8" fill="currentColor" />
            <circle cx="19" cy="10" r="1.8" fill="currentColor" />
          </svg>
          Data Science Lab
        </Link>
        <nav aria-label="Primary" className="hidden md:block">
          <ul className="flex items-center gap-1 text-sm">
            {NAV.map((n) => (
              <li key={n.href}>
                <Link
                  href={n.href}
                  aria-current={isActive(n.href) ? "page" : undefined}
                  className={`rounded-md px-3 py-1.5 ${isActive(n.href) ? "text-fg font-medium" : "text-fg-2 hover:text-fg"}`}
                >
                  {n.label}
                </Link>
              </li>
            ))}
          </ul>
        </nav>
        <div className="ml-auto flex items-center gap-1.5">
          <SearchDialog />
          <ThemeToggle />
          <button
            type="button"
            className="inline-flex h-8 w-8 items-center justify-center rounded-md text-fg-2 hover:bg-surface-2 md:hidden"
            aria-label="Menu"
            aria-expanded={menu}
            onClick={() => setMenu((m) => !m)}
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden>
              <path d={menu ? "M6 6l12 12M18 6L6 18" : "M4 7h16M4 12h16M4 17h16"} />
            </svg>
          </button>
        </div>
      </div>
      {menu && (
        <nav aria-label="Primary mobile" className="border-t border-line md:hidden">
          <ul className="mx-auto max-w-7xl px-4 py-2">
            {NAV.map((n) => (
              <li key={n.href}>
                <Link href={n.href} onClick={() => setMenu(false)} className={`block rounded-md px-2 py-2 ${isActive(n.href) ? "font-medium" : "text-fg-2"}`}>
                  {n.label}
                </Link>
              </li>
            ))}
          </ul>
        </nav>
      )}
    </header>
  );
}
