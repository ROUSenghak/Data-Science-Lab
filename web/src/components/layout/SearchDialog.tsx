"use client";

import { useRouter } from "next/navigation";
import { useEffect, useMemo, useRef, useState } from "react";
import { searchIndex, type SearchEntry } from "@/content/curriculum";

function score(e: SearchEntry, q: string): number {
  const terms = q.toLowerCase().split(/\s+/).filter(Boolean);
  if (!terms.length) return 0;
  const title = e.title.toLowerCase();
  const hay = `${title} ${e.context ?? ""} ${e.keywords ?? ""}`.toLowerCase();
  let s = 0;
  for (const t of terms) {
    if (!hay.includes(t)) return -1;
    s += title.startsWith(t) ? 3 : title.includes(t) ? 2 : 1;
  }
  return s;
}

export function SearchDialog() {
  const [open, setOpen] = useState(false);
  const [q, setQ] = useState("");
  const [active, setActive] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);
  const router = useRouter();
  const index = useMemo(() => searchIndex(), []);

  const results = useMemo(() => {
    if (!q.trim()) return index.filter((e) => e.kind !== "Section").slice(0, 8);
    return index
      .map((e) => ({ e, s: score(e, q) }))
      .filter((r) => r.s > 0)
      .sort((a, b) => b.s - a.s)
      .slice(0, 10)
      .map((r) => r.e);
  }, [q, index]);

  const show = () => {
    setQ("");
    setActive(0);
    setOpen(true);
  };

  useEffect(() => {
    const onKey = (ev: KeyboardEvent) => {
      if ((ev.metaKey || ev.ctrlKey) && ev.key.toLowerCase() === "k") {
        ev.preventDefault();
        if (open) setOpen(false);
        else show();
      } else if (ev.key === "/" && !(ev.target instanceof HTMLInputElement) && !(ev.target instanceof HTMLTextAreaElement)) {
        ev.preventDefault();
        show();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open]);

  const go = (e: SearchEntry | undefined) => {
    if (!e) return;
    setOpen(false);
    router.push(e.href);
  };

  return (
    <>
      <button
        type="button"
        onClick={show}
        className="inline-flex h-8 items-center gap-2 rounded-md border border-line px-2.5 text-sm text-fg-3 hover:border-line-strong hover:text-fg-2"
        aria-label="Search (Ctrl K)"
      >
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden>
          <circle cx="11" cy="11" r="7" />
          <path d="m20 20-3.5-3.5" />
        </svg>
        <span className="hidden sm:inline">Search</span>
        <kbd className="hidden rounded border border-line px-1 font-mono text-[10px] sm:inline">⌘K</kbd>
      </button>
      {open && (
        <div className="fixed inset-0 z-50 flex items-start justify-center bg-black/30 px-4 pt-[12vh]" onMouseDown={() => setOpen(false)}>
          <div
            role="dialog"
            aria-modal="true"
            aria-label="Search"
            className="w-full max-w-lg overflow-hidden rounded-lg border border-line bg-surface shadow-xl"
            onMouseDown={(e) => e.stopPropagation()}
          >
            <input
              ref={inputRef}
              autoFocus
              value={q}
              onChange={(e) => {
                setQ(e.target.value);
                setActive(0);
              }}
              onKeyDown={(e) => {
                if (e.key === "ArrowDown") {
                  e.preventDefault();
                  setActive((a) => Math.min(a + 1, results.length - 1));
                } else if (e.key === "ArrowUp") {
                  e.preventDefault();
                  setActive((a) => Math.max(a - 1, 0));
                } else if (e.key === "Enter") go(results[active]);
                else if (e.key === "Escape") setOpen(false);
              }}
              placeholder="Search lessons, sections, labs…"
              className="w-full border-b border-line bg-transparent px-4 py-3 text-base outline-none placeholder:text-fg-3"
              role="combobox"
              aria-expanded="true"
              aria-controls="search-results"
              aria-activedescendant={results[active] ? `sr-${active}` : undefined}
            />
            <ul id="search-results" role="listbox" className="max-h-80 overflow-y-auto py-1">
              {results.length === 0 && <li className="px-4 py-3 text-sm text-fg-3">No results. Only published lessons and labs are searchable.</li>}
              {results.map((r, i) => (
                <li
                  key={r.href + r.title}
                  id={`sr-${i}`}
                  role="option"
                  aria-selected={i === active}
                  onMouseEnter={() => setActive(i)}
                  onClick={() => go(r)}
                  className={`flex cursor-pointer items-baseline justify-between gap-3 px-4 py-2 text-sm ${i === active ? "bg-accent-soft" : ""}`}
                >
                  <span>
                    {r.title}
                    {r.context && <span className="ml-2 text-fg-3">{r.context}</span>}
                  </span>
                  <span className="shrink-0 text-xs text-fg-3">{r.kind}</span>
                </li>
              ))}
            </ul>
          </div>
        </div>
      )}
    </>
  );
}
