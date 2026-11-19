"use client";

import { Highlight, type PrismTheme } from "prism-react-renderer";
import { useEffect, useRef } from "react";
import type { ParamSpec } from "@/codelab/types";

/** Theme expressed with CSS variables so it follows light/dark mode. */
const theme: PrismTheme = {
  plain: { color: "var(--code-fg)", backgroundColor: "transparent" },
  styles: [
    { types: ["comment"], style: { color: "var(--code-comment)", fontStyle: "italic" } },
    { types: ["keyword", "boolean"], style: { color: "var(--code-keyword)" } },
    { types: ["string", "triple-quoted-string", "string-interpolation"], style: { color: "var(--code-string)" } },
    { types: ["number"], style: { color: "var(--code-number)" } },
    { types: ["function"], style: { color: "var(--code-function)" } },
    { types: ["builtin", "class-name"], style: { color: "var(--code-builtin)" } },
    { types: ["operator", "punctuation"], style: { color: "var(--code-fg)" } },
  ],
};

export function CodeView({
  source,
  active,
  groups,
  params,
  paramLines,
  values,
  onParamChange,
  onLineClick,
  maxHeight = 560,
}: {
  source: string;
  /** Inclusive 1-based line range currently executing. */
  active?: [number, number] | null;
  /** Step groups shown as gutter markers. */
  groups?: { id: string; start: number; end: number }[];
  params?: ParamSpec[];
  paramLines?: Record<string, number>;
  values?: Record<string, number>;
  onParamChange?: (name: string, value: number) => void;
  onLineClick?: (line: number) => void;
  maxHeight?: number;
}) {
  const scroller = useRef<HTMLDivElement>(null);
  const lineByParam = new Map(Object.entries(paramLines ?? {}).map(([k, v]) => [v, k]));
  const specs = new Map((params ?? []).map((p) => [p.name, p]));
  const marked = new Set<number>();
  for (const g of groups ?? []) for (let l = g.start; l <= g.end; l++) marked.add(l);

  // Keep the active line visible inside the code panel (without scrolling the page).
  useEffect(() => {
    const el = scroller.current;
    if (!el || !active) return;
    const line = el.querySelector<HTMLElement>(`[data-line="${active[0]}"]`);
    if (!line) return;
    const top = line.offsetTop;
    if (top < el.scrollTop + 24 || top > el.scrollTop + el.clientHeight - 48) {
      el.scrollTo({ top: Math.max(0, top - el.clientHeight / 3), behavior: "smooth" });
    }
  }, [active]);

  return (
    <div ref={scroller} className="overflow-auto rounded-md border border-line bg-code" style={{ maxHeight }}>
      <Highlight theme={theme} code={source.replace(/\n$/, "")} language="python">
        {({ tokens, getLineProps, getTokenProps }) => (
          <pre className="min-w-max py-2 font-mono text-[12.5px] leading-[1.6]">
            {tokens.map((line, i) => {
              const n = i + 1;
              const isActive = !!active && n >= active[0] && n <= active[1];
              const pname = lineByParam.get(n);
              const spec = pname ? specs.get(pname) : undefined;
              return (
                <div
                  key={i}
                  data-line={n}
                  {...getLineProps({ line })}
                  onClick={onLineClick && marked.has(n) ? () => onLineClick(n) : undefined}
                  className={`flex pr-4 ${isActive ? "bg-[var(--code-line-active)]" : ""} ${onLineClick && marked.has(n) ? "cursor-pointer" : ""}`}
                  aria-current={isActive ? "step" : undefined}
                >
                  <span className={`w-1 shrink-0 ${isActive ? "bg-[var(--viz-model)]" : marked.has(n) ? "bg-line-strong/60" : ""}`} aria-hidden />
                  <span className="w-10 shrink-0 select-none pr-3 text-right text-fg-3 tabular-nums">{n}</span>
                  <span className="whitespace-pre">
                    {spec && values && onParamChange ? (
                      <ParamLine spec={spec} value={values[spec.name]} onChange={(v) => onParamChange(spec.name, v)} rest={line.map((t) => t.content).join("").replace(/^[^#]*/, "")} />
                    ) : line.length === 1 && line[0].empty ? (
                      "\n"
                    ) : (
                      line.map((token, key) => <span key={key} {...getTokenProps({ token })} />)
                    )}
                  </span>
                </div>
              );
            })}
          </pre>
        )}
      </Highlight>
    </div>
  );
}

function ParamLine({ spec, value, onChange, rest }: { spec: ParamSpec; value: number; onChange: (v: number) => void; rest: string }) {
  const invalid = !Number.isFinite(value) || value < spec.min || value > spec.max || (spec.kind === "int" && !Number.isInteger(value));
  return (
    <>
      <span style={{ color: "var(--code-fg)" }}>{spec.name} = </span>
      <input
        type="number"
        aria-label={`${spec.label} (${spec.name})`}
        title={`${spec.description} Range ${spec.min} to ${spec.max}.`}
        value={Number.isFinite(value) ? value : ""}
        min={spec.min}
        max={spec.max}
        step={spec.step}
        onChange={(e) => onChange(e.target.value === "" ? NaN : Number(e.target.value))}
        className={`w-20 rounded border bg-surface px-1 font-mono text-[12.5px] tabular-nums ${invalid ? "border-bad text-bad" : "border-line-strong"}`}
        style={{ color: invalid ? undefined : "var(--code-number)" }}
      />
      {rest && <span style={{ color: "var(--code-comment)", fontStyle: "italic" }}>{"  " + rest}</span>}
      <span className="ml-3 font-sans text-[11px] text-fg-3">
        {spec.min}–{spec.max}
      </span>
    </>
  );
}

export function isParamValid(spec: ParamSpec, v: number) {
  return Number.isFinite(v) && v >= spec.min && v <= spec.max && (spec.kind === "float" || Number.isInteger(v));
}
