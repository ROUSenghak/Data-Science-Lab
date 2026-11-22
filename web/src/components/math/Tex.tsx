import katex from "katex";

/**
 * KaTeX rendering; works in server and client components.
 * Only \htmlData is trusted: equations use it to mark terms that are linked to plots
 * (see components/anim/Focus.tsx). `active` marks the term whose key is currently focused.
 */
function render(tex: string, display: boolean, active?: string | null) {
  const html = katex.renderToString(tex, {
    displayMode: display,
    throwOnError: false,
    strict: "ignore",
    output: "htmlAndMathml",
    trust: (ctx) => ctx.command === "\\htmlData",
  });
  return active ? html.replaceAll(`data-hl="${active}"`, `data-hl="${active}" data-active=""`) : html;
}

export function Tex({ children, className, active }: { children: string; className?: string; active?: string | null }) {
  return <span className={className} dangerouslySetInnerHTML={{ __html: render(children, false, active) }} />;
}

export function TexBlock({ children, className, active }: { children: string; className?: string; active?: string | null }) {
  return <div className={className} dangerouslySetInnerHTML={{ __html: render(children, true, active) }} />;
}
