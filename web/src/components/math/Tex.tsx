import katex from "katex";

/** KaTeX rendering; works in server and client components. */
function render(tex: string, display: boolean) {
  return katex.renderToString(tex, { displayMode: display, throwOnError: false, strict: "ignore", output: "htmlAndMathml" });
}

export function Tex({ children, className }: { children: string; className?: string }) {
  return <span className={className} dangerouslySetInnerHTML={{ __html: render(children, false) }} />;
}

export function TexBlock({ children, className }: { children: string; className?: string }) {
  return <div className={className} dangerouslySetInnerHTML={{ __html: render(children, true) }} />;
}
