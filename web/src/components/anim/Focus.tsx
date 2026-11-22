"use client";

import { createContext, useContext, useMemo, useState, type ReactNode } from "react";
import { Tex, TexBlock } from "@/components/math/Tex";
import { decodeFocus, encodeFocus, type Focus } from "@/lib/anim/focus";

/**
 * Shared focus between equations and plots. Equations mark terms with hl(focus, tex); hovering a
 * term sets the focus, and plots read it to emphasise the same object (and vice versa).
 */

interface FocusCtx {
  focus: Focus | null;
  setFocus: (f: Focus | null) => void;
}

const Ctx = createContext<FocusCtx | null>(null);
const NONE: FocusCtx = { focus: null, setFocus: () => {} };

export function FocusProvider({ children }: { children: ReactNode }) {
  const [focus, setFocus] = useState<Focus | null>(null);
  const value = useMemo(() => ({ focus, setFocus }), [focus]);
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

/** Outside a FocusProvider this is inert, so plots can always call it. */
export function useFocus(): FocusCtx {
  return useContext(Ctx) ?? NONE;
}

/** Container whose marked equation terms set the shared focus on hover or tap. */
export function LinkedMath({ children, className = "" }: { children: ReactNode; className?: string }) {
  const { setFocus } = useFocus();
  const pick = (t: EventTarget) => (t instanceof Element ? decodeFocus(t.closest("[data-hl]")?.getAttribute("data-hl")) : null);
  return (
    <div className={`linked-math ${className}`} onPointerOver={(e) => setFocus(pick(e.target))} onPointerLeave={() => setFocus(null)}>
      {children}
    </div>
  );
}

export function LTex({ children, className }: { children: string; className?: string }) {
  const { focus } = useFocus();
  return <Tex className={className} active={focus ? encodeFocus(focus) : null}>{children}</Tex>;
}

export function LTexBlock({ children, className }: { children: string; className?: string }) {
  const { focus } = useFocus();
  return <TexBlock className={className} active={focus ? encodeFocus(focus) : null}>{children}</TexBlock>;
}
