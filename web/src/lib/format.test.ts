import katex from "katex";
import { describe, expect, it } from "vitest";
import { texNum, texSq } from "./format";

const renders = (tex: string) => () => katex.renderToString(tex, { throwOnError: true, strict: "error" });

describe("numbers in equations", () => {
  // Diverging gradient descent produces residuals like 8.1e25; they must still typeset.
  const values = [0, 2, -3.5, 1e-4, -2.5e-7, 8.1e25, -7.6e26, 123456, Infinity, -Infinity, NaN];

  it("squares typeset for every magnitude and sign (no double superscripts)", () => {
    for (const v of values) expect(renders(`L = ${texSq(v, 2)} + ${texSq(v, 3)}`)).not.toThrow();
  });

  it("negatives and scientific notation are parenthesized before squaring", () => {
    expect(texSq(-3.5, 1)).toBe("(-3.5)^2");
    expect(texSq(2, 1)).toBe("2.0^2");
    expect(texSq(8.1e25, 2)).toBe(`(${texNum(8.1e25, 2)})^2`);
  });
});
