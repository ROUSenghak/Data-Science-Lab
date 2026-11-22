"use client";

import { useMemo } from "react";
import { generateDataset } from "@/lib/ml/linreg";
import { FitExplorer, GradientDescentExplorer } from "./explorers";

const useLessonData = () => useMemo(() => generateDataset({ n: 30, noise: 1, seed: 7 }), []);

export function LessonFit() {
  return <FitExplorer data={useLessonData()} />;
}

export function LessonGD() {
  return <GradientDescentExplorer data={useLessonData()} defaultDetailed />;
}
