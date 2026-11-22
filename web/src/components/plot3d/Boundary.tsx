"use client";

import { Component, type ReactNode } from "react";

/** Shows `fallback` instead of a 3D view that failed (e.g. the WebGL context could not be created). */
export class Plot3DBoundary extends Component<{ fallback: ReactNode; children: ReactNode }, { failed: boolean }> {
  state = { failed: false };

  static getDerivedStateFromError() {
    return { failed: true };
  }

  componentDidCatch(error: unknown) {
    console.warn("3D view failed; showing the 2D view instead.", error);
  }

  render() {
    return this.state.failed ? this.props.fallback : this.props.children;
  }
}
