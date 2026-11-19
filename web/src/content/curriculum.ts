/**
 * Single source of truth for what exists on the site.
 * `status: "available"` items have real pages; `planned` items are listed as such and never linked.
 */

export type Status = "available" | "planned";
export type Level = "Foundations" | "Intermediate" | "Advanced";

export interface Topic {
  slug: string;
  title: string;
  summary: string;
  status: Status;
  href?: string;
  /** In-page sections for search and the table of contents. */
  sections?: { id: string; title: string }[];
}

export interface Module {
  slug: string;
  title: string;
  blurb: string;
  topics: Topic[];
}

export const LINEAR_REGRESSION_SECTIONS = [
  { id: "motivation", title: "Motivation" },
  { id: "model", title: "The model" },
  { id: "loss", title: "Mean squared error" },
  { id: "closed-form", title: "Closed-form solution" },
  { id: "gradient-descent", title: "Gradient descent" },
  { id: "implementation", title: "Implementation" },
  { id: "limitations", title: "Assumptions and failure cases" },
  { id: "exercises", title: "Exercises" },
];

const planned = (slug: string, title: string, summary: string): Topic => ({ slug, title, summary, status: "planned" });

export const MODULES: Module[] = [
  {
    slug: "mathematical-foundations",
    title: "Mathematical Foundations",
    blurb: "Linear algebra, calculus and optimization used throughout the course.",
    topics: [
      planned("vectors-matrices", "Vectors and matrices", "Geometry of linear maps, rank, projections."),
      planned("derivatives-gradients", "Derivatives and gradients", "Partial derivatives, the chain rule, Jacobians."),
      planned("convex-optimization", "Convex optimization", "Convexity, first-order conditions, gradient methods."),
    ],
  },
  {
    slug: "probability-statistics",
    title: "Probability and Statistics",
    blurb: "Random variables, estimation and uncertainty.",
    topics: [
      planned("random-variables", "Random variables", "Distributions, expectation, variance."),
      planned("central-limit-theorem", "Central limit theorem", "Why sample means become Gaussian."),
      planned("maximum-likelihood", "Maximum likelihood", "Estimation as optimization."),
    ],
  },
  {
    slug: "machine-learning",
    title: "Machine Learning",
    blurb: "Supervised and unsupervised learning from first principles.",
    topics: [
      {
        slug: "linear-regression",
        title: "Linear Regression",
        summary: "Least squares, gradient descent, and why the learning rate has a hard limit.",
        status: "available",
        href: "/learn/machine-learning/linear-regression",
        sections: LINEAR_REGRESSION_SECTIONS,
      },
      planned("logistic-regression", "Logistic regression", "Probabilistic classification and cross-entropy."),
      planned("k-means", "K-Means clustering", "Lloyd's algorithm and its failure modes."),
      planned("pca", "Principal component analysis", "Variance, eigenvectors, and projections."),
    ],
  },
  {
    slug: "deep-learning",
    title: "Deep Learning",
    blurb: "Neural networks, backpropagation and training dynamics.",
    topics: [
      planned("mlp", "Multilayer perceptrons", "Nonlinear function approximation."),
      planned("backpropagation", "Backpropagation", "Reverse-mode automatic differentiation."),
    ],
  },
  {
    slug: "generative-ai",
    title: "Generative AI and LLMs",
    blurb: "Attention, transformers and language models.",
    topics: [planned("attention", "Attention and transformers", "Scaled dot-product attention, step by step.")],
  },
  {
    slug: "time-series-causal",
    title: "Time Series and Causal Inference",
    blurb: "Temporal dependence, forecasting and causal effects.",
    topics: [
      planned("autoregressive-models", "Autoregressive models", "AR processes and stationarity."),
      planned("causal-graphs", "Causal graphs", "Confounding and adjustment."),
    ],
  },
  {
    slug: "applied-data-science",
    title: "Applied Data Science",
    blurb: "Validation, feature engineering and communicating results.",
    topics: [planned("cross-validation", "Cross-validation", "Estimating generalization error honestly.")],
  },
];

export type Execution = "browser" | "python";

export interface Lab {
  slug: string;
  title: string;
  summary: string;
  topic: string;
  level: Level;
  status: Status;
  execution: Execution[];
  href?: string;
}

export const LABS: Lab[] = [
  {
    slug: "linear-regression",
    title: "Linear regression playground",
    summary: "Fit a line by hand, then watch gradient descent on the exact loss surface. Includes failure cases.",
    topic: "Machine Learning",
    level: "Foundations",
    status: "available",
    execution: ["browser"],
    href: "/labs/linear-regression",
  },
  {
    slug: "linear-regression-code",
    title: "Linear regression in NumPy, PyTorch and scikit-learn",
    summary: "Step through real Python execution with synchronized code, plots and equations.",
    topic: "Machine Learning",
    level: "Foundations",
    status: "available",
    execution: ["python"],
    href: "/code-lab",
  },
  { slug: "clt", title: "Central limit theorem", summary: "Sample means from arbitrary distributions.", topic: "Probability and Statistics", level: "Foundations", status: "planned", execution: ["browser"] },
  { slug: "k-means", title: "K-Means", summary: "Assignment and update steps; bad initializations.", topic: "Machine Learning", level: "Intermediate", status: "planned", execution: ["browser", "python"] },
  { slug: "pca", title: "PCA", summary: "Project data onto principal axes; explained variance.", topic: "Machine Learning", level: "Intermediate", status: "planned", execution: ["browser", "python"] },
  { slug: "mlp", title: "Neural network training", summary: "Activations, gradients and loss for a small MLP.", topic: "Deep Learning", level: "Intermediate", status: "planned", execution: ["python"] },
  { slug: "attention", title: "Transformer attention", summary: "Queries, keys, values and attention weights.", topic: "Generative AI and LLMs", level: "Advanced", status: "planned", execution: ["python"] },
];

export interface Project {
  title: string;
  summary: string;
  concepts: string[];
}

export const PROJECTS: Project[] = [
  { title: "Predicting house prices", summary: "Feature engineering, regularized regression and cross-validation on tabular data.", concepts: ["Linear regression", "Cross-validation", "Feature scaling"] },
  { title: "Customer segmentation", summary: "Cluster purchase behaviour and interpret the clusters.", concepts: ["K-Means", "PCA", "Standardization"] },
  { title: "Forecasting energy demand", summary: "Seasonality, autoregressive models and honest backtesting.", concepts: ["Time series", "Validation", "Regression"] },
];

export interface SearchEntry {
  title: string;
  href: string;
  kind: "Page" | "Lesson" | "Section" | "Lab";
  context?: string;
  keywords?: string;
}

export function searchIndex(): SearchEntry[] {
  const out: SearchEntry[] = [
    { title: "Home", href: "/", kind: "Page" },
    { title: "Learn — all modules", href: "/learn", kind: "Page", keywords: "curriculum catalogue topics" },
    { title: "Labs — all experiments", href: "/labs", kind: "Page", keywords: "playground experiments interactive" },
    { title: "Code Lab", href: "/code-lab", kind: "Page", keywords: "python numpy pytorch scikit-learn sklearn code execution" },
    { title: "Projects", href: "/projects", kind: "Page" },
  ];
  for (const m of MODULES)
    for (const t of m.topics) {
      if (t.status !== "available" || !t.href) continue;
      out.push({ title: t.title, href: t.href, kind: "Lesson", context: m.title, keywords: t.summary });
      for (const s of t.sections ?? []) out.push({ title: s.title, href: `${t.href}#${s.id}`, kind: "Section", context: t.title });
    }
  for (const l of LABS)
    if (l.status === "available" && l.href) out.push({ title: l.title, href: l.href, kind: "Lab", context: l.topic, keywords: l.summary });
  out.push(
    { title: "Learning rate and divergence", href: "/labs/linear-regression#gradient-descent", kind: "Lab", context: "Linear regression playground", keywords: "learning rate eta diverge stability hessian eigenvalue" },
    { title: "Outliers and nonlinear data", href: "/labs/linear-regression#failure-cases", kind: "Lab", context: "Linear regression playground", keywords: "failure outlier misspecification residuals" },
    { title: "PyTorch autograd: backward()", href: "/code-lab?framework=pytorch", kind: "Lab", context: "Code Lab", keywords: "backward gradient optimizer sgd zero_grad tensor" },
    { title: "scikit-learn LinearRegression", href: "/code-lab?framework=sklearn", kind: "Lab", context: "Code Lab", keywords: "fit closed form normal equations lstsq" },
  );
  return out;
}
