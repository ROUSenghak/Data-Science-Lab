import type { Metadata } from "next";
import Link from "next/link";
import { Breadcrumbs } from "@/components/layout/Breadcrumbs";
import { Tex, TexBlock } from "@/components/math/Tex";
import { Exercise, Figure, Lead } from "@/components/ui/lesson";
import { Callout, Expand, SectionHeading } from "@/components/ui/primitives";
import { K_MEANS_SECTIONS } from "@/content/curriculum";
import { KMeans3DExplorer } from "@/labs/k-means/KMeans3DExplorer";
import { LessonElbow, LessonFailures, LessonInit, LessonLloyd, LessonLocalMinimum } from "@/labs/k-means/LessonWidgets";

export const metadata: Metadata = {
  title: "K-Means Clustering",
  description: "Lloyd's algorithm, its convergence guarantee, local minima, initialization, choosing K, and failure cases.",
};

const linkCls = "text-accent underline underline-offset-2";

export default function KMeansLesson() {
  return (
    <div className="mx-auto max-w-7xl px-4 pb-24 pt-6 sm:px-6">
      <Breadcrumbs items={[{ label: "Learn", href: "/learn" }, { label: "Machine Learning", href: "/learn#machine-learning" }, { label: "K-Means Clustering" }]} />

      <div className="grid gap-10 lg:grid-cols-[minmax(0,1fr)_200px]">
        <article className="min-w-0 max-w-4xl">
          <header className="mt-4">
            <h1 className="font-serif text-4xl font-semibold tracking-tight">K-Means Clustering</h1>
            <p className="mt-2 text-[15px] text-fg-3">Prerequisites: vectors, Euclidean distance, the mean · about 30 minutes</p>
            <div className="mt-4 flex flex-wrap gap-2 text-[13px]">
              <Link href="/labs/k-means" className="rounded-md border border-line-strong px-3 py-1.5 hover:bg-surface-2">
                Open the playground
              </Link>
              <Link href="/code-lab/k-means" className="rounded-md border border-line-strong px-3 py-1.5 hover:bg-surface-2">
                Open in Code Lab
              </Link>
            </div>
          </header>

          {/* 1 */}
          <SectionHeading id="motivation">Motivation</SectionHeading>
          <Lead>
            Given unlabelled points <Tex>{"x_1,\\dots,x_n \\in \\mathbb R^d"}</Tex>, find <Tex>K</Tex> groups of points that are close to each other. K-Means summarizes each group by one prototype, its centroid <Tex>{"\\mu_j"}</Tex>, and puts every point in the group of its nearest centroid.
          </Lead>
          <div className="prose-lab">
            <ul>
              <li>Segmentation: customers, documents, image regions.</li>
              <li>Compression (vector quantization): replace each point by its centroid, e.g. reduce an image to K colours.</li>
              <li>A first look at structure in data, and a building block (e.g. initializing Gaussian mixtures).</li>
            </ul>
          </div>

          {/* 2 */}
          <SectionHeading id="clustering-vs-classification">Clustering vs classification</SectionHeading>
          <div className="overflow-x-auto">
            <table className="w-full text-left text-[14px]">
              <thead className="border-b border-line text-[12px] uppercase tracking-wide text-fg-3">
                <tr>
                  <th className="py-2 pr-4 font-medium" />
                  <th className="py-2 pr-4 font-medium">Classification (supervised)</th>
                  <th className="py-2 pr-4 font-medium">Clustering (unsupervised)</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-line">
                <tr>
                  <td className="py-2.5 pr-4 text-fg-2">Data</td>
                  <td className="py-2.5 pr-4">
                    Pairs <Tex>{"(x_i, y_i)"}</Tex>
                  </td>
                  <td className="py-2.5 pr-4">
                    Points <Tex>{"x_i"}</Tex> only
                  </td>
                </tr>
                <tr>
                  <td className="py-2.5 pr-4 text-fg-2">Goal</td>
                  <td className="py-2.5 pr-4">Predict given labels for new points</td>
                  <td className="py-2.5 pr-4">Discover groups</td>
                </tr>
                <tr>
                  <td className="py-2.5 pr-4 text-fg-2">Labels</td>
                  <td className="py-2.5 pr-4">Fixed meaning (&ldquo;spam&rdquo;)</td>
                  <td className="py-2.5 pr-4">Arbitrary ids: renumbering the clusters changes nothing</td>
                </tr>
                <tr>
                  <td className="py-2.5 pr-4 text-fg-2">Evaluation</td>
                  <td className="py-2.5 pr-4">Accuracy against true labels</td>
                  <td className="py-2.5 pr-4">Internal criteria (J, silhouette) or usefulness downstream</td>
                </tr>
              </tbody>
            </table>
          </div>

          {/* 3 */}
          <SectionHeading id="objective">The objective</SectionHeading>
          <Lead>
            A clustering is a label <Tex>{"c_i \\in \\{1,\\dots,K\\}"}</Tex> per point plus K centroids. K-Means minimizes the <b className="font-medium text-fg">within-cluster sum of squares</b> (scikit-learn: <code className="font-mono text-[14px]">inertia_</code>):
          </Lead>
          <TexBlock>{"J(c, \\mu) = \\sum_{i=1}^{n} \\lVert x_i - \\mu_{c_i} \\rVert^2 = \\sum_{j=1}^{K} \\sum_{i \\in C_j} \\lVert x_i - \\mu_j \\rVert^2, \\qquad C_j = \\{\\, i : c_i = j \\,\\}"}</TexBlock>
          <div className="prose-lab">
            <ul>
              <li>Small J means every point is near its centroid: compact clusters.</li>
              <li>
                With membership indicators <Tex>{"r_{ij} = [c_i = j]"}</Tex>: <Tex>{"J = \\sum_i \\sum_j r_{ij}\\lVert x_i - \\mu_j\\rVert^2"}</Tex>. The NumPy code builds exactly this matrix.
              </li>
              <li>
                Finding the global minimum is NP-hard in general, so K-Means uses a local search instead.
              </li>
            </ul>
          </div>
          <Expand title="Equivalent form: average pairwise distance inside clusters">
            <p>For one cluster with mean <Tex>{"\\bar x_j"}</Tex>,</p>
            <TexBlock>{"\\sum_{i \\in C_j} \\lVert x_i - \\bar x_j \\rVert^2 = \\frac{1}{2|C_j|} \\sum_{i \\in C_j}\\sum_{i' \\in C_j} \\lVert x_i - x_{i'} \\rVert^2."}</TexBlock>
            <p>So, at the optimal centroids, minimizing J is the same as minimizing size-weighted squared distances between points of the same cluster. No centroid is needed to state the goal; it is only a convenient summary.</p>
          </Expand>

          {/* 4 */}
          <SectionHeading id="lloyd">Lloyd&apos;s algorithm</SectionHeading>
          <Lead>Minimize J over labels and centroids in turn. Each half-step is an exact minimization over one block of variables:</Lead>
          <TexBlock>{"\\textbf{assign:}\\quad c_i^{(t)} = \\arg\\min_{j} \\lVert x_i - \\mu_j^{(t-1)} \\rVert^2 \\qquad\\qquad \\textbf{update:}\\quad \\mu_j^{(t)} = \\frac{1}{|C_j^{(t)}|} \\sum_{i \\in C_j^{(t)}} x_i"}</TexBlock>
          <div className="prose-lab">
            <ul>
              <li>Start from initial centroids <Tex>{"\\mu^{(0)}"}</Tex>; stop when no label changes.</li>
              <li>
                The boundary between two centroids is their <b>perpendicular bisector</b>. Each cluster region is an intersection of half-planes (a Voronoi cell), hence convex.
              </li>
              <li>
                Ties go to the lowest index. An <b>empty cluster</b> has no mean; here it keeps its previous centroid (scikit-learn instead moves it to a far-away point).
              </li>
            </ul>
          </div>
          <Expand title="Why each half-step minimizes J">
            <p>
              <b>Assignment.</b> For fixed <Tex>\mu</Tex>, J is a sum of independent terms <Tex>{"\\lVert x_i - \\mu_{c_i}\\rVert^2"}</Tex>, each minimized by choosing the nearest centroid.
            </p>
            <p>
              <b>Update.</b> For fixed labels, J separates over clusters. For a cluster with mean <Tex>{"\\bar x"}</Tex> and any <Tex>\mu</Tex>:
            </p>
            <TexBlock>{"\\sum_{i \\in C} \\lVert x_i - \\mu \\rVert^2 = \\sum_{i \\in C} \\lVert x_i - \\bar x \\rVert^2 + |C|\\,\\lVert \\bar x - \\mu \\rVert^2,"}</TexBlock>
            <p>
              because the cross term <Tex>{"2(\\bar x - \\mu)^\\top \\sum_i (x_i - \\bar x)"}</Tex> vanishes. The second term is zero exactly at <Tex>{"\\mu = \\bar x"}</Tex>: the mean is the unique minimizer.
            </p>
          </Expand>
          <Figure title="Step through Lloyd's algorithm (click a point to see its distances)">
            <LessonLloyd />
          </Figure>

          {/* 5 */}
          <SectionHeading id="convergence">Convergence and local minima</SectionHeading>
          <Callout title="Guarantee: J never increases, and the algorithm stops">
            <TexBlock>{"J(c^{(t)}, \\mu^{(t-1)}) \\;\\ge\\; J(c^{(t)}, \\mu^{(t)}) \\;\\ge\\; J(c^{(t+1)}, \\mu^{(t)})"}</TexBlock>
            Each half-step minimizes J over one block of variables, so it cannot increase J. There are finitely many labelings and none can repeat (expand below), so Lloyd&apos;s algorithm stops after finitely many iterations.
          </Callout>
          <Expand title="Proof that it terminates">
            <p>
              Suppose a complete iteration leaves J unchanged. Then the update was an equality, so every non-empty cluster&apos;s centroid already equalled its mean (the minimizer is unique); empty clusters keep theirs. The centroids did not move, so the next assignment, which is deterministic, reproduces the same labels and the loop stops.
            </p>
            <p>
              Hence J strictly decreases in every iteration except the final ones. A labeling can therefore never repeat, and there are at most <Tex>{"K^n"}</Tex> labelings. In practice a few dozen iterations suffice, although worst-case inputs needing exponentially many iterations exist.
            </p>
          </Expand>
          <div className="prose-lab">
            <p>
              <b>What is not guaranteed:</b> the global minimum. The algorithm stops at a <i>fixed point</i>, where every point is in its nearest cluster and every centroid is its cluster&apos;s mean. A typical bad fixed point puts two centroids in one true cluster and one centroid between two others. No single step can repair this, because moving any one centroid alone increases J.
            </p>
          </div>
          <Figure title="Same data, same K, two random starts">
            <LessonLocalMinimum />
          </Figure>

          {/* 6 */}
          <SectionHeading id="initialization">Initialization</SectionHeading>
          <div className="prose-lab">
            <ul>
              <li>
                <b>Random points</b> (Forgy): K distinct data points. Simple; often puts two centroids in one cluster.
              </li>
              <li>
                <b>Uniform in the bounding box</b>: centroids can land far from all data and end up with empty clusters.
              </li>
              <li>
                <b>k-means++</b>: draw the first centroid uniformly, then each next one with probability
                <TexBlock>{"P(x_i) = \\frac{D(x_i)^2}{\\sum_{i'} D(x_{i'})^2}, \\qquad D(x) = \\min_{\\text{chosen } \\mu} \\lVert x - \\mu \\rVert ."}</TexBlock>
                Spread-out starts are likely. Guarantee (Arthur &amp; Vassilvitskii, 2007): the <i>expected</i> J of the initialization alone is at most <Tex>{"8(\\ln K + 2)"}</Tex> times the optimum. This holds in expectation, not for every run.
              </li>
              <li>
                <b>Restarts</b>: run several initializations and keep the lowest J (scikit-learn&apos;s <code>n_init</code>).
              </li>
            </ul>
          </div>
          <Figure title="30 initializations per method on the lesson data (K = 3)">
            <LessonInit />
          </Figure>

          {/* 7 */}
          <SectionHeading id="choosing-k">Choosing K</SectionHeading>
          <Lead>
            J cannot choose K. The best achievable value <Tex>{"J^*(K)"}</Tex> never increases with K, and <Tex>{"J^*(n) = 0"}</Tex> with one centroid per point.
          </Lead>
          <div className="prose-lab">
            <ul>
              <li>
                <b>Elbow:</b> look for the K after which <Tex>{"J^*"}</Tex> stops dropping sharply. Often ambiguous.
              </li>
              <li>
                <b>Silhouette:</b> for point i, with <Tex>{"a_i"}</Tex> the mean distance to its own cluster and <Tex>{"b_i"}</Tex> the smallest mean distance to another cluster,
                <TexBlock>{"s_i = \\frac{b_i - a_i}{\\max(a_i, b_i)} \\in [-1, 1]."}</TexBlock>
                Pick K with a high mean <Tex>{"s_i"}</Tex>. It is undefined for K = 1.
              </li>
              <li>
                <b>Purpose:</b> in practice the use of the clusters (segments a team can act on, a colour budget) often fixes K.
              </li>
            </ul>
          </div>
          <Figure title="Elbow and silhouette on the lesson data">
            <LessonElbow />
          </Figure>

          <SectionHeading id="more-features">More than two features</SectionHeading>
          <Lead>
            Nothing in Lloyd&apos;s algorithm depends on the number of features. The distance simply sums over all of them, <Tex>{"\\lVert x - \\mu\\rVert^2 = \\sum_{d=1}^{D}(x_d - \\mu_d)^2"}</Tex>, and each centroid averages every feature separately.
          </Lead>
          <div className="prose-lab">
            <p>
              The data below has three features and three generating groups. Two groups have the same centre in <Tex>x_1</Tex> and <Tex>x_2</Tex> and differ only in <Tex>x_3</Tex>. Step through a run, then press <i>Top view</i>: it shows exactly the projection onto <Tex>{"(x_1, x_2)"}</Tex>, where those two groups are a single blob. A plot of two features can hide structure that the algorithm, using all features, finds.
            </p>
            <p>
              The flip side: because the squared differences are added, a feature measured on a larger scale dominates the distance. Standardize features (or choose their units deliberately) before clustering.
            </p>
          </div>
          <Figure title="Lloyd's algorithm with three features">
            <KMeans3DExplorer />
          </Figure>

          {/* 8 */}
          <SectionHeading id="limitations">Limitations and failure cases</SectionHeading>
          <div className="prose-lab">
            <ul>
              <li>
                <b>Round, similar-size clusters.</b> K-Means behaves like a mixture of Gaussians with equal, isotropic variances and hard assignments. Elongated or unequal clusters are cut along bisectors.
              </li>
              <li>
                <b>Convex cells only.</b> Rings, moons and other non-convex shapes cannot be single clusters.
              </li>
              <li>
                <b>Scale.</b> Euclidean distance mixes units: standardize features first, or one feature dominates.
              </li>
              <li>
                <b>Outliers.</b> Squared distances let a few far points drag a mean. K-medoids and K-medians are more robust.
              </li>
              <li>
                <b>Always K clusters.</b> Even uniform noise is split into K &ldquo;clusters&rdquo;.
              </li>
            </ul>
          </div>
          <Figure title="Best of 10 k-means++ runs on data that violates the assumptions">
            <LessonFailures />
          </Figure>
          <p className="text-[14px] text-fg-3">
            Load these datasets in the{" "}
            <Link className={linkCls} href="/labs/k-means#failure-cases">
              playground
            </Link>{" "}
            and colour the points by their generating groups to compare.
          </p>

          {/* 9 */}
          <SectionHeading id="implementation">Implementation</SectionHeading>
          <Lead>The Code Lab runs two implementations on the same data and records each step for inspection.</Lead>
          <div className="mt-4 overflow-x-auto">
            <table className="w-full text-left text-[14px]">
              <thead className="border-b border-line text-[12px] uppercase tracking-wide text-fg-3">
                <tr>
                  <th className="py-2 pr-4 font-medium">Version</th>
                  <th className="py-2 pr-4 font-medium">Computed by</th>
                  <th className="py-2 pr-4 font-medium">What you can inspect</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-line">
                <tr>
                  <td className="py-2.5 pr-4">
                    <Link className={linkCls} href="/code-lab/k-means?framework=numpy">
                      NumPy
                    </Link>
                  </td>
                  <td className="py-2.5 pr-4">Our own Lloyd loop</td>
                  <td className="py-2.5 pr-4">Distance matrix, labels, membership sums, centroids, J after every half-step</td>
                </tr>
                <tr>
                  <td className="py-2.5 pr-4">
                    <Link className={linkCls} href="/code-lab/k-means?framework=sklearn">
                      scikit-learn
                    </Link>
                  </td>
                  <td className="py-2.5 pr-4">
                    <code className="font-mono text-[13px]">KMeans.fit</code> (compiled code)
                  </td>
                  <td className="py-2.5 pr-4">
                    Only <code className="font-mono text-[13px]">cluster_centers_</code>, <code className="font-mono text-[13px]">labels_</code>, <code className="font-mono text-[13px]">inertia_</code>, <code className="font-mono text-[13px]">n_iter_</code>, <code className="font-mono text-[13px]">transform</code>; iterations are not exposed
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
          <p className="mt-3 text-[14px] text-fg-3">
            With the same initial centroids, <code className="font-mono">tol=0</code> and the same <code className="font-mono">max_iter</code>, both give identical labels, centroids, inertia and iteration counts. The test suite checks this.
          </p>

          {/* 10 */}
          <SectionHeading id="exercises">Exercises</SectionHeading>
          <ol className="mt-2">
            <Exercise
              n={1}
              solution={
                <>
                  <p>
                    Iteration 1: labels (1, 2, 2, 2), <Tex>{"J = 0 + 0 + 9 + 16 = 25"}</Tex>. Update: <Tex>{"\\mu_1 = 0,\\ \\mu_2 = 10/3"}</Tex>, <Tex>{"J = 49/9 + 4/9 + 25/9 = 78/9 \\approx 8.67"}</Tex>.
                  </p>
                  <p>
                    Iteration 2: the point 1 is nearer to 0 (distance² 1) than to 10/3 (49/9), so labels become (1, 1, 2, 2), <Tex>{"J = 1 + 4/9 + 25/9 = 38/9 \\approx 4.22"}</Tex>. Update: <Tex>{"\\mu = (0.5, 4.5)"}</Tex>, <Tex>{"J = 1"}</Tex>.
                  </p>
                  <p>Iteration 3: no label changes, so the algorithm stops with J = 1. The sequence 25, 8.67, 4.22, 1 never increases.</p>
                </>
              }
            >
              <p>
                Points on a line at 0, 1, 4, 5 with K = 2 and initial centroids <Tex>{"\\mu_1 = 0,\\ \\mu_2 = 1"}</Tex>. Run Lloyd&apos;s algorithm by hand and record J after every half-step.
              </p>
            </Exercise>
            <Exercise
              n={2}
              solution={
                <p>
                  Take an optimal solution with K clusters. Some point <Tex>{"x_i"}</Tex> is not equal to its centroid unless <Tex>{"J^*(K) = 0"}</Tex>. Add a centroid at <Tex>{"x_i"}</Tex> and keep everything else. That term drops to 0 and no other term grows, so <Tex>{"J^*(K+1) \\le J^*(K)"}</Tex>, strictly less if <Tex>{"J^*(K) > 0"}</Tex>.
                </p>
              }
            >
              <p>
                Show that <Tex>{"J^*(K+1) \\le J^*(K)"}</Tex>. Why does this make J useless for choosing K on its own?
              </p>
            </Exercise>
            <Exercise
              n={3}
              solution={
                <p>
                  Every K-Means cluster is a Voronoi cell, which is convex. A convex set containing the whole outer ring contains its convex hull, i.e. the whole disk, including the inner ring. So no K = 2 clustering separates the rings. Fixes: transform the features (e.g. use the radius <Tex>{"\\lVert x \\rVert"}</Tex>), or use a method that allows non-convex clusters, such as spectral clustering or DBSCAN.
                </p>
              }
            >
              <p>Prove that K-Means with K = 2 cannot separate two concentric rings, whatever the initialization. Suggest a fix.</p>
            </Exercise>
            <Exercise
              n={4}
              solution={
                <p>
                  Distances are dominated by the feature with the largest spread. If income is in dollars (spread ≈ 10⁴) and age in years (spread ≈ 10), the clusters are essentially income bands. Standardize each feature, <Tex>{"(x - \\bar x)/s"}</Tex>, or choose weights that reflect what &ldquo;similar&rdquo; should mean.
                </p>
              }
            >
              <p>Customers are described by yearly income (dollars) and age (years). What will K-Means find without preprocessing, and why?</p>
            </Exercise>
          </ol>

          <nav className="mt-14 flex flex-wrap justify-between gap-3 border-t border-line pt-5 text-sm" aria-label="Lesson navigation">
            <Link href="/learn/machine-learning/linear-regression" className="text-fg-2 hover:text-fg">
              ← Linear regression
            </Link>
            <span className="text-fg-3">Next: Principal component analysis (planned)</span>
          </nav>
        </article>

        <aside className="hidden lg:block">
          <nav aria-label="On this page" className="sticky top-20 mt-6 text-[13px]">
            <div className="mb-2 font-medium text-fg-2">On this page</div>
            <ul className="space-y-1.5 border-l border-line">
              {K_MEANS_SECTIONS.map((s) => (
                <li key={s.id}>
                  <a href={`#${s.id}`} className="-ml-px block border-l border-transparent pl-3 text-fg-3 hover:border-fg-2 hover:text-fg">
                    {s.title}
                  </a>
                </li>
              ))}
            </ul>
            <div className="mt-6 mb-2 font-medium text-fg-2">Related</div>
            <ul className="space-y-1.5 text-fg-3">
              <li>
                <Link href="/labs/k-means" className="hover:text-fg">
                  Playground
                </Link>
              </li>
              <li>
                <Link href="/code-lab/k-means" className="hover:text-fg">
                  Code Lab
                </Link>
              </li>
            </ul>
          </nav>
        </aside>
      </div>
    </div>
  );
}
