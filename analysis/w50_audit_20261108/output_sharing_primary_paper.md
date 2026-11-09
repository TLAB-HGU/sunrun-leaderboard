# Research: GBDT-MO original-paper evidence for output sharing

## Request Type

Primary-paper reference lookup for a possible output-strategy control. No training, inference, score/truth access, experiment selection, outreach, policy change, or edits to earlier evidence.

## Direct Answer

**The full paper supports a tradeoff between partition flexibility and the regularization supplied by sharing. It does not establish that independent partitions improve forecasting.** Its scalar trees often fit training data faster, while shared vector trees achieve better test behavior in the studied correlated-output problems. Thus the paper is both support for investigating a training-fit restriction and counterevidence to treating training-fit gains as sufficient grounds to remove sharing.

Full text is now verified: [Zhang and Jung, *GBDT-MO: Gradient Boosted Decision Trees for Multiple Outputs*, arXiv:1909.04373v2](https://arxiv.org/pdf/1909.04373v2), dated28December2019, marked “UNDER REVIEW.” Exa's unversioned PDF endpoint returned the complete13-page document, including its explicit v2 version stamp, derivation, experiments and appendices. The earlier abstract-only retrieval gap is resolved **in this new artifact**. The earlier mechanism report remains unchanged.

## Primary-Paper Derivation

In [SectionIII-A, equations10–16](https://arxiv.org/pdf/1909.04373v2#page=4), one partition q(x) selects a leaf containing a vector w of d outputs. For training rows in a leaf, write G=Σ_i g_i and H=Σ_i H_i. Under the second-order loss approximation and L2 penalty λ‖w‖²/2:

- Exact leaf optimum: **w*=−(H+λI)⁻¹G**.
- Optimized objective, omitting the prediction-independent term: **−Gᵀ(H+λI)⁻¹G/2**.
- For diagonal H: **w*_j=−G_j/(H_j+λ)** and objective **−Σ_j G_j²/(H_j+λ)/2**.

[SectionIII-C](https://arxiv.org/pdf/1909.04373v2#page=5) explicitly states that the split's objective gain is the sum of gains over outputs. A candidate split therefore maximizes the aggregate child-versus-parent improvement, while output-specific weights remain separate. This is shared partition selection, not cancellation from adding opposite signed gradients across outputs.

For non-diagonal Hessians, the authors use a diagonal approximation to avoid matrix storage/inversion. [AppendixA, equations26–29](https://arxiv.org/pdf/1909.04373v2#page=12) assumes a scaling γ satisfying γH_ii≥Σ_j|H_ij| and derives a bound on the quadratic form using γ·diag(H). It omits the Taylor remainder and discusses absorbing that scale through learning-rate and L2 adjustments. This is **not** a proof that any unscaled diagonal surrogate is exact or provides the same bound for the current tail-plus-spectral objective.

The authors acknowledge that only subsets of outputs may be correlated and introduce L0-constrained sparse leaf values ([SectionIII-B/D](https://arxiv.org/pdf/1909.04373v2#page=4)). Sparse leaves are a distinct mechanism, not equivalent to independent trees. This report does not recommend adopting that variant.

## Evidence for Sharing, and Limits on the Claim

The authors describe shared structure as “indirect regularization.” On their synthetic problems, outputs deliberately share latent structure; their train/test curves show a smaller generalization gap for GBDT-MO. On MNIST and Caltech101, they explicitly report higher training loss but better test performance for MO than SO. They also explicitly acknowledge **slower convergence, especially early in training**. [Figures3–4](https://arxiv.org/pdf/1909.04373v2#page=8), [SectionsV–VI](https://arxiv.org/pdf/1909.04373v2#page=10).

This supports the following distinction:

| Statement | Evidence status |
|---|---|
| Shared cuts can constrain local training fit | Consistent with the derivation and reported slower convergence. |
| Independent cuts may better follow output-specific predictive structure | Mechanistic inference; not tested here on solar wind. |
| Independent cuts may also fit output-specific noise | Supported by the paper's observed sharing/generalization tradeoff; not a measured current-model effect. |
| Sharing always wins | Contradicted by uneven comparisons in the paper. |
| Current local split regret predicts forecast improvement | Not established by the paper or a training-only diagnostic. |

There are useful counterexamples and weak results within the paper itself. [TableV](https://arxiv.org/pdf/1909.04373v2#page=9) gives Yeast accuracy62.94% for XGBoost versus62.29% for MO. On Student-por, SO/MO RMSE is0.24408/0.24392, a small difference. [TableXII](https://arxiv.org/pdf/1909.04373v2#page=13) reports the authors' confidence that MO beats SO on Student-por as0.561, and MO beats XGBoost on Yeast as0.067. These values must not be presented as conventional strong statistical evidence of universal superiority. The authors' confidence calculation is described in AppendixC, not a modern multiple-comparison guarantee.

The paper does not present a controlled sweep from compatible to antagonistic output partitions, nor a quantified negative-transfer threshold. Its motivation about output correlation cannot substitute for measuring candidate-cut compatibility. High target correlation alone does not prove that the useful tree partitions coincide.

## Exact Experimental Conditions

[Synthetic experiments, SectionV-A and TableIII](https://arxiv.org/pdf/1909.04373v2#page=7):

- **Modified Friedman1:**10 inputs sampled from U(−1,1), with f(x)=sin(πx1x2)+2(x3−0.5)²+x4+0.5x5. Five outputs are independently noise-corrupted observations of this same f, using0.1×standard-Gaussian noise. The final five inputs are irrelevant.
- **Random projection:**four inputs and a4×8 weight matrix sampled from U(−1,1); eight outputs are the resulting overcomplete linear projection.
- Each has10,000 training and10,000 test samples, MSE fitting, RMSE evaluation and five seeds. SO/MO RMSE is0.1540/0.1429 for Friedman1 and0.0204/0.0180 for random projection. Curve comparisons use matched learning rate and maximum depth.

These are independent-sample regression tasks deliberately containing shared output structure. They do not contain forecast origins, overlapping horizon labels, solar-wind regimes or a tail-plus-spectral loss.

[Real-data experiments, TablesIV–V and SectionV-B](https://arxiv.org/pdf/1909.04373v2#page=9):

| Dataset | Train / test | Inputs / outputs | Task |
|---|---:|---:|---|
| MNIST | 50,000 /10,000 |784 /10 | Digit classification |
| Yeast |1,038 /446 |8 /10 | Protein localization classification |
| Caltech101 |6,073 /2,604 |324 /101 | Image classification using HOG |
| MNIST-inpainting |50,000 /10,000 |200 /24 | Predict a4×6 pixel patch from an upper10×20 patch |
| Student-por |454 /159 |41 /3 | Three grade scores, scaled to[−1,1] |
| NUS-WIDE |161,789 /107,859 |128 /81 | Image multi-label task using cVLAD+ |

The24-output inpainting case is relevant to output dimensionality, **not** evidence about24 forecast leads. MNIST, MNIST-inpainting and NUS-WIDE use official train/test splits; the others use70/30 random splits repeated ten times. Multiclass tasks use cross-entropy; regression and multi-label tasks use MSE. Regression is assessed with RMSE and classification with top-1 accuracy. No chronological multihorizon forecasting experiment is reported.

[AppendixB](https://arxiv.org/pdf/1909.04373v2#page=12) searches depth4–10 and learning rates{0.05,0.1,0.25,0.5}, fixes λ=1 and maximum leaves at0.75×2^depth, and reports different selected settings across methods. Training stops after25 rounds without performance improvement. The inspected text does not clearly identify a separate selection set for every tuning/stopping decision, so its protocol must not be imported as a compliant template for this repository.

## Complexity and XGBoost Compatibility

The paper's SO and MO are the authors' own C++ implementations, with the same surrounding implementation and different split-finding logic. They grow trees best-first; they are **not XGBoost2.1.4**. [SectionIII-E/F](https://arxiv.org/pdf/1909.04373v2#page=6).

TablesI–II give:

- Histogram split search: O(bmd) for both dense SO and MO; full-Hessian MO requires O(bmd³).
- Prediction comparisons: O(tdh) for SO versus O(th) for MO.
- Prediction additions: O(td) for both dense methods.

The claimed speed benefit comes partly from sharing routing/partition overhead, not eliminating all output-dependent work. Reported timing uses a XeonE5-2698v4, four threads for synthetic tasks and eight for real tasks; per-round timing averages ten rounds and repeats three times. Faster per-round training is not automatically faster convergence to a fixed predictive quality.

**Do not equate the paper's formulas or controls with the current library:**

1. The paper's main derivation uses L2-only, unclipped leaf weights. Exact XGBoost2.1.4 behavior also includes L1 thresholding, max_delta_step clipping and minimum-Hessian eligibility details documented in the earlier [mechanism evidence](output_sharing_mechanism_evidence.md).
2. The paper's equation gain includes a1/2 factor, while algorithmic scores omit common constants. Crucially, AppendixB says its gain threshold is compared with the **average gain over involved output variables**. The pinned XGBoost vector path compares a **summed** score with gamma. Equal numeric thresholds therefore do not imply equivalent regularization.
3. The paper searches different depths/rates and stops at different rounds; its result table is not a single fixed-depth, fixed-round topology-only comparison.
4. For the current one-booster,24-output custom objective, independent scalar partitions can retain full-matrix spectral gradient coupling. The paper's broad phrase “learned independently” must not be interpreted to mean that changing multi_strategy necessarily removes the current spectral loss.

## Current Probe Applicability

The parent reports local in-sample split-gain regret in18 frozen contexts. This task did not inspect or recompute probe outputs.

Those observations indicate a local surrogate partition tradeoff under the probe's frozen conditions. The paper supplies a reason **both** for potential fit improvement from separate partitions and for loss of useful regularization. It cannot establish the net forecast effect, generalization uncertainty, preferable capacity budget or appropriate baseline adoption.

The existing mechanism report already specifies required matched-input/loss observations, eligibility caveats and refutation conditions. This paper lookup neither changes those requirements nor selects a control. No new fit, validation split, threshold or tuning procedure is authorized by the literature.

## Version Note

Full-text receipt retrieved2026-11-09 local date. Extraction came from the original arXiv unversioned PDF URL, but the document explicitly identifies **v2,28December2019**. Citations above use that immutable version. The document is marked “UNDER REVIEW”; no journal acceptance or later revision is asserted.

## Source-Reference Evidence

- [Original versioned paper](https://arxiv.org/pdf/1909.04373v2) — derivation, experiments, complexity and limitations.
- [Full extracted receipt](output_sharing_paper/arxiv_1909_04373_v2_extracted.txt) —13 pages and AppendicesA–C, with version marker.
- [Structured evidence](output_sharing_primary_paper.json) — equation locations, exact task conditions, compatibility boundaries and source links.

PDF extraction breaks some mathematical notation across lines. The formulas above are reconstructed from numbered equations, definitions and accompanying algorithms, not from the abstract. No third-party interpretation is required.

## Reusable Takeaway

GBDT-MO supplies primary evidence for **a flexibility-versus-regularization tradeoff**. Independent partitions can improve local fit without improving held-out forecasts. The paper's averaged gain threshold and experimental settings differ materially from XGBoost2.1.4; neither its positive results nor the current training-only regret establishes solar-wind skill or justifies adopting a hypothesis.

