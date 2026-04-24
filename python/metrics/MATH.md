# MATH.md

## Canonical Mathematical Program for Mech-Lab

## 0. Purpose

This document is the canonical mathematical source of truth for the project.

It exists to answer five questions clearly:

1. What is the mathematics of the project?
2. Which mathematical objects are primary, and which are derived?
3. Where does each mathematical component live in the product and repo?
4. In what order should the mathematical program be developed?
5. Which claims are mathematical, which are empirical, and which are product or protocol claims?

This document is the math layer only.

It is **not** the implementation spec, CLI spec, cutover policy, evidence policy, or product docs.
Those documents may reference this one, but they do not redefine the mathematics.

---

## 1. Scope and boundaries

### In scope

- latent geometry
- sparse coding
- local charts and atlases
- block-level state bundles
- transport maps and connection-like structure
- topology over sparse codes and transport graphs
- higher-order feature interaction structure
- causal mechanism validation criteria
- mathematical metrics and objective functions
- mathematical evaluation criteria

### Out of scope

- package layout
- CLI commands
- schema versioning details
- MAIR mechanics except where mathematical objects are referenced
- serving/inference plumbing
- hardware and deployment choices
- migration/deprecation policy

---

## 2. The one-sentence mathematical thesis

Mech-Lab treats internal model computation not as a flat collection of neuron activations, but as a family of sparse, charted, transported, and optionally topologized coordinate systems over block-level state trajectories.

---

## 3. Primary mathematical objects

These are the objects the rest of the project should be built around.

### 3.1 State space

For a model with hidden-state dimension $d$, let

$$
X \subseteq \mathbb{R}^d
$$

be the ambient hidden-state space.

For Qwen-style block geometry, the atomic observed object is not a single hidden state but a **block trajectory**:

$$
B_{b,t} = (x^{(1)}_{b,t}, x^{(2)}_{b,t}, \dots, x^{(m)}_{b,t})
$$

where:
- $b$ indexes the block
- $t$ indexes token position or sequence step
- $m$ is the number of hook slots in the block template

For the current Qwen3.5-2B reference adapter:

$$
B_{b,t} = (x^{preD1}, x^{postD1}, x^{postD2}, x^{postD3}, x^{postAttn}, x^{blockOut})
$$

This is the primary observed mathematical object for block geometry.

### 3.2 Sparse code

A block trajectory is encoded into a sparse code

$$
z = (z^{global}, z^{local})
$$

where:
- $z^{global}$ is the block-global code
- $z^{local}$ is the hook-local or slot-local refinement code

The sparse code is the main coordinate object used for reconstruction, transport, topology, and evidence emission.

### 3.3 Chart assignment

Each encoded block is assigned to one or more local charts:

$$
\pi(z) \in \Delta^{K-1}
$$

where $K$ is the number of charts and $\Delta^{K-1}$ is the probability simplex.

This makes the latent representation an atlas rather than one global linear dictionary.

### 3.4 Decoder / reconstruction map

A decoder reconstructs target states from sparse codes:

$$
\hat{x}^{(j)} = D_j(z^{global}, z^{local}, \pi(z))
$$

for hook slot $j$.

### 3.5 Transport maps

A transport map moves code or state information from one local regime to another.

In the block setting the first important transports are intra-block:

$$
T_{j \to j+1}(z_j) \approx z_{j+1}
$$

In the chart setting, transition maps between overlapping charts are:

$$
\tau_{\alpha\beta}: U_\alpha \supseteq U_\alpha \cap U_\beta \to U_\beta
$$

### 3.6 Topological summaries

Given sparse codes, coactivation graphs, chart-overlap graphs, or transport-edge graphs, define topological summaries

$$
\mathcal{T}(G)
$$

which may include connected components, cycle structure, persistence summaries, Mapper summaries, or bifiltration-derived quantities.

### 3.7 Mechanism objects

A candidate mechanism is a structured tuple

$$
M = (F, E, T, I)
$$

where:
- $F$ = feature or code set
- $E$ = edges / relations
- $T$ = transport structure
- $I$ = intervention profile

A validated mechanism is a candidate mechanism that satisfies explicit causal and stability criteria.

---

## 4. Mathematical layers of the project

The project math should be organized into seven layers.

## Layer A. Sparse representation

### Main question
What sparse coordinate system best represents model-internal computation?

### Mathematical objects
- sparse code $z$
- encoder $E$
- decoder $D$
- reconstruction map
- sparsity penalty

### Core equations

$$
z = E(x)
$$

$$
\hat{x} = D(z)
$$

$$
L_{recon} = \|x - \hat{x}\|^2
$$

$$
L_{sparse} = \lambda \|z\|_1
$$

or top-k / JumpReLU style sparsity in implementation.

### Repo ownership
- `mech_lab.geometry.encoders`
- `mech_lab.geometry.runtime`

### Slice first touched
- `qwen35_block_metric`

---

## Layer B. Metric / local geometry

### Main question
How do latent distances and neighborhoods reflect hidden-state geometry?

### Mathematical objects
- local metric
- pullback geometry
- neighborhood graph
- distortion measure
- local tangent approximation

### Core equations
If $D$ is differentiable, define local pullback metric:

$$
g_z = J_D(z)^\top J_D(z)
$$

where $J_D(z)$ is the Jacobian of the decoder.

Metric-preservation objective:

$$
L_{metric} = \sum_{(i,j) \in \mathcal{N}} \left(d_X(x_i,x_j) - d_g(z_i,z_j)\right)^2
$$

Neighborhood-preservation proxy:

$$
L_{nbr} = \text{distortion of local kNN structure}
$$

### Repo ownership
- `mech_lab.geometry.metric`
- `mech_lab.geometry.eval`

### Slice first touched
- `qwen35_block_metric`

---

## Layer C. Atlas / chart structure

### Main question
How do we replace one global sparse codebook with multiple local coordinate charts?

### Mathematical objects
- chart family $\{U_\alpha\}$
- chart assignment $\pi(z)$
- overlap regions $U_\alpha \cap U_\beta$
- transition maps $\tau_{\alpha\beta}$

### Core equations
Chart-conditioned reconstruction:

$$
\hat{x} = \sum_{\alpha=1}^K \pi_\alpha(z) D_\alpha(z)
$$

Overlap consistency:

$$
L_{overlap} = \sum_{z \in U_\alpha \cap U_\beta} \|\tau_{\alpha\beta}(z_\alpha) - z_\beta\|^2
$$

Assignment constraint:

$$
\sum_{\alpha=1}^K \pi_\alpha(z) = 1
$$

### Repo ownership
- `mech_lab.geometry.atlas`
- `mech_lab.geometry.encoders`

### Slice first touched
- `qwen35_block_atlas`

---

## Layer D. Block geometry

### Main question
What is the right geometric object for hybrid recurrent/attention models?

### Mathematical objects
- block trajectory $B_{b,t}$
- hook-slot ordering
- slot-local delta
- block-global code
- hook-local code

### Core equations
State reconstruction:

$$
L_{state} = \sum_j \|x^{(j)} - \hat{x}^{(j)}\|^2
$$

Delta reconstruction:

$$
L_{delta} = \sum_j \|(x^{(j+1)} - x^{(j)}) - (\hat{x}^{(j+1)} - \hat{x}^{(j)})\|^2
$$

This is the mathematical core of block-level geometry.

### Repo ownership
- `mech_lab.geometry.encoders`
- `mech_lab.adapters.qwen35_2b`
- `mech_lab.protocol` for the corresponding IR objects

### Slice first touched
- `qwen35_block_metric`

---

## Layer E. Transport / connection structure

### Main question
How does structure move through blocks, slots, layers, and charts?

### Mathematical objects
- intra-block transport
- inter-chart transition
- inter-layer transport
- tract transport
- bridge transport

### Core equations
Intra-block code transport:

$$
T_{j \to j+1}(z_j) \approx z_{j+1}
$$

Transport residual:

$$
L_{trans} = \sum_j \|T_{j \to j+1}(z_j) - z_{j+1}\|^2
$$

Metric-compatibility aspiration:

$$
\langle T v, T w \rangle_{g_{j+1}} \approx \langle v, w \rangle_{g_j}
$$

### Special block-level split
For hybrid models, transport is split into:
- **tract transport**: recurrent / DeltaNet-side state evolution
- **bridge transport**: attention-mediated transfer

This split is mathematically important and should not be collapsed into one undifferentiated transport notion.

### Repo ownership
- `mech_lab.geometry.transport`
- `mech_lab.protocol` for transport artifacts

### Slice first touched
- `qwen35_block_transport`

---

## Layer F. Topological sidecar

### Main question
What multiscale structure exists over sparse codes, overlaps, and transport graphs?

### Mathematical objects
- coactivation graph
- chart-overlap graph
- transport-edge graph
- persistence summaries
- Mapper summaries
- bifiltration over scale and intervention

### Core quantities
Topological susceptibility:

$$
\chi = \text{change in topological statistic as a function of intervention strength and geometric scale}
$$

Bridge dependence:

$$
\beta = \text{fraction of effective predictive or attribution flow that relies on bridge edges rather than tract structure}
$$

### Important status
This layer is mathematically important but **provisional** in protocol status during the first product cutover.

### Repo ownership
- `mech_lab.geometry.topology`
- `mech_lab.trace.sidecar`

### Slice first touched
- `qwen35_block_topology`

---

## Layer G. Higher-order interactions and causal mechanisms

### Main question
When is a sparse code family merely descriptive, and when does it define a real mechanism?

### Mathematical objects
- higher-order decoder terms
- hyperedges / conjunctions
- synergy scores
- intervention families
- validated mechanism hyperpaths

### Core equations
Interaction-aware decoding schematic:

$$
\hat{x} = D_1(z) + D_2(z \otimes z) + \cdots
$$

Synergy-style score, schematic:

$$
\mathrm{Syn}(S) = \Delta(S) - \sum_{i \in S} \Delta(\{i\})
$$

This layer is where candidate mechanisms become validated mechanism objects.

### Repo ownership
- later `mech_lab.geometry.hypergraph`
- later `mech_lab.geometry.causal`

### Slice status
Deferred until after capture / metric / atlas / transport are stable.

---

## 5. Canonical development order

This is the canonical mathematical sequence.

### Stage 1. Sparse block reconstruction
Goal: learn sparse block-global and hook-local codes that reconstruct states and deltas.

### Stage 2. Metric-aware block encoding
Goal: preserve local neighborhoods and reduce distortion in latent geometry.

### Stage 3. Atlas construction
Goal: replace one global codebook with local charts and overlap consistency.

### Stage 4. Transport structure
Goal: estimate tract and bridge transport maps and evaluate transport residuals.

### Stage 5. Topological summaries
Goal: compute optional multiscale summaries over sparse-code-derived graphs.

### Stage 6. Higher-order interactions
Goal: move beyond additive decoding and represent conjunctive structure.

### Stage 7. Causal mechanism validation
Goal: elevate candidate features/paths into mechanism claims using interventions and stability criteria.

This is the mathematical order even when implementation work temporarily proceeds in different local increments.

---

## 6. Canonical timeline for the PhD-facing math program

This is the cleanest PhD-oriented timeline.

### Paper / chapter 1 — Metric-aware sparse representation
Focus:
- sparse codes
- decoder-induced geometry
- neighborhood preservation
- distortion metrics

Output:
- formalism for metric-aware sparse encoding
- baseline experiments and ablations

### Paper / chapter 2 — Atlas sparse autoencoding
Focus:
- local charts
- chart assignment
- overlap consistency
- local-to-global representational stability

Output:
- atlas formalism
- multi-chart experiments

### Paper / chapter 3 — Block geometry for hybrid architectures
Focus:
- block trajectory as primary object
- state + delta reconstruction
- slot-aware geometry
- recurrent tract vs attention bridge distinction

Output:
- block-bundle geometry paper anchored on Qwen-style hybrid blocks

### Paper / chapter 4 — Transport and connection structure
Focus:
- intra-block transport
- inter-chart transitions
- metric-compatible transport
- tract/bridge transport split

Output:
- transport-aware block geometry paper

### Paper / chapter 5 — Topology sidecar
Focus:
- coactivation topology
- chart overlap topology
- transport-edge topology
- susceptibility and bridge-dependence metrics

Output:
- TDA sidecar paper

### Paper / chapter 6 — Higher-order interactions
Focus:
- nonlinear decoding
- hyperedges
- conjunction structure
- interaction-aware representations

Output:
- higher-order mechanism representation paper

### Paper / chapter 7 — Causal mechanism admission
Focus:
- intervention protocols
- synergy / irreducibility
- validated hyperpaths
- admission criteria for mechanism claims

Output:
- causal validation paper

---

## 7. What lives where

This section is the direct “what and where” map.

| Mathematical topic | Canonical home in math program | First implementation home | First slice | Protocol artifact |
|---|---|---|---|---|
| Sparse coding | Layer A | `mech_lab.geometry.encoders` | `qwen35_block_metric` | `block_code_batch.v1` |
| Metric geometry | Layer B | `mech_lab.geometry.metric` | `qwen35_block_metric` | `block_code_batch.v1` + eval outputs |
| Atlas / charts | Layer C | `mech_lab.geometry.atlas` | `qwen35_block_atlas` | `block_chart_bundle.v1` |
| Block trajectory geometry | Layer D | `mech_lab.geometry.encoders` + adapter | `qwen35_block_metric` | `block_state_batch.v1` |
| Tract / bridge transport | Layer E | `mech_lab.geometry.transport` | `qwen35_block_transport` | `block_transport_bundle.v1` |
| Topology sidecar | Layer F | `mech_lab.geometry.topology` / `trace.sidecar` | `qwen35_block_topology` | provisional `block_topology_sidecar.v1` |
| Interaction / hypergraph math | Layer G | later `mech_lab.geometry.hypergraph` | deferred | later artifact |
| Causal validation math | Layer G | later `mech_lab.geometry.causal` | deferred | later artifact |

---

## 8. Canonical definitions that should stay stable

These terms should have one mathematical meaning across the project.

### Block trajectory
Ordered tuple of slot-indexed hidden states captured from one block for one token position.

### Block-global code
Sparse latent component shared across the whole block trajectory.

### Hook-local code
Sparse latent component specific to a hook slot or local transition.

### Chart
Local coordinate regime in latent space with its own decoder behavior.

### Overlap consistency
Agreement condition between multiple charts on shared regions.

### Tract transport
Transport associated with recurrent/linear-attention-style state evolution inside the tract portion of a hybrid block.

### Bridge transport
Transport associated with the periodic attention bridge inside a hybrid block.

### Topological susceptibility
Sensitivity of a chosen topological summary to intervention strength across geometric scales.

### Bridge dependence
Fraction of effective predictive or attribution flow that relies on bridge edges rather than tract structure.

### Mechanism hyperpath
A validated higher-order path through feature, transport, and interaction structure that survives intervention tests.

---

## 9. Mathematical claims hierarchy

This is how claims should be separated.

### Type I — definitional claims
Examples:
- what a block trajectory is
- what tract transport means
- what a chart assignment is

These belong here in `MATH.md`.

### Type II — theorem-like or formal claims
Examples:
- invariance properties
- consistency conditions
- equivalence / non-equivalence of constructions
- sufficient conditions for stable overlap alignment

These belong in math notes or appendices that hang off this document.

### Type III — empirical claims
Examples:
- Qwen3.5-2B achieves error threshold X
- chart overlap loss converges below Y
- topology metric tracks intervention strength on benchmark Z

These belong in experiment reports, not in the canonical math document.

### Type IV — product claims
Examples:
- artifact emitted
- CLI supported
- schema version frozen
- evidence bundle resolves successfully

These belong in cutover / product / protocol docs, not here.

---

## 10. Required companion math files

`MATH.md` should stay the top-level canonical map, but the real math program will likely need companion files.

Recommended companion set:

- `math/01_sparse_representation.md`
- `math/02_metric_geometry.md`
- `math/03_atlas_geometry.md`
- `math/04_block_geometry.md`
- `math/05_transport.md`
- `math/06_topology_sidecar.md`
- `math/07_higher_order_interactions.md`
- `math/08_causal_validation.md`
- `math/appendix_notation.md`
- `math/appendix_open_problems.md`

Rule:
- `MATH.md` = canonical map and ordering
- companion files = full derivations, definitions, experiments-to-math alignment, open problems

---

## 11. Minimum notation appendix

Recommended notation to keep stable:

- $x$: hidden state
- $B_{b,t}$: block trajectory
- $z$: sparse code
- $z^{global}$: block-global sparse code
- $z^{local}$: hook-local sparse code
- $D$: decoder
- $E$: encoder
- $\pi$: chart assignment distribution
- $U_\alpha$: chart domain
- $\tau_{\alpha\beta}$: chart transition map
- $T$: transport map
- $g$: latent metric
- $\chi$: topological susceptibility
- $\beta$: bridge dependence

---

## 12. Immediate next writing move

Write this document first, then write companion files in this order:

1. `math/04_block_geometry.md`
2. `math/02_metric_geometry.md`
3. `math/03_atlas_geometry.md`
4. `math/05_transport.md`
5. `math/06_topology_sidecar.md`
6. `math/07_higher_order_interactions.md`
7. `math/08_causal_validation.md`

That order matches the actual center of gravity of the project:
- block geometry first
- then metric/atlas/transport
- then topology
- then higher-order and causal work

---

## 13. Final rule

If a future document changes the mathematics, it must update `MATH.md`.
If a future document only changes implementation, evidence, product surface, or repo structure, it must not redefine the mathematics here.
