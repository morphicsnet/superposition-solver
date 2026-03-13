# SHG Blueprint (Reference)

Note: This blueprint is conceptual. For implementation details, see ARCHITECTURE.md and API_SPEC.md.

The Transition to Neuro-Symbolic
Topology: A Blueprint for H-SNN/SHG
Interpretability Architectures
1. The Epistemic Crisis and the Limits of Linear
Feature Extraction
The contemporary landscape of artificial intelligence is defined by a paradox of scale: as model
capabilities expand exponentially, our ability to interpret their internal mechanisms diminishes at
a commensurate rate. We stand at the precipice of an "epistemic crisis,
" where the most
powerful cognitive engines in human history—Large Language Models (LLMs) and deep
generative systems—function as opaque black boxes, their reasoning obscured by billions of
high-dimensional parameters. While the industry has coalesced around Sparse Autoencoders
(SAEs) as the standard countermeasure to this opacity, attempting to decompose dense
representations into "monosemantic" features, this report argues that SAEs represent a
fundamental theoretical dead-end. They are palliative care for a structural disease.
The prevailing interpretability pipeline relies on a linear algebraic assumption: that a model's
"thought" is the additive sum of independent feature vectors. This assumption, codified in
techniques like SHAP (SHapley Additive exPlanations) and the current generation of SAEs, fails
to capture the non-linear, compositional nature of Transformer architectures. Language
understanding does not emerge from the accumulation of isolated concepts but from "meetings"
of features—complex, high-order interactions where the combination of tokens produces
meaning irreducible to their individual parts. A negation, an idiom, or a sarcastic inflection
resides entirely in the structural interaction, a reality that pairwise correlations and linear basis
vectors fundamentally fail to map.
To resolve this, we must dismantle the linear interpretability stack and replace it with a
Hypergraph Spiking Neural Network (H-SNN) / Symbolic Hypergraph (SHG) Ensemble.
This is not a superficial substitution of terminology but a complete architectural redesign. We
propose moving from the static analysis of activation magnitudes to the dynamic analysis of
temporal spike coincidences, and from the dyadic simplicity of graphs to the polyadic complexity
of hypergraphs. This report serves as the comprehensive "Replacement Plan,
" detailing how to
adapt the hardware specifications of the Neuro-Symbolic Interpretability Co-Processor
(NSI-CP) to support this new paradigm, thereby eliminating superposition rather than merely
mitigating it.
1.1 The Failure of the Linear Assumption
The "additivity fallacy" is the central obstacle to true mechanistic interpretability. Current
approaches, including standard SAEs, assign scalar importance values to input tokens or latent
features, implicitly modeling the output Y as Y = \sum \phi
i x
_
_
i. This ignores the supermodular
nature of neural computation. In a Transformer, the attention mechanism is inherently tri-partite
(Query, Key, Value), and the MLP layers act as logic gates (AND/OR/XOR). These are not
additive operations; they are functional computations where a specific set of inputs \{x
1, \dots,
_
x
_
k\} jointly determines an output.
When we force these interactions into a linear framework—or even a standard graph G=(V,E)
where edges connect exactly two nodes—we lose the "group effect"
. A polysemantic neuron
that activates only upon the conjunction of "Eiffel,
" "Tower,
" and "Context: Geography" is
decomposed by a standard graph into separate edges (Eiffel \to Neuron, Tower \to Neuron).
This results in representational collapse. The "Eiffel" edge appears to have a probabilistic
weight, as does the "Tower" edge, but the necessity of their conjunction is lost. The map
becomes a cloud of correlations rather than a circuit of causes.
1.2 The Promise of H-SNN/SHG Ensembles
The proposed architecture pivots to a Hypergraph Formalism H=(V,E), where a hyperedge e
\in E acts as a subset e \subseteq V with |e| \geq 2. This allows the hardware to explicitly model
the "meetings" of features. However, simply using hypergraphs on top of standard SAE nodes is
insufficient if the nodes themselves remain polluted by superposition.
The "Replacement Plan" introduces the Orthogonal Ensemble. Instead of relying on a single
SAE to extract features, we deploy an ensemble of Spiking Neural Networks (H-SNNs), each
initialized with disparate structural biases and sparsity constraints. The core insight is that while
a single neuron in one observer may be superposed (representing "Bank" as both "River" and
"Finance"), it is statistically impossible for the same superposition to exist across a diverse
ensemble of independent observers. By defining the "true" feature as the Intersection of
Hyperedges across the ensemble, we eliminate superposition entirely. The feature "River Bank"
is no longer a vector direction; it is a topological invariant—a reliable subgraph that persists
across the ensemble's differing perspectives.
Furthermore, the transition to Spike-Native Analysis leverages time as a disentangling
dimension. In a standard activation vector, multiple signals are compressed into a single scalar
magnitude. In a spike train, signals can be phase-coded. Two concepts sharing the same
neuron can be distinguished by their temporal firing patterns relative to a global clock. The
NSI-CP, originally designed for graph streaming, is perfectly adapted to function as a massive
coincidence detector for these temporal events.
2. Theoretical Foundations: The Hypergraph Physics
of Intelligence
To execute this replacement plan, we must first establish the rigorous theoretical physics that
governs the H-SNN/SHG architecture. The shift is from vector spaces to topological spaces, and
from calculus to combinatorial game theory.
2.1 Hypergraph Topology vs. Graph Theory
Traditional graph theory has long served as the default abstraction for neural networks, yet it
introduces critical information loss in post-hoc explanation. A standard directed graph restricts
interaction to dyadic pairs. In the context of the NSI-CP, we adopt the Hypergraph definition
where edges are arbitrary subsets. This is crucial for modeling the Attention Mechanism of
Transformers. Attention is not a link from token A to token B; it is a tri-partite interaction
involving a Query, a Key, and a Value. In the H-SNN/SHG framework, we model this as a
hyperedge \{Q, K, V, \text{Head}, \text{Output}\}, capturing the multi-way dependency that
standard graphs obscure.
This formalism aligns with the logic of Boolean satisfiability. If a model predicts "Paris" only
when "France" AND "Capital" are present, a graph representation would show two edges
flowing into "Paris.
" A hypergraph representation shows a single hyperedge encompassing
\{France, Capital, Paris\}. This distinction is mathematically profound. In the graph view,
removing "France" leaves the edge from "Capital" intact, implying "Paris" might still be predicted
(albeit with lower probability). In the hypergraph view, removing "France" invalidates the entire
hyperedge, correctly predicting the collapse of the output. This is the difference between
correlation and causation.
2.2 The Game-Theoretic Metric: STII
How do we identify which hyperedges exist in the vast combinatorial space of the ensemble?
We rely on the Shapley-Taylor Interaction Index (STII). Unlike the classical Shapley value,
which "flattens" interaction effects by splitting credit among participants, STII decomposes the
model function into main effects (k=1), pairwise interactions (k=2), and higher-order terms.
The STII satisfies the Interaction Distribution Axiom, ensuring interactions are allocated to the
minimal subset generating them.
●
Positive STII (I
_
S > 0): Indicates Synergy. The presence of the group creates more
activation than the sum of the individuals. This maps to excitatory hyperedges in our
SHG.
●
Negative STII (I
_
S < 0): Indicates Interference or Redundancy. The group achieves
less than the sum. This maps to inhibitory hyperedges or "backup circuits" where features
are mutually exclusive.
The calculation of STII is computationally prohibitive (O(2^N)) for software-only solutions. This is
the primary justification for the NSI-CP hardware. The "Archipelago" framework implemented on
the chip decouples detection from estimation, using heuristic "islands" to constrain the search
space before applying the rigorous STII calculus.
2.3 Temporal Disentanglement in SNNs
The move to Spiking Neural Networks (SNNs) introduces a temporal dimension to
interpretability. In a continuous-valued network (like a standard SAE), superposition is a spatial
problem: two vectors sum to a third vector that lies in the same space. Disentangling them
requires finding a projection that separates them.
In an SNN, activation is discrete. A neuron fires at t
1, t
2, t
_
_
_
3. Superposition in an SNN would
require two concepts to trigger the same neuron at the exact same microsecond consistently. By
enforcing Temporal Sparsity and using phase-coding (where the timing of the spike relative to
a cycle encodes information), we effectively expand the dimensionality of the representation
space by the clock frequency of the chip. The "meetings" of features become Temporal
Coincidences. If Feature A and Feature B are truly interacting to cause Output C, their spikes
must arrive at the destination within a specific causal window \Delta t. If they are merely
superposed on the same hardware but functionally unrelated, their firing times will likely be
uncorrelated or phase-shifted. The NSI-CP's Graph Streaming Engine (GSE) is repurposed in
this plan to detect these temporal coincidences.
3. Hardware Architecture: The Neuro-Symbolic
Interpretability Co-Processor (NSI-CP)
The Neuro-Symbolic Interpretability Co-Processor (NSI-CP) is the physical substrate for the
H-SNN/SHG architecture. It is a heterogeneous System-on-Chip (SoC) designed to bridge the
gap between high-velocity neural events and symbolic reasoning. While originally specified for
SAEs, its modular design is perfectly suited for the "Replacement Plan.
"
3.1 Global Architecture and Interconnect
The SoC comprises five primary modules connected via a high-bandwidth Network-on-Chip
(NoC) utilizing AMBA AXI4 protocols.
●
Data Plane (AXI4-Stream): Handles the bulk flow of spike events and hyperpaths. This
facilitates a continuous processing pipeline where data flows from ingestion to symbolic
rule generation without CPU intervention.
●
Control Plane (AXI4-Lite): Allows the Meta-Controller (MC) to configure thresholds,
window sizes, and SNN parameters dynamically.
The NSI-CP functions as a "Sidecar" to the host AI system (e.g., an H100 cluster), tapping into
the activation stream via CXL or PCIe Gen5 to build the interpretability graph in real-time
without blocking the critical path of the host model.
3.2 The Graph Streaming Engine (GSE): From Vector Ingestion to
Spike Coincidence
Original Role: Ingest SAE features and perform Temporal Neighbor Search to find co-occurring
activations. Adapted Role (H-SNN): Spatiotemporal Coincidence Detector.
The GSE serves as the sensory frontend. In the H-SNN architecture, it ingests Address Event
Representation (AER) packets from the ensemble. Each packet contains {Ensemble
ID,
_
Neuron
_
ID, Timestamp}.
●
Mechanism: The GSE maintains a sliding window buffer, representing the causal
integration time of the biological metaphor (e.g., 20ms).
●
Island Formation: When a spike arrives, the GSE queries its buffer for other spikes that
fall within the valid \Delta t. Unlike a standard k-NN search which looks for vector
similarity, this search looks for Temporal Proximity.
●
Filtering: The GSE applies a "Cross-Ensemble Constraint.
" A valid candidate hyperedge
must include spikes from different members of the H-SNN ensemble. This enforces the
orthogonality requirement: we only care about patterns that are robust across different
structural observers.
●
Output: The GSE generates ADD
CANDIDATE
_
_
HYPEREDGE requests for the memory
fabric. If the memory is saturated, the GSE uses TVALID/TREADY handshaking to apply
backpressure, ensuring no causal events are dropped during bursty periods.
3.3 The Graph Memory Fabric (GMF): The Distributed Hypergraph
Store
Original Role: Processing-in-Memory (PIM) storage for the dynamic graph. Adapted Role
(H-SNN): The Symbolic Hypergraph (SHG) State.
The GMF solves the "memory wall" by executing graph updates in situ using ReRAM or
high-density SRAM.
●
Data Structure: The GMF stores the Dynamic Causal Hypergraph (DCH). Nodes are
not raw neurons but (Ensemble, Feature) tuples. Hyperedges represent the verified
interactions.
●
Metadata: Each hyperedge stores its STII Weight (causal strength) and a Reliability
Score (frequency of observation).
●
Atomic Updates: The PIM architecture allows the chip to increment the reliability score of
a hyperedge or update its STII weight without moving the adjacency list to a central CPU.
This is critical for handling the millions of updates per second generated by the H-SNN
ensemble.
●
Distributed Storage: The hypergraph is partitioned across multiple GMF banks, allowing
for massive parallelism. Multiple hyperedges can be updated simultaneously, mirroring the
distributed nature of the neural network itself.
3.4 The Parallel Traversal Accelerator (PTA): The Causal Engine
Original Role: Massive parallel random walks for causal tracing and ACDC. Adapted Role
(H-SNN): Counterfactual Simulation Engine.
The PTA is the "reasoning" core. It verifies the candidates proposed by the GSE.
●
Walkers: The PTA launches thousands of lightweight "walkers" (traversal threads). In the
H-SNN regime, these walkers follow Spike Propagation Paths.
●
Temporal Logic: The walkers enforce strict causality: a cause must precede its effect. A
walker at Node B (t=10) can only step to Node A if Node A fired at t < 10.
●
Archipelago Estimation: The PTA executes the rigorous STII calculation. For a
candidate "island" (hyperedge), the PTA coordinates with the host to perform "virtual
lesions"
—masking specific spikes and measuring the divergence in the host model's
output. The PTA accumulates these deltas to compute the Shapley value.
●
ACDC Pruning: The PTA implements Automated Circuit Discovery. It iteratively tests
hyperedges. If masking a hyperedge results in negligible KL divergence (output
unchanged), the PTA marks the edge for removal in the GMF. This distills the dense web
of coincidences into a sparse, necessary circuit.
3.5 The Frequent Subgraph Miner (FSM): The Symbolic Bridge
Original Role: Mining motifs and generating symbolic rules. Adapted Role (H-SNN):
Cross-Ensemble Motif Discovery.
The FSM bridges the sub-symbolic world of spikes and the symbolic world of rules.
●
Pipeline: It uses a deep pipeline to process one hyperpath per clock cycle (10^9
patterns/sec).
●
Canonical Labeling: To count patterns, the FSM must recognize structural isomorphisms
(e.g., recognizing that "A+B->C" is the same pattern regardless of node IDs). It uses a
hardware-accelerated DFS Code algorithm to generate a unique "Canonical Label" for
each graph structure.
●
CAM Acceleration: A Content-Addressable Memory (CAM) array stores the counts of
these canonical labels. This allows O(1) latency for pattern recognition.
●
Rule Extraction: When a pattern's frequency exceeds a threshold (configured by the
MC), it is promoted to a Rule. In the H-SNN context, these rules represent "Universal
Motifs"
—reasoning structures (like Induction Heads) that appear consistently across the
entire ensemble.
4. The Replacement Plan: Step-by-Step
Implementation
This section details the operational roadmap for replacing a standard SAE-based pipeline with
the H-SNN/SHG Ensemble architecture.
Phase 1: The Orthogonal Ensemble Encoder
Objective: Replace the single SAE with a diversity of Spiking Neural Networks.
1. Ensemble Initialization: Instead of training one SAE with dimension d
_{SAE} = 32 \times
d
_{model}, we instantiate K distinct SNN encoders (SNN
1 \dots SNN
_
_
K).
○
Heterogeneity: Each SNN is initialized with different random seeds, different
sparsity penalties (L
1 vs L
_
_
0), and potentially different neuron models (LIF vs.
Izhikevich) to induce structural diversity.
○
Goal: We want the "errors" (superpositions) of each SNN to be uncorrelated. If
SNN
_
1 confuses "Dog" and "Finance,
" SNN
2 must not make that same confusion.
_
2. Rate-to-Time Transcoding: The host LLM outputs continuous activation vectors (e.g.,
GELU outputs). The NSI-CP input interface must transcode these into spikes.
○
Mechanism: We use a Latency-Phase Code. The magnitude of the activation
determines the time of the spike within a fixed window \Delta T.
○
Formula: t
_{spike} = T
_{start} + (1 - \text{sigmoid}(activation)) \times \Delta T.
○
Result: High activations spike early in the window; low activations spike late or not
at all. This preserves magnitude information in the temporal domain, which the GSE
is optimized to process.
3. Stream Aggregation: The spike trains from all K encoders are multiplexed into a single
AXI4-Stream. The payload is expanded to 128 bits:
``
.
Phase 2: The Archipelago Protocol (Hypergraph Construction)
Objective: Construct the "Mental Model" in the GMF using the Archipelago framework.
1. GSE Configuration (Detection Phase): The Meta-Controller configures the GSE's
sliding window \tau to match the encoding window \Delta T.
○
Coincidence Logic: The GSE buffers incoming spikes. For every spike s
i, it
_
scans the buffer for s
_j such that |t
i - t
_
_j| < \tau.
○
Cross-Validation: The GSE filters pairs where s
i.EnsembleID == s
_
_j.EnsembleID.
We only want "meetings" between different observers. This is the primary filter for
superposition.
2. Island Formation: Groups of coincident spikes \{s
1, s
2, \dots, s
_
_
_
m\} form a candidate
"Island.
" The GSE constructs a hyperedge request containing these IDs and dispatches it
to the GMF.
3. GMF State Update: The GMF receives the request.
○
Lookup: It hashes the sorted list of IDs to find if this hyperedge already exists.
○
Update: If it exists, the GMF increments its observation
count. If not, it allocates a_
○
new entry.
Backpressure: If the GMF write queue is full, it asserts TREADY=0 to the GSE.
The GSE buffers events. If the buffer fills, the NSI-CP asserts flow control to the
host via CXL, slightly pausing the LLM inference to ensure interpretability fidelity
(no dropped frames).
Phase 3: Causal Verification (The STII Engine)
Objective: Eliminate spurious correlations and measure true causal weight.
1. Task Scheduling: The GMF flags "High-Frequency Islands" (candidates that appear
often but haven't been verified). The Meta-Controller assigns these to the PTA.
2. Perturbation Masking: The PTA generates a Perturbation Schedule. To calculate the
STII of island S=\{A, B, C\}, the PTA needs the model's output for subsets \emptyset, \{A\},
\{B\} \dots \{A,B,C\}.
○
Semantic Caching: Before running the expensive forward pass on the host, the
PTA queries the Redis Semantic Cache. If a similar prompt with the same mask
was recently evaluated, the result is retrieved, saving GPU cycles.
3. STII Calculation: If not cached, the PTA instructs the host to run the masked forward
pass. The host returns the logit change.
○
Computation: The PTA PEs (Processing Elements) compute the alternating sum of
marginal contributions (the STII formula).
○
Verdict:
■ I
DELETE.
■ I
VERIFY and store
4. _
S \approx 0: The hyperedge is noise. Action: GMF
_
_
S > 0: The hyperedge is a true synergy. Action: GMF
_
weight.
ACDC Pruning: Simultaneously, the PTA runs the ACDC algorithm. It traverses the
verified graph and attempts to prune edges that are causally redundant even if they have
high correlation. This ensures the final graph is the minimal circuit required to explain the
behavior.
Phase 4: Symbolic Extraction (The Motif Miner)
Objective: Generate human-readable rules from the verified circuits.
1. Hyperpath Streaming: The PTA streams sequences of verified hyperedges (e.g.,
"Token
A -> Hyperedge 1 -> Hyperedge 2 -> Output") to the FSM.
2. Canonical Labeling: The FSM's Stage 1 pipeline applies the DFS Code algorithm. It
explores the graph structure, generating 5-tuples (i, j, l
i, l
e, l
_
_
_j) and selecting the
lexicographically smallest sequence as the ID. This ensures that "Induction Head A" and
"Induction Head B" are recognized as instances of the same class of mechanism.
3. Rule Promotion: The FSM Stage 2 (CAM) counts these IDs. When the count exceeds
the threshold, Stage 3 generates a Rule Notification.
○
Example Rule: "IF (Ensemble
1:Feat
X AND Ensemble
2:Feat
_
_
_
_
Y) THEN
(Induction
Head
_
_
Z ACTIVATE).
"
4. HIF Serialization: The Meta-Controller serializes these rules and the underlying
hypergraph into the Hypergraph Interchange Format (HIF) JSON. This file is pushed to
the external analyst dashboard.
5. Comparative Analysis: Linear SAE vs. H-SNN/SHG
The following comparisons illustrate the profound differences between the legacy architecture
and the proposed H-SNN replacement.
5.1 The Resolution of Polysemanticity
Feature Legacy SAE Pipeline H-SNN/SHG Ensemble
Representation Single Vector Basis Intersection of K Ensembles
Superposition Mitigated (Expanded Dims) Eliminated (Topological
Intersection)
Ambiguity "Ghost" Activations Filtered by Ensemble
Consensus
Mechanism Spatial Projection Spatiotemporal Coincidence
In the SAE model, a polysemantic neuron is a noisy channel. In the H-SNN, because we require
consensus across structurally diverse observers (the ensemble), the noise cancels out. The
probability that K independent networks superpose the same two unrelated concepts on the
same feature ID is vanishingly small. The intersection yields the pure concept.
5.2 Interaction Modeling
Feature Graph (G) Hypergraph (H)
Edge Type Dyadic (u,v) Polyadic e \subseteq V
Logic Additive (u+v \to y) Supermodular (Synergy)
Attention Split (Q\to A, K\to A) Unified (Q,K,V \to Output)
Metric Weight/SHAP STII (Interaction Index)
The H-SNN allows us to model the logic gates of the network. An MLP neuron acting as an AND
gate is a single hyperedge. In a graph, it is a set of incoming edges that lose their "AND-ness.
"
5.3 Computational Complexity and Hardware
Feature Software (GPU) NSI-CP (Hardware)
Search Space O(2^N) (Intractable) O(2^k) (Archipelago PIM)
Update Latency Batch (Seconds) Real-Time (Nanoseconds)
Pattern Mining O(N^2) (gSpan CPU) O(1) (FSM CAM)
Memory Access Bus Bottleneck Processing-in-Memory
The NSI-CP is essential. The "Replacement Plan" requires calculating STII and mining frequent
subgraphs at the speed of thought. The FSM's ability to process 10^9 patterns/second allows
the interpretability layer to keep pace with the inference layer, something no GPU kernel can
achieve due to the irregular memory access patterns of graph traversal.
6. Operational Workflows and Case Studies
The H-SNN/SHG architecture enables new categories of safety checks.
6.1 Real-Time Hallucination Detection (The Geometric Truth Test)
Problem: Detecting when an LLM confabulates facts. H-SNN Solution: Topological
Connectedness.
1. Ingest: The GSE monitors the stream. The model generates "The Eiffel Tower is in
Berlin.
"
2. 3. Trace: The PTA triggers causal tracing from the "Berlin" token.
Topology Check: The walkers attempt to find a high-STII hyperpath connecting "Berlin"
to the context "Eiffel Tower.
"
4. Verdict: In a truthful generation, there is a strong causal chain. In a hallucination, the
"Berlin" token is generated by low-probability associations (weak edges). The hypergraph
reveals a Disconnected Component or a "Bridge" with negligible weight.
5. Action: The MC flags the generation as "Ungrounded" in real-time.
6.2 Automated Circuit Discovery (ACDC) for Bias
Problem: Detecting if a model uses a protected attribute (e.g., Gender) to make a decision
(e.g., Loan Approval). H-SNN Solution: Causal Circuit Mapping.
1. 2. 3. 4. Define Target: Output node "Loan Denied.
"
Run ACDC: The PTA runs the pruning algorithm, removing all non-essential hyperedges.
Inspect: Does the remaining minimal circuit include the node "Feature: Gender"?
Result: If yes, the model is definitively biased. This is not a correlation check; it is a
causal proof. The STII value quantifies exactly how much the gender feature contributed
to the denial.
7. Conclusion: Mapping the Messy City
The transition to the Hypergraph Spiking Neural Network (H-SNN) architecture represents the
maturation of AI interpretability from a soft science of correlations to a hard science of topology
and causality. We are moving beyond the "linear gaze"
—the limitation of seeing neural networks
as stacks of matrices—to perceiving them as "messy cities" of interacting agents.
This replacement plan provides the necessary blueprint. By adopting the Neuro-Symbolic
Interpretability Co-Processor (NSI-CP), we gain the physical infrastructure to handle the
combinatorial explosion of the Hypergraph Interpretability Framework. The integration of the
Graph Streaming Engine for temporal coincidence, the Graph Memory Fabric for distributed
state, the Parallel Traversal Accelerator for causal verification, and the Frequent Subgraph
Miner for symbolic rule extraction creates a unified pipeline.
This pipeline does more than visualize; it verifies. It eliminates superposition through the
orthogonality of ensembles. It measures synergy through the rigors of STII. And it maps the
emergent reasoning of the machine not as a list of features, but as a dynamic, navigable
hypergraph. As we entrust more of our world to opaque algorithmic decision-makers, this
architecture offers the essential cartography required to ensure those decisions remain
transparent, logical, and aligned with human intent.
8. Detailed Technical Addendum: Hardware
Specifications & Data Structures
To facilitate the immediate engineering of this system, we provide the granular specifications
derived from the NSI-CP documentation.
8.1 HIF (Hypergraph Interchange Format) Specification
The output of the NSI-CP follows the HIF standard.
●
JSON Structure:
{
"network-type": "directed-hypergraph"
"nodes":,
"edges":,
"incidences":
,
}
This format allows interoperability with HyperNetX (HNX) and Cytoscape.js.
8.2 FSM Hash Table Configuration
For FPGA prototyping (before ASIC tape-out), the FSM's CAM is replaced by BRAM-based
Hash Tables.
●
Strategy: Cuckoo Hashing to resolve collisions and guarantee constant-time lookup for
the vast majority of access patterns.
●
Cache: A small L1 CAM "Hot Cache" handles the top 1% of most frequent patterns to
maintain throughput.
●
Capacity: 128MB On-Chip BRAM allows tracking ~4 Million unique patterns
simultaneously.
8.3 Power Estimates
●
●
●
FSM Module: ~6 Watts (ASIC).
Total SoC: ~115 mm^2 area on 7nm node.
Efficiency: By offloading the irregular graph workload, the NSI-CP saves the host H100
from pipeline stalls, effectively increasing the system-level efficiency of the inference
cluster.
The "Replacement Plan" is complete. The theoretical derivation is sound, the hardware
specification is defined, and the operational workflow is established. It is now a matter of