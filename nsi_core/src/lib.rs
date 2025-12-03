//! NSI Core Library: minimal skeleton with encoders, spike encoding, hypergraph, metrics.

pub mod ensemble;
pub mod encoding;
pub mod hypergraph;
pub mod metrics;

pub use ensemble::{Encoder, Ensemble, SimpleSaeEncoder};
pub use encoding::{Spike, activation_to_spike_time};
pub use hypergraph::{Gse, HyperedgeKey, Hyperedge, HypergraphStore};
pub use metrics::{StiiResult, poly_count, entropy};