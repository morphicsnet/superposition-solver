//! Metrics utilities: STII placeholder aggregator and polysemanticity helpers.
//
//! - STII: stable placeholder aggregation via weighted average by subset size.
//! - Polysemanticity helpers: count above epsilon and entropy with normalization.

use crate::hypergraph::{HyperedgeKey, HypergraphStore};

/// Result of a STII computation for a given hyperedge.
pub struct StiiResult {
    pub hyperedge_key: HyperedgeKey,
    pub stii_value: f32,
}

impl HypergraphStore {
    /// Compute a stable placeholder STII value for the given hyperedge key.
    ///
    /// Aggregation rule (deterministic):
    /// - `deltas`: (subset_size, delta_value)
    /// - Returns weighted average: sum_i (w_i * v_i) / sum_i (w_i), w_i = subset_size_i
    /// - If no positive weight, returns 0.0
    ///
    /// Implementation note:
    /// - Updates the store's hyperedge STII weight if the edge exists or will
    ///   be created on update by `set_stii_weight` (see hypergraph module).
    pub fn compute_stii(&mut self, key: &HyperedgeKey, deltas: &[(u64, f32)]) -> StiiResult {
        let mut num = 0.0_f64;
        let mut den = 0.0_f64;
        for &(w, v) in deltas.iter() {
            if w > 0 {
                let wf = w as f64;
                num += wf * (v as f64);
                den += wf;
            }
        }
        let stii = if den > 0.0 { (num / den) as f32 } else { 0.0 };

        // Update store's record (safe upsert).
        // This method is provided by the hypergraph module and will insert or update the edge.
        self.set_stii_weight(key, stii);

        StiiResult {
            hyperedge_key: key.clone(),
            stii_value: stii,
        }
    }
}

/// Count entries strictly greater than epsilon (with small numerical safeguard).
pub fn poly_count(prob: &[f32], eps: f32) -> usize {
    let e = if eps.is_finite() { eps.max(0.0) } else { 1e-6 };
    prob.iter().filter(|&p| *p > e).count()
}

/// Entropy (natural log) of a probability vector with normalization and epsilon for stability.
/// If sum is non-positive, returns 0.0.
pub fn entropy(prob: &[f32]) -> f32 {
    if prob.is_empty() {
        return 0.0;
    }
    let sum: f32 = prob.iter().cloned().sum();
    if !sum.is_finite() || sum <= 0.0 {
        return 0.0;
    }
    let eps = 1e-12_f32;
    let mut h = 0.0_f32;
    for &p_raw in prob.iter() {
        let p = (p_raw.max(0.0)) / sum;
        if p > 0.0 {
            let pl = (p + eps).ln();
            h -= p * pl;
        }
    }
    h
}