//! Spike encoding primitives.

/// Map a scalar activation to an absolute spike time using a sigmoid,
/// dropping very low-activity values. Higher activations spike earlier
/// within the window [t_start, t_start + delta_t].
///
/// Returns:
/// - Some(t) if activation is sufficiently large
/// - None if the (sigmoid) probability is below a small cutoff
///
/// Notes:
/// - Deterministic, pure function.
/// - Uses a numerically stable sigmoid for typical ranges.
pub fn activation_to_spike_time(activation: f32, t_start: f32, delta_t: f32) -> Option<f32> {
    // Sigmoid
    let s = 1.0 / (1.0 + (-activation).exp());
    // Drop very low activations
    if s < 1e-3 {
        return None;
    }
    // Earlier time for higher activation
    let t = t_start + (1.0 - s) * delta_t.max(0.0);
    Some(t)
}

/// A spike event identified by ensemble and neuron, carrying an absolute time.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Spike {
    pub ensemble_id: u16,
    pub neuron_id: u32,
    pub t: f32,
}

impl Spike {
    /// Canonical 64-bit node id used by hypergraph (HIF export):
    /// (ensemble_id << 32) | neuron_id
    pub fn node_id(&self) -> u64 {
        ((self.ensemble_id as u64) << 32) | (self.neuron_id as u64)
    }
}