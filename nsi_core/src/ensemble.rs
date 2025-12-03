//! Ensemble encoders, including a simple SAE-like encoder.
//!
//! Deterministic, seedable weight init and simple linear -> ReLU -> top-k sparsification.
//!
//! Example
//! -------
//! ```rust
//! use nsi_core::ensemble::{Encoder, SimpleSaeEncoder, Ensemble};
//!
//! let enc = SimpleSaeEncoder::new(4, 6, 2, 42);
//! let outs = enc.encode(&[0.1, 0.2, -0.1, 0.3]);
//! assert_eq!(outs.len(), 6);
//!
//! let ens = Ensemble { encoders: vec![enc.clone(), enc] };
//! let all = ens.encode_all(&[0.1, 0.2, -0.1, 0.3]);
//! let inter = ens.intersect_features(&all, 0.0);
//! assert_eq!(inter.len(), 6.min(all[0].len()));
//! ```

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Encoder trait: transforms activations into sparse codes.
pub trait Encoder {
    fn encode(&self, activations: &[f32]) -> Vec<f32>;
}

/// Simple SAE-like encoder:
/// y = ReLU(W x + b) then keep the top-k activations (ties broken by index).
#[derive(Clone)]
pub struct SimpleSaeEncoder {
    pub in_dim: usize,
    pub out_dim: usize,
    pub weights: Vec<Vec<f32>>, // [out_dim][in_dim]
    pub biases: Vec<f32>,       // [out_dim]
    pub top_k: usize,
    pub seed: u64,
}

impl SimpleSaeEncoder {
    /// Create a new encoder with deterministic weights/biases using StdRng::from_seed.
    pub fn new(in_dim: usize, out_dim: usize, top_k: usize, seed: u64) -> Self {
        assert!(in_dim > 0 && out_dim > 0, "in_dim and out_dim must be > 0");
        // Expand u64 seed into 32 bytes for StdRng::from_seed
        let mut seed_bytes = [0u8; 32];
        let s = seed.to_le_bytes();
        for i in 0..4 {
            seed_bytes[i * 8..(i + 1) * 8].copy_from_slice(&s);
        }
        let mut rng = StdRng::from_seed(seed_bytes);

        // Small uniform initialization for readability and determinism.
        let mut weights = Vec::with_capacity(out_dim);
        for _ in 0..out_dim {
            let mut row = Vec::with_capacity(in_dim);
            for _ in 0..in_dim {
                let r: f32 = rng.gen(); // [0,1)
                row.push(r * 0.2 - 0.1); // [-0.1, 0.1)
            }
            weights.push(row);
        }
        let mut biases = Vec::with_capacity(out_dim);
        for _ in 0..out_dim {
            let r: f32 = rng.gen();
            biases.push(r * 0.02 - 0.01); // [-0.01, 0.01)
        }

        Self {
            in_dim,
            out_dim,
            weights,
            biases,
            top_k,
            seed,
        }
    }
}

impl Encoder for SimpleSaeEncoder {
    fn encode(&self, activations: &[f32]) -> Vec<f32> {
        // Note: trait returns Vec not Result; on mismatch, return all-zeros deterministically.
        if activations.len() != self.in_dim {
            return vec![0.0; self.out_dim];
        }

        // Linear + ReLU
        let mut y = vec![0.0f32; self.out_dim];
        for (j, row) in self.weights.iter().enumerate() {
            let mut acc = self.biases[j];
            for i in 0..self.in_dim {
                acc += row[i] * activations[i];
            }
            y[j] = if acc > 0.0 { acc } else { 0.0 };
        }

        // Stable top-k sparsification (ties broken by index). Keep exactly k non-zeros if k <= out_dim.
        let k = self.top_k.min(self.out_dim);
        if k == 0 {
            for v in &mut y {
                *v = 0.0;
            }
            return y;
        }
        if k < self.out_dim {
            let mut idx_vals: Vec<(usize, f32)> = y.iter().copied().enumerate().collect();
            idx_vals.sort_by(|a, b| {
                // Sort by value desc; then by index asc to break ties deterministically.
                match b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal) {
                    std::cmp::Ordering::Equal => a.0.cmp(&b.0),
                    other => other,
                }
            });
            let mut keep = vec![false; self.out_dim];
            for (idx, _) in idx_vals.into_iter().take(k) {
                keep[idx] = true;
            }
            for j in 0..self.out_dim {
                if !keep[j] {
                    y[j] = 0.0;
                }
            }
        }
        y
    }
}

/// A collection of encoders that can be applied to the same activation vector.
pub struct Ensemble<E: Encoder> {
    pub encoders: Vec<E>,
}

impl<E: Encoder> Ensemble<E> {
    /// Encode with all encoders, returning a vector of each encoder's output.
    pub fn encode_all(&self, activations: &[f32]) -> Vec<Vec<f32>> {
        self.encoders.iter().map(|e| e.encode(activations)).collect()
    }

    /// Feature-wise intersection across encoders:
    /// returns a boolean mask where an index is true if every encoder's output at that
    /// position is strictly greater than `threshold`. The length equals the minimum
    /// output length across the provided outputs for safety.
    pub fn intersect_features(&self, outputs: &[Vec<f32>], threshold: f32) -> Vec<bool> {
        if outputs.is_empty() {
            return Vec::new();
        }
        let min_len = outputs.iter().map(|v| v.len()).min().unwrap_or(0);
        let mut mask = vec![false; min_len];
        for i in 0..min_len {
            let mut all = true;
            for out in outputs.iter() {
                if out[i] <= threshold {
                    all = false;
                    break;
                }
            }
            mask[i] = all;
        }
        mask
    }
}