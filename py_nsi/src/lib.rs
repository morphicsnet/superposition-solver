//! Python bindings for nsi_core using pyo3 (ABI3, Python 3.10+).
//!
//! Exposes:
//! - PySimpleSaeEncoder: simple SAE-like encoder
//! - PyEnsemble: ensemble of encoders
//! - PySpike: spike event
//! - PyGse: temporal coincidence (GSE) islander
//! - PyHypergraphStore: hypergraph aggregation + HIF export + STII
//!
//! Minimal smoke test (conceptual):
//! ```python
//! from py_nsi import PySimpleSaeEncoder, PyEnsemble, PySpike, PyGse, PyHypergraphStore
//! enc = PySimpleSaeEncoder(4, 6, 2, 42)
//! ens = PyEnsemble([enc])
//! outs = ens.encode_all([0.1, 0.2, -0.1, 0.3])
//! mask = ens.intersect(outs, 0.0)
//! gse = PyGse(0.5)
//! s1 = PySpike(0, 1, 0.0); s2 = PySpike(1, 2, 0.2)
//! islands = gse.ingest(s1); islands += gse.ingest(s2)
//! store = PyHypergraphStore()
//! for isl in islands: store.add_island(isl)
//! _ = store.compute_stii([(0 << 32) | 1, (1 << 32) | 2], [(2, 1.0), (2, 0.0)])
//! ```

use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

#[pyclass]
pub struct PySimpleSaeEncoder {
    pub(crate) inner: nsi_core::SimpleSaeEncoder,
}

#[pymethods]
impl PySimpleSaeEncoder {
    /// Create a simple SAE-like encoder (deterministic with `seed`).
    /// Args:
    ///   in_dim: input dimension (> 0)
    ///   out_dim: output features (> 0)
    ///   top_k: number of nonzeros to keep (0..=out_dim)
    ///   seed: seed for deterministic init
    #[new]
    pub fn new(in_dim: usize, out_dim: usize, top_k: usize, seed: u64) -> PyResult<Self> {
        if in_dim == 0 || out_dim == 0 {
            return Err(PyValueError::new_err("in_dim and out_dim must be > 0"));
        }
        if top_k > out_dim {
            return Err(PyValueError::new_err("top_k must be <= out_dim"));
        }
        Ok(Self {
            inner: nsi_core::SimpleSaeEncoder::new(in_dim, out_dim, top_k, seed),
        })
    }
}

#[pyclass]
pub struct PyEnsemble {
    pub(crate) inner: nsi_core::Ensemble<nsi_core::SimpleSaeEncoder>,
}

#[pymethods]
impl PyEnsemble {
    /// Create an ensemble from a list of PySimpleSaeEncoder objects.
    #[new]
    pub fn new(encoders: Vec<Py<PySimpleSaeEncoder>>) -> PyResult<Self> {
        Python::with_gil(|py| {
            let mut inner_encs = Vec::with_capacity(encoders.len());
            for e in encoders {
                let borrowed = e.borrow(py);
                inner_encs.push(borrowed.inner.clone());
            }
            Ok(Self {
                inner: nsi_core::Ensemble { encoders: inner_encs },
            })
        })
    }

    /// Encode activations through all encoders, returning list of outputs.
    pub fn encode_all(&self, activations: Vec<f32>) -> PyResult<Vec<Vec<f32>>> {
        Ok(self.inner.encode_all(&activations))
    }

    /// Feature-wise intersection across encoder outputs using threshold.
    pub fn intersect(&self, outputs: Vec<Vec<f32>>, threshold: f32) -> PyResult<Vec<bool>> {
        Ok(self.inner.intersect_features(&outputs, threshold))
    }
}

#[pyclass]
#[derive(Clone)]
pub struct PySpike {
    #[pyo3(get, set)]
    pub ensemble_id: u16,
    #[pyo3(get, set)]
    pub neuron_id: u32,
    #[pyo3(get, set)]
    pub t: f32,
}

#[pymethods]
impl PySpike {
    #[new]
    pub fn new(ensemble_id: u16, neuron_id: u32, t: f32) -> Self {
        Self { ensemble_id, neuron_id, t }
    }

    pub fn node_id(&self) -> u64 {
        ((self.ensemble_id as u64) << 32) | (self.neuron_id as u64)
    }

    pub fn __repr__(&self) -> String {
        format!("PySpike(ensemble_id={}, neuron_id={}, t={})", self.ensemble_id, self.neuron_id, self.t)
    }
}

impl From<&PySpike> for nsi_core::Spike {
    fn from(s: &PySpike) -> Self {
        nsi_core::Spike {
            ensemble_id: s.ensemble_id,
            neuron_id: s.neuron_id,
            t: s.t,
        }
    }
}

impl From<nsi_core::Spike> for PySpike {
    fn from(s: nsi_core::Spike) -> Self {
        PySpike { ensemble_id: s.ensemble_id, neuron_id: s.neuron_id, t: s.t }
    }
}

#[pyclass]
pub struct PyGse {
    pub(crate) inner: nsi_core::Gse,
}

#[pymethods]
impl PyGse {
    #[new]
    pub fn new(window: f32) -> Self {
        Self { inner: nsi_core::Gse::new(window) }
    }

    /// Ingest a spike and possibly emit temporal islands.
    pub fn ingest(&mut self, spike: &PySpike) -> PyResult<Vec<Vec<PySpike>>> {
        let islands = self.inner.ingest(nsi_core::Spike::from(spike));
        let out: Vec<Vec<PySpike>> = islands
            .into_iter()
            .map(|isl| isl.into_iter().map(PySpike::from).collect())
            .collect();
        Ok(out)
    }
}

#[pyclass]
pub struct PyHypergraphStore {
    pub(crate) inner: nsi_core::HypergraphStore,
}

#[pymethods]
impl PyHypergraphStore {
    #[new]
    pub fn new() -> Self {
        Self { inner: nsi_core::HypergraphStore::new() }
    }

    /// Add a temporal island (list of PySpike) into the hypergraph aggregator.
    pub fn add_island(&mut self, island: Vec<PySpike>) -> PyResult<()> {
        let core_island: Vec<nsi_core::Spike> = island.iter().map(nsi_core::Spike::from).collect();
        self.inner.add_island(&core_island);
        Ok(())
    }

    /// Return edges as a list of Python dicts:
    /// { "key": [u64,...], "observation_count": int, "stii_weight": float }
    pub fn edges(&self, py: Python<'_>) -> PyResult<Vec<PyObject>> {
        let mut out = Vec::new();
        for e in self.inner.edges() {
            let d = PyDict::new(py);
            d.set_item("key", e.key.node_ids.clone())?;
            d.set_item("observation_count", e.observation_count)?;
            d.set_item("stii_weight", e.stii_weight)?;
            out.push(d.into());
        }
        Ok(out)
    }

    /// Export minimal HIF JSON to a file.
    pub fn export_hif(&self, path: &str) -> PyResult<()> {
        self.inner
            .export_hif(path)
            .map_err(|e| PyIOError::new_err(format!("export_hif failed: {}", e)))
    }

    /// Compute placeholder STII for the given hyperedge nodes and deltas.
    /// Returns the computed STII value (also updates the store).
    pub fn compute_stii(&mut self, mut node_ids: Vec<u64>, deltas: Vec<(u64, f32)>) -> PyResult<f32> {
        // Canonicalize key to match store representation.
        node_ids.sort_unstable();
        node_ids.dedup();
        if node_ids.len() < 2 {
            return Err(PyValueError::new_err("compute_stii requires at least two distinct node ids"));
        }
        let key = nsi_core::HyperedgeKey { node_ids };
        let res = self.inner.compute_stii(&key, &deltas);
        Ok(res.stii_value)
    }
}

#[pymodule]
fn py_nsi(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<PySimpleSaeEncoder>()?;
    m.add_class::<PyEnsemble>()?;
    m.add_class::<PySpike>()?;
    m.add_class::<PyGse>()?;
    m.add_class::<PyHypergraphStore>()?;
    Ok(())
}