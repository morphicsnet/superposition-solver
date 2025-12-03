//! Hypergraph primitives and GSE (temporal coincidence) builder.
//!
//! GSE maintains a sliding window of spikes and forms cross-ensemble temporal
//! islands. Each island contains at most one (latest) spike per ensemble that
//! occurred within the window of the newest spike.
//!
//! HypergraphStore aggregates temporal islands into hyperedges keyed by the
//! set of participating node ids. A minimal HIF exporter is provided.

use crate::encoding::Spike;
use serde::{Deserialize, Serialize};
use std::collections::{hash_map::Entry, BTreeSet, HashMap, VecDeque};
use std::io::Write;
use std::path::Path;

/// GSE: temporal coincidence detector with a sliding time window.
pub struct Gse {
    window: f32,
    buffer: VecDeque<Spike>,
}

impl Gse {
    /// Create a new GSE with the given window (in the same time units as Spike.t).
    pub fn new(window: f32) -> Self {
        Self {
            window: window.max(0.0),
            buffer: VecDeque::new(),
        }
    }

    /// Ingest a spike and produce zero or one temporal islands.
    ///
    /// Behavior:
    /// - Drop spikes older than (newest.t - window)
    /// - For each ensemble present within the window, keep the latest spike
    /// - If at least two distinct ensembles are present, emit one island
    pub fn ingest(&mut self, spike: Spike) -> Vec<Vec<Spike>> {
        // Push and purge old spikes
        let t_now = spike.t;
        self.buffer.push_back(spike);
        while let Some(front) = self.buffer.front() {
            if t_now - front.t > self.window {
                self.buffer.pop_front();
            } else {
                break;
            }
        }

        // Collect latest spike per ensemble in current window
        let mut latest_by_ens: HashMap<u16, &Spike> = HashMap::new();
        for s in self.buffer.iter().rev() {
            if t_now - s.t <= self.window {
                latest_by_ens.entry(s.ensemble_id).or_insert(s);
            } else {
                // Since we iterate in reverse time, we can break once outside window
                break;
            }
        }

        if latest_by_ens.len() < 2 {
            return Vec::new();
        }

        // Deterministic ordering: sort by node_id
        let mut island: Vec<Spike> = latest_by_ens.values().cloned().cloned().collect();
        island.sort_by_key(|s| s.node_id());

        vec![island]
    }
}

/// Canonical hyperedge key: sorted, unique node ids.
#[derive(Clone, Debug, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub struct HyperedgeKey {
    pub node_ids: Vec<u64>,
}

impl HyperedgeKey {
    pub fn from_spikes(spikes: &[Spike]) -> Option<Self> {
        let mut ids: Vec<u64> = spikes.iter().map(|s| s.node_id()).collect();
        ids.sort_unstable();
        ids.dedup();
        if ids.len() < 2 {
            return None;
        }
        Some(Self { node_ids: ids })
    }
}

/// Hyperedge data aggregated over observations.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Hyperedge {
    pub key: HyperedgeKey,
    pub observation_count: u64,
    pub stii_weight: f32,
}

/// In-memory store of hyperedges aggregated from temporal islands.
pub struct HypergraphStore {
    map: HashMap<HyperedgeKey, Hyperedge>,
}

impl HypergraphStore {
    pub fn new() -> Self {
        Self { map: HashMap::new() }
    }

    /// Aggregate a temporal island into the hypergraph.
    pub fn add_island(&mut self, island: &[Spike]) {
        if let Some(key) = HyperedgeKey::from_spikes(island) {
            let entry = self.map.entry(key.clone()).or_insert(Hyperedge {
                key,
                observation_count: 0,
                stii_weight: 0.0,
            });
            entry.observation_count = entry.observation_count.saturating_add(1);
            // stii_weight left as-is; can be updated by metrics::compute_stii.
        }
    }

    /// Set or upsert the STII weight for a hyperedge key.
    /// If the edge does not exist, it is created with observation_count = 0.
    pub fn set_stii_weight(&mut self, key: &HyperedgeKey, value: f32) {
        match self.map.entry(key.clone()) {
            Entry::Occupied(mut e) => {
                e.get_mut().stii_weight = value;
            }
            Entry::Vacant(v) => {
                v.insert(Hyperedge {
                    key: key.clone(),
                    observation_count: 0,
                    stii_weight: value,
                });
            }
        }
    }

    /// Iterate over stored hyperedges.
    pub fn edges(&self) -> impl Iterator<Item = &Hyperedge> {
        self.map.values()
    }

    /// Export a minimal HIF JSON.
    ///
    /// Schema (minimal):
    /// {
    ///   "network-type": "hypergraph",
    ///   "nodes": [ { "id": u64 }, ... ],
    ///   "edges": [ { "id": usize, "key": [u64,...], "observation_count": u64, "stii_weight": f32 }, ... ],
    ///   "incidences": [ { "edge": usize, "nodes": [u64,...] }, ... ]
    /// }
    pub fn export_hif<P: AsRef<Path>>(&self, path: P) -> std::io::Result<()> {
        // Collect nodes and deterministically order edges.
        let mut node_set: BTreeSet<u64> = BTreeSet::new();
        let mut edges: Vec<&Hyperedge> = self.map.values().collect();
        edges.sort_by(|a, b| a.key.node_ids.cmp(&b.key.node_ids));
        for e in &edges {
            for n in &e.key.node_ids {
                node_set.insert(*n);
            }
        }

        #[derive(Serialize)]
        struct Node {
            id: u64,
        }
        #[derive(Serialize)]
        struct EdgeOut {
            id: usize,
            key: Vec<u64>,
            observation_count: u64,
            stii_weight: f32,
        }
        #[derive(Serialize)]
        struct Incidence {
            edge: usize,
            nodes: Vec<u64>,
        }
        #[derive(Serialize)]
        struct Hif {
            #[serde(rename = "network-type")]
            network_type: &'static str,
            nodes: Vec<Node>,
            edges: Vec<EdgeOut>,
            incidences: Vec<Incidence>,
        }

        let nodes: Vec<Node> = node_set.into_iter().map(|id| Node { id }).collect();
        let mut edges_out = Vec::with_capacity(edges.len());
        let mut incidences = Vec::with_capacity(edges.len());
        for (eid, e) in edges.iter().enumerate() {
            edges_out.push(EdgeOut {
                id: eid,
                key: e.key.node_ids.clone(),
                observation_count: e.observation_count,
                stii_weight: e.stii_weight,
            });
            incidences.push(Incidence {
                edge: eid,
                nodes: e.key.node_ids.clone(),
            });
        }

        let hif = Hif {
            network_type: "hypergraph",
            nodes,
            edges: edges_out,
            incidences,
        };

        let mut file = std::fs::File::create(path)?;
        let s = serde_json::to_string_pretty(&hif)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        file.write_all(s.as_bytes())
    }
}