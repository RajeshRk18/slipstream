// Copyright © 2026 Znewco, Inc. (d/b/a Zcash Open Development Lab)
// SPDX-License-Identifier: AGPL-3.0-only
//
// This file is part of ZODL Slipstream.
//
// ZODL Slipstream is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License,
// version 3 only, as published by the Free Software Foundation.
//
// ZODL Slipstream is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
// Affero General Public License for more details.
//
// Commercial licensing: see COMMERCIAL-LICENSE.md.

//! Lookup-map subtree build (v0.4 Plan B / Phase B0 shared machinery): batch-compute
//! every combine a fragment needs, then run shardtree's `LocatedTree::from_iter`
//! VERBATIM through a `Hashable` wrapper whose `combine` consults the precomputed
//! map — retention/pruning/checkpoint semantics untouched, byte-identical output.
//! Correct-by-construction: any combine the precompute misses (true Nil-padding
//! boundaries, odd tails) falls back to the scalar `MerkleHashOrchard::combine`,
//! so the map is an OPTIMIZATION SET, never a correctness surface.
//!
//! Consumers: the v0.4 `batch_combine` path (CPU batch-affine kernel,
//! batch_sinsemilla.rs — always compiled) and the banked B0 GPU offload
//! (gpu_subtree.rs, feature `gpu`). Extracted from gpu_subtree.rs in Task 13.

use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use incrementalmerkletree::{Hashable, Level, Position, Retention};
use orchard::tree::MerkleHashOrchard;
use rayon::iter::{IndexedParallelIterator as _, ParallelIterator as _};
use rayon::slice::ParallelSliceMut as _;
use shardtree::{LocatedPrunableTree, LocatedTree, Node, PrunableTree, Tree};
use zcash_protocol::consensus::BlockHeight;

use crate::persist::BUILD_CHUNK_SIZE;

/// The batched-combine contract shared by the CPU batch-affine kernel and the
/// GPU offload: `output[i]` byte-identical to
/// `MerkleHashOrchard::combine(Level::from(layers[i]), &lefts[i], &rights[i])`.
pub(crate) type BatchCombineFn = fn(&[u8], &[[u8; 32]], &[[u8; 32]]) -> Vec<[u8; 32]>;

thread_local! {
    /// Precomputed combines for the fragment `from_iter` is currently building on THIS thread.
    static PRECOMP: RefCell<Option<HashMap<[u8; 65], [u8; 32]>>> = const { RefCell::new(None) };
}

/// Map key: level byte ‖ left (32 LE) ‖ right (32 LE). Keying on the level too avoids any
/// (astronomically unlikely) cross-level hash-pair aliasing.
fn combine_key(level: Level, a: &MerkleHashOrchard, b: &MerkleHashOrchard) -> [u8; 65] {
    let mut k = [0u8; 65];
    k[0] = u8::from(level);
    k[1..33].copy_from_slice(&a.to_bytes());
    k[33..65].copy_from_slice(&b.to_bytes());
    k
}

/// The precomputed parent for this combine, if the map holds it. Shared with the
/// test's `TracedLookup` so a coverage check can never drift into observing a
/// lookup the engine no longer performs.
fn map_lookup(
    level: Level,
    a: &MerkleHashOrchard,
    b: &MerkleHashOrchard,
) -> Option<MerkleHashOrchard> {
    PRECOMP
        .with(|p| {
            p.borrow()
                .as_ref()
                .and_then(|m| m.get(&combine_key(level, a, b)).copied())
        })
        .and_then(|bytes| Option::from(MerkleHashOrchard::from_bytes(&bytes)))
}

/// `MerkleHashOrchard` whose `combine` consults the thread-local precomputed map.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct LookupHashOrchard(pub MerkleHashOrchard);

impl Hashable for LookupHashOrchard {
    fn empty_leaf() -> Self {
        Self(MerkleHashOrchard::empty_leaf())
    }
    fn empty_root(level: Level) -> Self {
        Self(MerkleHashOrchard::empty_root(level))
    }
    fn combine(level: Level, a: &Self, b: &Self) -> Self {
        match map_lookup(level, &a.0, &b.0) {
            Some(node) => Self(node),
            None => Self(MerkleHashOrchard::combine(level, &a.0, &b.0)), // scalar fallback
        }
    }
}

/// Build the combines `from_iter` will perform over leaves at absolute positions
/// `start..start + leaves.len()`, bottom-up (each level one call to `batch`),
/// recording every `(level, left, right) → parent`.
///
/// Pairs by ABSOLUTE position parity, because `from_iter` unites only at
/// `position.is_right_child()`: the combines it performs are exactly the internal
/// nodes of the dyadic decomposition of `start..end`. `start` is the pool's tree
/// size, so pairing by chunk-relative index instead misses on every combine
/// whenever it is odd. A node whose sibling falls outside the range is dropped,
/// not padded — `from_iter` gives it a `Nil` sibling and hashes nothing.
///
/// `lookup_map_covers_every_combine` pins the coverage; a miss stays correct via
/// the scalar fallback, so the map is an OPTIMIZATION SET, never a correctness
/// surface.
fn precompute_shard_map(
    batch: BatchCombineFn,
    start: u64,
    leaves: &[MerkleHashOrchard],
) -> HashMap<[u8; 65], [u8; 32]> {
    let mut map = HashMap::new();
    if leaves.len() <= 1 {
        return map;
    }
    let mut level: Vec<MerkleHashOrchard> = leaves.to_vec();
    // Absolute index of `level[0]` at the current level.
    let mut idx = start;
    let mut lvl: u8 = 0;
    while level.len() > 1 {
        // A leading right child pairs with a node outside this range: skip it.
        let lead = (idx & 1) as usize;
        let pairs = level.len().saturating_sub(lead) / 2;
        if pairs == 0 {
            break;
        }
        let layers = vec![lvl; pairs];
        let lefts: Vec<[u8; 32]> = (0..pairs).map(|i| level[lead + 2 * i].to_bytes()).collect();
        let rights: Vec<[u8; 32]> = (0..pairs)
            .map(|i| level[lead + 2 * i + 1].to_bytes())
            .collect();
        let parents = batch(&layers, &lefts, &rights);
        let mut next = Vec::with_capacity(pairs);
        for i in 0..pairs {
            let (l, r) = (&level[lead + 2 * i], &level[lead + 2 * i + 1]);
            map.insert(combine_key(Level::from(lvl), l, r), parents[i]);
            let p = Option::from(MerkleHashOrchard::from_bytes(&parents[i]))
                .unwrap_or_else(|| MerkleHashOrchard::combine(Level::from(lvl), l, r));
            next.push(p);
        }
        idx = (idx + lead as u64) / 2;
        level = next;
        lvl += 1;
    }
    map
}

/// Recursively rebuild a `PrunableTree<LookupHashOrchard>` as `PrunableTree<MerkleHashOrchard>`
/// (converts both the annotation `A` and the leaf value `V`).
fn convert_tree(t: &PrunableTree<LookupHashOrchard>) -> PrunableTree<MerkleHashOrchard> {
    match &**t {
        Node::Parent { ann, left, right } => Tree::parent(
            ann.as_ref().map(|h| Arc::new(h.0)),
            convert_tree(left),
            convert_tree(right),
        ),
        Node::Leaf { value } => Tree::leaf((value.0.0, value.1)),
        Node::Nil => Tree::empty(),
    }
}

/// Batched equivalent of `persist::build_subtrees` for Orchard. Same chunking and
/// structure; the only difference is batch-computed combines + the H-type convert.
/// Output is identical to the scalar path (gated by `lookup_build_matches_scalar`).
pub(crate) fn build_subtrees_lookup<const SHARD_HEIGHT: u8>(
    batch: BatchCombineFn,
    start_position: Position,
    commitments: &mut [Option<(MerkleHashOrchard, Retention<BlockHeight>)>],
) -> Vec<(
    LocatedPrunableTree<MerkleHashOrchard>,
    BTreeMap<BlockHeight, Position>,
)> {
    commitments
        .par_chunks_mut(BUILD_CHUNK_SIZE)
        .enumerate()
        .filter_map(|(i, chunk)| {
            let start = start_position + (i * BUILD_CHUNK_SIZE) as u64;
            let end = start + chunk.len() as u64;
            let leaves: Vec<MerkleHashOrchard> = chunk
                .iter()
                .map(|n| n.as_ref().expect("always Some").0)
                .collect();
            let map = precompute_shard_map(batch, start.into(), &leaves);

            PRECOMP.with(|p| *p.borrow_mut() = Some(map));
            let res = LocatedTree::from_iter(
                start..end,
                Level::from(SHARD_HEIGHT),
                chunk.iter_mut().map(|n| {
                    let (h, r) = n.take().expect("always Some");
                    (LookupHashOrchard(h), r)
                }),
            );
            PRECOMP.with(|p| *p.borrow_mut() = None);

            res.map(|res| {
                let converted = LocatedTree::from_parts(
                    res.subtree.root_addr(),
                    convert_tree(res.subtree.root()),
                )
                .expect("converted tree is structure-preserving, so it matches its root address");
                (converted, res.checkpoints)
            })
        })
        .collect()
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use incrementalmerkletree::Marking;
    use std::cell::Cell;
    use zcash_client_backend::data_api::ORCHARD_SHARD_HEIGHT;

    /// Deterministic valid Orchard nodes; the last leaf carries a `Checkpoint` retention.
    pub(crate) fn synth(n: usize) -> Vec<Option<(MerkleHashOrchard, Retention<BlockHeight>)>> {
        let mut out = Vec::with_capacity(n);
        let mut a = MerkleHashOrchard::empty_leaf();
        let mut b = MerkleHashOrchard::empty_root(Level::from(0));
        for i in 0..n {
            let c = MerkleHashOrchard::combine(Level::from(0), &a, &b);
            a = b;
            b = c;
            let r = if i + 1 == n {
                Retention::Checkpoint {
                    id: BlockHeight::from(100u32 + i as u32),
                    marking: Marking::Reference,
                }
            } else {
                Retention::Ephemeral
            };
            out.push(Some((c, r)));
        }
        out
    }

    thread_local! {
        /// (hits, misses) for the build `trace_build` is driving.
        static TRACE: Cell<(u64, u64)> = const { Cell::new((0, 0)) };
    }

    /// `LookupHashOrchard`'s twin that records whether each combine hit the map.
    /// Shares `map_lookup` with it, so it observes the real lookup, not a copy.
    #[derive(Clone, Debug, PartialEq)]
    struct TracedLookup(MerkleHashOrchard);

    impl Hashable for TracedLookup {
        fn empty_leaf() -> Self {
            Self(MerkleHashOrchard::empty_leaf())
        }
        fn empty_root(level: Level) -> Self {
            Self(MerkleHashOrchard::empty_root(level))
        }
        fn combine(level: Level, a: &Self, b: &Self) -> Self {
            let (h, m) = TRACE.get();
            match map_lookup(level, &a.0, &b.0) {
                Some(node) => {
                    TRACE.set((h + 1, m));
                    Self(node)
                }
                None => {
                    TRACE.set((h, m + 1));
                    Self(MerkleHashOrchard::combine(level, &a.0, &b.0))
                }
            }
        }
    }

    /// Precompute + `from_iter` for ONE chunk, returning (hits, misses).
    fn trace_build(
        start: u64,
        rows: Vec<Option<(MerkleHashOrchard, Retention<BlockHeight>)>>,
    ) -> (u64, u64) {
        let leaves: Vec<MerkleHashOrchard> =
            rows.iter().map(|r| r.as_ref().expect("some").0).collect();
        let map = precompute_shard_map(
            crate::batch_sinsemilla::orchard_combine_batch_cpu,
            start,
            &leaves,
        );
        let n = rows.len() as u64;
        TRACE.set((0, 0));
        PRECOMP.with(|p| *p.borrow_mut() = Some(map));
        let _ = LocatedTree::from_iter(
            Position::from(start)..Position::from(start + n),
            Level::from(ORCHARD_SHARD_HEIGHT),
            rows.into_iter().map(|r| {
                let (h, ret) = r.expect("some");
                (TracedLookup(h), ret)
            }),
        );
        PRECOMP.with(|p| *p.borrow_mut() = None);
        TRACE.get()
    }

    /// `start_position` is the pool's tree size, so the pairing must hold at every
    /// 2-adic valuation. Both gates sweep it.
    const STARTS: [u64; 11] = [0, 1, 2, 3, 4, 7, 8, 1023, 1024, 12345, 1_000_001];

    /// THE byte-equal gate for the v0.4 batch-affine path (always compiled).
    ///
    /// Swept over `start_position` as well as length: `start` is the pool's tree
    /// size when the call begins, so it carries an arbitrary 2-adic valuation in
    /// production, and the precompute's pairing must track absolute position
    /// parity to match `from_iter` at every one of them.
    #[test]
    fn lookup_build_matches_scalar() {
        for start in STARTS {
            for n in [1usize, 7, 1000, 1024, 2000, 5000] {
                let start = Position::from(start);
                let mut a = synth(n);
                let mut b = synth(n);
                let scalar = crate::persist::build_subtrees::<
                    MerkleHashOrchard,
                    ORCHARD_SHARD_HEIGHT,
                >(start, &mut a, crate::persist::BUILD_CHUNK_SIZE);
                let batched = build_subtrees_lookup::<ORCHARD_SHARD_HEIGHT>(
                    crate::batch_sinsemilla::orchard_combine_batch_cpu,
                    start,
                    &mut b,
                );
                assert_eq!(
                    scalar, batched,
                    "batch-affine build diverged from scalar at start={start:?} n={n}"
                );
            }
        }
    }

    /// The precompute must cover EVERY combine `from_iter` performs, at any
    /// `start_position` — a miss is silently correct (scalar fallback) but voids
    /// the optimization, which is exactly how the chunk-relative pairing hid.
    #[test]
    fn lookup_map_covers_every_combine() {
        for start in STARTS {
            // One chunk per case, so every n stays at or below BUILD_CHUNK_SIZE.
            // 976 is here for its odd level widths (61, 15, 7, 3) — the tail shape.
            for n in [2usize, 7, 976, 1000, 1023, 1024] {
                let (hits, misses) = trace_build(start, synth(n));
                assert_eq!(
                    misses,
                    0,
                    "lookup map missed {misses} of {} combines at start={start} n={n}",
                    hits + misses,
                );
            }
        }
    }
}
