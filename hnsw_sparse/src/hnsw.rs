use std::cmp::min;
use std::cmp::Ordering;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::collections::HashSet;
// most basic API

// uh somewhere we also need to worry about the type of data the vector holds (probably floats)
// an hnsw index is parametrized by its vector type T and a distance function D
// we are interested in sparse vectors, so T will be a list of (id, value) pairs and D will be
// cosine similarity

pub trait Distance<T> {
    fn eval(&self, v1: &SparseVector<T>, v2: &SparseVector<T>) -> f32;
}

// sparse vector
// for now, let's say it's just an alias for Vec: a sorted list of (id, value) pairs
// we can think about whether that will make sorting hard later
type SparseVector<T> = Vec<(usize, T)>;

pub struct CosineSimilarity;

// d = 1.0 - sum(Ai*Bi) / sqrt(sum(Ai*Ai) * sum(Bi*Bi))

type InternalKey = usize;

// T is the data type of the vector (float, int, etc.)
// D is the distance function
// K is the type for user-provided IDs
// TODO: we should have some restrictions on K and T
pub struct HNSWSparseIndex<K, T, D: Distance<T>> {
    // fill this in with parameters
    // the graph itself: this owns all the nodes!
    graph: HNSWGraph<K, T>,
    // entry point
    // Option because in new graphs it doesn't exist
    entry_point: Option<InternalKey>,
    max_layers: usize, // m_L
    // controls how big your search is
    ef_construction: usize, // efConstruction
    // number of neighbors a node gets on insertion
    num_insert_neighbors: usize, // M
    // max number of neighbors a node can have in layers > 0
    max_neighbors: usize, // M_max
    // max numbers of neighbors a node can have in layer 0
    max_neighbors_0: usize, // M_max0
    // in heuristic neighbor selection, should we extend candidates by their neighbors?
    extend_candidates: bool, // extendCandidates
    // in heuristic neighbor selection, should we add some originally discarded candidates?
    keep_pruned: bool, // keepPrunedConnections
    // distance function for nodes
    dist_func: D,
}

// TODO: concurrency
impl<K: Copy, T, D: Distance<T>> HNSWSparseIndex<K, T, D> {
    pub fn new(
        max_layers: usize,
        ef_construction: usize,
        num_insert_neighbors: usize,
        max_neighbors: usize,
        max_neighbors_0: usize,
        extend_candidates: bool,
        keep_pruned: bool,
        dist_func: D,
    ) -> Self {
        // TODO: is this correct rust syntax lol
        let mut graph = HNSWGraph::<K, T>::new();
        HNSWSparseIndex {
            graph: graph,
            entry_point: None,
            max_layers: max_layers,
            ef_construction: ef_construction,
            num_insert_neighbors: num_insert_neighbors,
            max_neighbors: max_neighbors,
            max_neighbors_0: max_neighbors_0,
            extend_candidates: extend_candidates,
            keep_pruned: keep_pruned,
            dist_func: dist_func,
        }
    }

    pub fn insert(&self, external_id: K, v: SparseVector<T>) {}

    // size of dynamic candidate list should be hidden
    // the size of the return array is dependent on k, so IDK what the right type is, AI said
    // Vec<T> or Box<[T]>
    pub fn k_nn_search(&self, query: &SparseVector<T>, k: usize) -> Vec<K> {
        if self.entry_point.is_none() {
            // graph is empty, we have no neighbors
            return Vec::new();
        }
        // running list of nearest elts
        let mut nearest_elts;
        // start by checking dist(query, entry_point)
        let enter_point: &Node<K, T> = self.get_node_with_internal_id(self.entry_point.unwrap());
        let mut enter_dist: NodeWithDistance = NodeWithDistance {
            internal_id: enter_point.internal_id,
            dist: self.dist_func.eval(query, &enter_point.data),
        };
        // we go down layers
        // in all but the bottom layer, we only select one element (the nearest)
        for layer in (1..=enter_point.max_layer).rev() {
            nearest_elts = self.search_layer(query, &BinaryHeap::from([enter_dist]), 1, layer);
            enter_dist = *nearest_elts.peek().unwrap();
        }
        // in the bottom layer, we finally actually find the ef_construction nearest elements
        nearest_elts = self.search_layer(
            query,
            &BinaryHeap::from([enter_dist]),
            self.ef_construction,
            0,
        );
        // get only the nearest k elements
        let mut k_nearest: Vec<K> = Vec::with_capacity(k);
        for i in 0..k {
            let nearest = nearest_elts.pop();
            if nearest.is_none() {
                // no more elements! we're done
                break;
            }
            let nearest_node = self.get_node_with_internal_id(nearest.unwrap().internal_id);
            k_nearest[i] = nearest_node.external_id;
        }
        k_nearest
    }

    // TODO: THIS SHOULD BE AN OPTION OR PANIC
    fn get_node_with_internal_id(&self, internal_id: InternalKey) -> &Node<K, T> {
        // right now, just check our big graph vec
        return self.graph.get(internal_id).unwrap();
    }

    // oh my god deletion seems really complicated
    // it is implemented by usearch and discussed in https://github.com/nmslib/hnswlib/issues/4. I
    // should do this last.
    // TODO: change key type
    pub fn remove(&self, external_id: K) {}

    // TODO: usize -> custom type
    // search layer is internal, so it only needs to use internal IDs
    // we use a heap because sorting
    // tl;dr for search on a specific layer: we start with a list of candidates (enter_points) of
    // neighbors and we improve this list by repeatedly checking if anything in the neighborhood of the
    // candidates is closer than the current neighbor list
    // this returns a binary heap of nodes with at most num_neighbors nodes
    // TODO: add some asserts
    fn search_layer(
        &self,
        query: &SparseVector<T>,
        enter_points: &BinaryHeap<NodeWithDistance>,
        num_neighbors: usize,
        layer_number: usize,
    ) -> BinaryHeap<NodeWithDistance> {
        // we keep track of the current ef nearest neighbors
        // and the candidates for nearest neighbors whose neighborhoods we should check
        // candidates is a min-heap since we care about the closest candidate
        let mut candidates: BinaryHeap<NodeWithDistance> = enter_points.clone();
        // BUT neighbors is a max-heap because we only care about the farthest neighbor
        let mut neighbors: BinaryHeap<Reverse<NodeWithDistance>> = BinaryHeap::new();
        // TODO: nicer way to do this?
        for &e in enter_points.iter() {
            neighbors.push(Reverse(e));
        }
        // we keep track also of visited nodes
        // at the beginning all enter_points are visited
        let mut visited: HashSet<InternalKey> =
            enter_points.iter().map(|&x| x.internal_id).collect();
        while !candidates.is_empty() {
            // c is the closest candidate to query
            let c: NodeWithDistance = candidates.pop().unwrap();
            // f is the furthest neighbor to query
            // we .0 because of the reverse
            // TODO: fix reverse stuff
            let f: NodeWithDistance = neighbors.peek().unwrap().0;
            if c.dist > f.dist {
                // all candidates too far, we're done
                break;
            }
            // otherwise, check the neighborhood of each candidate
            // any better candidate neighbor should be added to neighbors and candidates
            let c_node: &Node<K, T> = self.get_node_with_internal_id(c.internal_id);
            // TODO: check that layer number is valid
            for e in c_node.neighbors[layer_number].iter() {
                // if unvisited, let's see where e is
                if !visited.contains(&e.internal_id) {
                    // add e to visited
                    visited.insert(e.internal_id);
                    let e_dist = self.dist_func.eval(&e.data, query);
                    let e_with_distance = NodeWithDistance {
                        internal_id: e.internal_id,
                        dist: e_dist,
                    };
                    // if we don't have enough neighbors yet, just add it
                    if neighbors.len() < num_neighbors {
                        candidates.push(e_with_distance);
                        neighbors.push(Reverse(e_with_distance));
                    } else {
                        // otherwise, compare e to the furthest neighbor and see if it's better
                        let f = neighbors.peek().unwrap().0;
                        // if it is better, we should add it to neighbors
                        if e_dist < f.dist {
                            candidates.push(e_with_distance);
                            neighbors.push(Reverse(e_with_distance));
                            // now neighbors too large, so we pop the furthest one again
                            neighbors.pop();
                        }
                    }
                }
            } // end for loop through neighbors of c_node
        } // end while
          // at the end, convert neighbors back into a min-heap for use in select-neighbors
          // TODO: nicer way to do this?
        neighbors.iter().map(|x| x.0).collect()
    }

    // this is the select_neighbors_heurisitc from the paper, not the greedy version
    // we don't care about distances from base in the returned list of neighbors
    // TODO: should base be a vector, an ID, a node, or what? right now it's a vector
    // because base is only used if we want to extend our candidates
    fn select_neighbors(
        &self,
        base: &SparseVector<T>,
        candidates: BinaryHeap<NodeWithDistance>,
        num_neighbors: usize,
        layer_number: usize,
    ) -> Vec<InternalKey> {
        let mut working: BinaryHeap<NodeWithDistance> = candidates.clone();
        // we might add back some discarded candidates, so we keep track of them
        let mut discarded: BinaryHeap<NodeWithDistance> = BinaryHeap::new();
        // TODO: better way than "reverse"?
        let mut neighbors: Vec<InternalKey> = Vec::with_capacity(num_neighbors);
        // extend_candidates flag controls whether we add neighbors of candidates to candidates
        // paper says this should only be turned on if we expect data to be very clustered
        if self.extend_candidates {
            // make a hash set of the IDs
            let mut visited: HashSet<InternalKey> =
                candidates.iter().map(|&x| x.internal_id).collect();
            // add neighbors of candidates to the working queue
            // TODO: check layer number is valid
            for e in candidates.iter() {
                for e_adj in
                    self.get_node_with_internal_id(e.internal_id).neighbors[layer_number].iter()
                {
                    // no duplicates!
                    if !visited.contains(&e_adj.internal_id) {
                        working.push(NodeWithDistance {
                            internal_id: e_adj.internal_id,
                            dist: self.dist_func.eval(base, &e_adj.data),
                        });
                        visited.insert(e_adj.internal_id);
                    }
                }
            }
        } // end extend candidates
          // now select some neighbors from the candidates
        while !working.is_empty() && neighbors.len() < num_neighbors {
            let e: NodeWithDistance = working.pop().unwrap();
            // check distance from e to each possible neighbor
            let mut discard: bool = false;
            for &n in neighbors.iter() {
                // if e is closer to a neighbor, discard = true
                let e_n_dist: f32 = self.dist_func.eval(
                    &self.get_node_with_internal_id(e.internal_id).data,
                    &self.get_node_with_internal_id(n).data,
                );
                if e_n_dist < e.dist {
                    discard = true;
                }
            }
            if discard {
                discarded.push(e);
            } else {
                neighbors.push(e.internal_id);
            }
        } // end neighbor selection
          // add pruned connections if needed: goal is to return as close to num_neighbors neighbors
          // as possible
        if self.keep_pruned {
            // add min(num_neighbors - neighbors.len(), discarded.len()) elts to neighbors
            let num_to_add: usize = min(num_neighbors - neighbors.len(), discarded.len());
            for _ in 0..num_to_add {
                neighbors.push(discarded.pop().unwrap().internal_id);
            }
        } // end restoring pruned connections
        neighbors
    } // end select_neighbors
}

// a graph is an array of nodes
// nodes have an internal_id, an external_id, the vector, and their neighbors
// nodes are identified by internal_id
// but API users use external_id
// node neighbors are an array of L "neighbors per layer"

// the node's internal ID is just its index in the array
// but the external ID is separate
// for deletions, we might need something fancier, unclear

// the hnsw algorithm only cares about neighbors per layer
// so we don't need to keep track of which nodes are on which layer
// it's enough to know the entry point

// "node with distance" is used to make searching easier
// the point we are taking distance relative to is not stored
#[derive(PartialEq, Copy, Clone)]
struct NodeWithDistance {
    internal_id: InternalKey,
    dist: f32,
}

// careful because can be nan()
// floats only implement PartialOrd and PartialEq
// note that in order to make our binary heaps sort by min distance, we reverse the ordering
impl PartialOrd for NodeWithDistance {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        other.dist.partial_cmp(&self.dist)
    }
}

impl Ord for NodeWithDistance {
    fn cmp(&self, other: &Self) -> Ordering {
        if self.dist.is_nan() || other.dist.is_nan() {
            panic!("distance for node {} is NaN", self.internal_id);
        } else {
            other.partial_cmp(&self).unwrap()
        }
    }
}

// TODO: hnswlib-rs just makes Eq blank
impl Eq for NodeWithDistance {}

// NOTE: we should never be cloning or copying nodes.
// we should touch nodes.neighbors
// but all other parts of Node should be created during insertion and only read afterwards
struct Node<K, T> {
    data: SparseVector<T>,
    internal_id: InternalKey,
    // the layer is used when we want to adjust entry point (maybe)
    max_layer: usize,
    external_id: K,
    // one neighbor list per layer
    neighbors: Vec<NeighborLayer<K, T>>,
}

type HNSWGraph<K, T> = Vec<Node<K, T>>;
type NeighborLayer<K, T> = Vec<Node<K, T>>;
