use rand::Rng;
use std::cmp::min;
use std::cmp::Ordering;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::collections::HashMap;
use std::collections::HashSet;
// most basic API

// uh somewhere we also need to worry about the type of data the vector holds (probably floats)
// an hnsw index is parametrized by its vector type T and a distance function D
// we are interested in sparse vectors, so T will be a list of (id, value) pairs and D will be
// cosine similarity

type DistType = f64;
// apparently a way to allow multiple distance implementations for the same type
trait Dist {}
struct CosSim;
impl Dist for CosSim {}

pub trait Distance<T: Dist> {
    fn dist(&self, other: &Self) -> DistType;
}

// sparse vector
// for now, let's say it's just an alias for Vec: a sorted list of (id, value) pairs
// we can think about whether that will make sorting hard later
type SparseVector = Vec<(usize, f64)>;

trait Normed {
    fn norm(&self) -> DistType;
}

impl Normed for SparseVector {
    fn norm(&self) -> DistType {
        self.iter().fold(0.0, |acc, (_, &val)| acc + val*val).sqrt()
    }
}

impl Distance<CosSim> for SparseVector {
    fn dist(&self, other: &Self) -> DistType {
        let accum : DistType = 0.0;
        // because both are sorted, it's easy to do sorted set intersection
        let mut v1 = self.iter().peekable();
        let mut v2 = other.iter().peekable();
        // compare v1.peek() and v2.peek()
        // if they're the same, they go in intersection
        // if v1 < v2, advance v1
        // if v2 > v1, advance v2
        while !(v1.peek().is_none() || v2.peek().is_none()) {
            let w1 = v1.peek().unwrap();
            let w2 = v2.peek().unwrap();
            if w1.0 == w2.0 {
                accum += w1.1 * w2.1;
            } else if w1.0 < w2.0 {
                v1.next();
            } else {
                v2.next();
            }
        } 
        1.0 - accum / (self.norm() * other.norm())
    }
}

// d = 1.0 - sum(Ai*Bi) / sqrt(sum(Ai*Ai) * sum(Bi*Bi))


type InternalKey = usize;

// D is any data with a distance function (we care about SparseVector)
// K is the type for user-provided IDs
// TODO: we should have some restrictions on K and T
pub struct HNSWSparseIndex<K, D> {
    // fill this in with parameters
    // the graph itself: this owns all the nodes!
    graph: HNSWGraph<K, D>,
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
    curr_internal_id: InternalKey,
}

// TODO: concurrency
impl<O: Dist, K: Copy, D: Distance<O>> HNSWSparseIndex<K, D> {
    pub fn new(
        max_layers: usize,
        ef_construction: usize,
        num_insert_neighbors: usize,
        max_neighbors: usize,
        max_neighbors_0: usize,
        extend_candidates: bool,
        keep_pruned: bool,
    ) -> Self {
        // TODO: is this correct rust syntax lol
        let graph = HNSWGraph::<K, D>::new();
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
            curr_internal_id: 0,
        }
    }

    fn rand_layer(&self) -> usize {
        let mut rng = rand::thread_rng();
        (rng.gen::<f64>().ln() * (self.max_layers as f64)).floor() as usize
    }

    fn next_internal_id(&mut self) -> InternalKey {
        let id = self.curr_internal_id;
        self.curr_internal_id += 1;
        id
    }

    // TODO: this isn't building because I am borrowing self as mutable (when I update neighbors)
    // as well as immutable :( 
    // Apparently I can fix this using reference counting
    pub fn insert(&mut self, external_id: K, v: D) {
        let ins_layer: usize = self.rand_layer();
        let id: InternalKey = self.next_internal_id();
        let mut new_node: Node<K, D> = Node::new(v, id, ins_layer, external_id);
        // the really dumb case: empty graph
        // there are no connections to make, so we just set the entry point and finish
        if self.entry_point.is_none() {
            self.entry_point = Some(id);
            return;
        }
        // graph nonempty: we need to make some connections

        // convert entry_point into a NodeWithDistance
        let entry_id: InternalKey = self.entry_point.unwrap();
        let entry_dist = NodeWithDistance {
            internal_id: entry_id,
            dist: v.dist(&self.get_node_with_internal_id(entry_id).data),
        };
        let mut nearest: BinaryHeap<NodeWithDistance> = BinaryHeap::from([entry_dist]);
        // on the layers atop ins_layer, just find one element
        for curr_layer in (ins_layer..=self.max_layers).rev() {
            nearest = self.search_layer(&v, &nearest, 1, curr_layer);
        }
        // then from ins_layer to 0, we need to give our node connections!
        for curr_layer in (0..=ins_layer).rev() {
            // find ef_construction nearest elements
            nearest = self.search_layer(&v, &nearest, self.ef_construction, curr_layer);
            // from those, select neighbors
            let neighbors: Vec<InternalKey> =
                self.select_neighbors(&v, &nearest, self.num_insert_neighbors, curr_layer);
            // add each other as neighbors!
            // easy for the new node
            new_node.neighbors[curr_layer] = HashSet::from_iter(neighbors);
            // for everyone else
            for &n in neighbors.iter() {
                let n_node : &mut Node<K, D> = self.get_mut_node_with_internal_id(n);
                n_node.neighbors[curr_layer].insert(id);
                let max_neighbors: usize = if curr_layer == 0 {
                    self.max_neighbors
                } else {
                    self.max_neighbors_0
                };
                // now shrink connections if needed
                if n_node.neighbors[curr_layer].len() > max_neighbors {
                    let n_data: &D = &n_node.data;
                    // select new neighbors for n
                    // first convert the neighbors of n to BinaryHeap<NodeWithDistance>
                    let n_old_neighbors: BinaryHeap<NodeWithDistance> =
                        BinaryHeap::from_iter(n_node.neighbors[curr_layer].iter().map(|&x| {
                            NodeWithDistance {
                                internal_id: x,
                                dist: n_data.dist(&self.get_node_with_internal_id(x).data),
                            }
                        }));
                    // then select new neighbors from old_neighbors
                    let n_new_neighbors: Vec<InternalKey> =
                        self.select_neighbors(n_data, &n_old_neighbors, max_neighbors, curr_layer);
                    // TODO: prune any connections between n and discarded? This is not mentioned
                    // in the paper, but I would assume the graph is bidirectional
                    n_node.neighbors[curr_layer] = HashSet::from_iter(n_new_neighbors);
                } // end shrink connections for neighbors
            } // end adding connections on layer l
        } // end connection-building for layers l to 0
        // add our node to the graph!
        self.graph.insert(id, new_node);
    } // end insert

    // size of dynamic candidate list should be hidden
    // the size of the return array is dependent on k, so IDK what the right type is, AI said
    // Vec<T> or Box<[T]>
    pub fn k_nn_search(&self, query: &D, k: usize) -> Vec<K> {
        if self.entry_point.is_none() {
            // graph is empty, we have no neighbors
            return Vec::new();
        }
        // running list of nearest elts
        let mut nearest_elts: BinaryHeap<NodeWithDistance>;
        // start by checking dist(query, entry_point)
        let enter_point: &Node<K, D> = self.get_node_with_internal_id(self.entry_point.unwrap());
        let mut enter_dist: NodeWithDistance = NodeWithDistance {
            internal_id: enter_point.internal_id,
            dist: query.dist(&enter_point.data),
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
    fn get_node_with_internal_id(&self, internal_id: InternalKey) -> &Node<K, D> {
        // right now, just check our big graph vec
        return self.graph.get(&internal_id).unwrap();
    }
    
    // get node but it's mutable
    fn get_mut_node_with_internal_id(&mut self, internal_id : InternalKey) -> &mut Node<K, D> {
        return self.graph.get_mut(&internal_id).unwrap();
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
        query: &D,
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
            let c_node: &Node<K, D> = self.get_node_with_internal_id(c.internal_id);
            // TODO: check that layer number is valid
            for &e in c_node.neighbors[layer_number].iter() {
                // if unvisited, let's see where e is
                if !visited.contains(&e) {
                    // add e to visited
                    visited.insert(e);
                    let e_dist = query.dist(&self.get_node_with_internal_id(e).data);
                    let e_with_distance = NodeWithDistance {
                        internal_id: e,
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
        base: &D,
        candidates: &BinaryHeap<NodeWithDistance>,
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
                for &e_adj in
                    self.get_node_with_internal_id(e.internal_id).neighbors[layer_number].iter()
                {
                    // no duplicates!
                    if !visited.contains(&e_adj) {
                        working.push(NodeWithDistance {
                            internal_id: e_adj,
                            dist: base.dist(&self.get_node_with_internal_id(e_adj).data),
                        });
                        visited.insert(e_adj);
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
                let e_n_dist: DistType = self
                    .get_node_with_internal_id(n)
                    .data
                    .dist(&self.get_node_with_internal_id(e.internal_id).data);
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
    dist: DistType,
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
struct Node<K, D> {
    data: D,
    internal_id: InternalKey,
    // the layer is used when we want to adjust entry point (maybe)
    max_layer: usize,
    external_id: K,
    // one neighbor list per layer
    neighbors: Vec<NeighborLayer>,
}

impl<K, D> Node<K, D> {
    fn new(data: D, internal_id: InternalKey, max_layer: usize, external_id: K) -> Self {
        Node {
            data: data,
            internal_id: internal_id,
            max_layer: max_layer,
            external_id: external_id,
            neighbors: Vec::with_capacity(max_layer),
        }
    }
}

type HNSWGraph<K, D> = HashMap<InternalKey, Node<K, D>>;
type NeighborLayer = HashSet<InternalKey>;
