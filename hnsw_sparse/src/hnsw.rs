use std::collections::BinaryHeap;
use std::cmp::Ordering;
// most basic API

// uh somewhere we also need to worry about the type of data the vector holds (probably floats)
// an hnsw index is parametrized by its vector type T and a distance function D 
// we are interested in sparse vectors, so T will be a list of (id, value) pairs and D will be
// cosine similarity
// alternately, we could only think about sparse vectors, so T is the type of data the vector holds

pub trait Distance<T> {
    fn eval(&self, v1 : SparseVector<T>, v2 : SparseVector<T>) -> f32;
}

// sparse vector
// for now, let's say it's just an alias for Vec
// we can think about whether that will make sorting hard later

type SparseVector<T> = Vec<(u16, T)>;

type InternalKey = usize;

// T is the data type of the vector (float, int, etc.)
// D is the distance function
// K is the type for user-provided IDs
// TODO: we should have some restrictions on K and T
pub struct hnsw_sparse_index<K, T, D : Distance<T>> {
    // fill this in with parameters
    // the graph itself
    graph : Graph<K, T>,
    // entry point (IDK how this is set)
    entry_point : Node<K, T>,
    max_layers : usize, // m_L
    // controls how big your search is
    ef_construction : usize, // efConstruction
    // number of neighbors a node gets on insertion
    num_insert_neighbors : usize, // M
    // max number of neighbors a node can have in layers > 0
    max_neighbors : usize, // M_max
    // max numbers of neighbors a node can have in layer 0
    max_neighbors_0 : usize, // M_max0
}

// TODO: concurrency
impl <K, T, D : Distance<T>> hnsw_sparse_index<K, T, D> {
    // TODO: add params
    pub fn new() -> Self {
        hnsw_sparse_index {
            // params
        }
    }

    pub fn insert(&self, v : SparseVector<T>) {
    }

    // size of dynamic candidate list should be hidden
    // the size of the return array is dependent on k, so IDK what the right type is, AI said
    // Vec<T> or Box<[T]>
    pub fn k_nn_search(&self, query : SparseVector<T>, k : u16) -> Vec<SparseVector<T>> {
    }

    // oh my god deletion seems really complicated
    // it is implemented by usearch and discussed in https://github.com/nmslib/hnswlib/issues/4. I
    // should do this last.
    // TODO: change key type
    pub fn remove(&self, key : Key) {
    }

    // TODO: usize -> custom type
    // search layer is internal, so it only needs to use internal IDs
    // we use a heap because sorting
    fn search_layer(&self, query : SparseVector<T>, enter_points : BinaryHeap<NodeWithDistance>, layer_number : usize) -> BinaryHeap<NodeWithDistance> {
    }

    // TODO: the binary heap needs a pair Key, "dist"
    fn select_neighbors(&self, base : Node<T>, candidates : BinaryHeap<NodeWithDistance>, num_neigbors : u16, layer_number : usize, extend_candidates : bool, keep_pruned : bool) -> BinaryHeap<NodeWithDistance> {
    }
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
struct NodeWithDistance {
    internal_id : InternalKey,
    dist : f32,
}

impl Ord for NodeWithDistance {
    fn cmp(&self, other: &Self) -> Ordering {
        self.dist.cmp(&other.dist)
    }
}

struct Node<K, T> {
    data : SparseVector<T>,
    internal_id : InternalKey,
    // TODO: user determines type of user_id
    user_id : K,
    // one neighbor list per layer
    neighbors : Vec<NeighborLayer<K, T>>,
}

type HNSWGraph<K, T> = Vec<Node<K, T>>;
type NeighborLayer<K, T> = Vec<Node<K, T>>;
