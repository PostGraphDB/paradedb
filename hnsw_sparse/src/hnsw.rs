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

type Key = u16;

pub struct hnsw_sparse_index<T, D : Distance<T>> {
    // fill this in with parameters
}

// TODO: all entries should include a unique ID somehow 
// TODO: what is Arc<PointWithOrder> in hnswlib
// TODO: concurrency
impl <T, D : Distance<T>> hnsw_sparse_index<T, D> {
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

    // TODO: is this right for enter_points
    // TODO: the binary heap needs a pair Key, "dist"
    // we use a heap because sorting
    fn search_layer(&self, query : SparseVector<T>, enter_points : Vec<Key>, layer_number : u16) -> BinaryHeap<Key> {
    }

    // TODO: the binary heap needs a pair Key, "dist"
    fn select_neighbors(&self, base : Key, candidates : Vec<Key>, num_neigbors : u16, layer_number : u16, extend_candidates : bool, keep_pruned : bool) -> BinaryHeap<Key> {
    }
}

// TODO: how to store the multilevel graph?
