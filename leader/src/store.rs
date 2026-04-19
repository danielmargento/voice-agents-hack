use common::EmbeddingChunk;
use std::sync::RwLock;

#[derive(Clone)]
pub struct StoredChunk {
    pub chunk: EmbeddingChunk,
}

pub struct QueryFilter {
    pub time_start_ms: Option<u64>,
    pub time_end_ms: Option<u64>,
    pub camera_ids: Option<Vec<String>>,
    pub top_k: usize,
    /// When present, chunks are ranked by 0.7 * cosine_similarity + 0.3 * recency.
    /// Compared only against the video portion of the stored embedding (first video_dim dims).
    /// When absent, falls back to pure recency ordering.
    pub query_embedding: Option<Vec<f32>>,
}

pub struct EmbeddingStore {
    chunks: RwLock<Vec<StoredChunk>>,
    max_size: usize,
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }
    let dot: f32 = a[..n].iter().zip(&b[..n]).map(|(x, y)| x * y).sum();
    let na: f32 = a[..n].iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b[..n].iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 { 0.0 } else { dot / (na * nb) }
}

impl EmbeddingStore {
    pub fn new(max_size: usize) -> Self {
        Self {
            chunks: RwLock::new(Vec::new()),
            max_size,
        }
    }

    pub fn push(&self, chunk: EmbeddingChunk) {
        let mut chunks = self.chunks.write().unwrap_or_else(|e| e.into_inner());
        chunks.push(StoredChunk { chunk });
        if chunks.len() > self.max_size {
            chunks.remove(0);
        }
    }

    pub fn query(&self, filter: &QueryFilter) -> Vec<StoredChunk> {
        let chunks = self.chunks.read().unwrap_or_else(|e| e.into_inner());
        let mut matched: Vec<StoredChunk> = chunks
            .iter()
            .filter(|sc| {
                if let Some(start) = filter.time_start_ms {
                    if sc.chunk.end_ts_ms < start {
                        return false;
                    }
                }
                if let Some(end) = filter.time_end_ms {
                    if sc.chunk.start_ts_ms > end {
                        return false;
                    }
                }
                if let Some(ref cam_ids) = filter.camera_ids {
                    if !cam_ids.contains(&sc.chunk.camera_id) {
                        return false;
                    }
                }
                true
            })
            .cloned()
            .collect();

        if let Some(ref q_emb) = filter.query_embedding {
            let max_ts = matched.iter().map(|s| s.chunk.start_ts_ms).max().unwrap_or(0);
            let min_ts = matched.iter().map(|s| s.chunk.start_ts_ms).min().unwrap_or(0);
            let ts_span = max_ts.saturating_sub(min_ts).max(1) as f32;
            matched.sort_by(|a, b| {
                // Compare only against the video portion; use full embedding if video_dim unset.
                let vd_a = if a.chunk.video_dim > 0 { a.chunk.video_dim.min(a.chunk.embedding.len()) } else { a.chunk.embedding.len() };
                let vd_b = if b.chunk.video_dim > 0 { b.chunk.video_dim.min(b.chunk.embedding.len()) } else { b.chunk.embedding.len() };
                let sem_a = cosine_similarity(q_emb, &a.chunk.embedding[..vd_a]);
                let sem_b = cosine_similarity(q_emb, &b.chunk.embedding[..vd_b]);
                let rec_a = a.chunk.start_ts_ms.saturating_sub(min_ts) as f32 / ts_span;
                let rec_b = b.chunk.start_ts_ms.saturating_sub(min_ts) as f32 / ts_span;
                let score_a = 0.7 * sem_a + 0.3 * rec_a;
                let score_b = 0.7 * sem_b + 0.3 * rec_b;
                score_b.partial_cmp(&score_a).unwrap_or(std::cmp::Ordering::Equal)
            });
        } else {
            matched.sort_by(|a, b| b.chunk.start_ts_ms.cmp(&a.chunk.start_ts_ms));
        }

        matched.truncate(filter.top_k);
        matched
    }

    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        self.chunks.read().unwrap_or_else(|e| e.into_inner()).len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_chunk(camera_id: &str, ts_ms: u64) -> EmbeddingChunk {
        EmbeddingChunk {
            chunk_id: format!("{camera_id}-{ts_ms}"),
            camera_id: camera_id.into(),
            start_ts_ms: ts_ms,
            end_ts_ms: ts_ms + 5000,
            embedding: vec![0.1, 0.2, 0.3],
            video_dim: 3,
            audio_dim: 0,
            caption: None,
            representative_jpeg: None,
        }
    }

    fn make_chunk_with_embedding(camera_id: &str, ts_ms: u64, emb: Vec<f32>) -> EmbeddingChunk {
        let dim = emb.len();
        EmbeddingChunk {
            chunk_id: format!("{camera_id}-{ts_ms}"),
            camera_id: camera_id.into(),
            start_ts_ms: ts_ms,
            end_ts_ms: ts_ms + 5000,
            embedding: emb,
            video_dim: dim,
            audio_dim: 0,
            caption: None,
            representative_jpeg: None,
        }
    }

    fn no_filter(top_k: usize) -> QueryFilter {
        QueryFilter { time_start_ms: None, time_end_ms: None, camera_ids: None, top_k, query_embedding: None }
    }

    #[test]
    fn push_stores_chunk() {
        let store = EmbeddingStore::new(100);
        store.push(make_chunk("cam-0", 1000));
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn query_no_filter_returns_top_k() {
        let store = EmbeddingStore::new(100);
        for i in 0..10u64 {
            store.push(make_chunk("cam-0", i * 1000));
        }
        let results = store.query(&no_filter(5));
        assert_eq!(results.len(), 5);
    }

    #[test]
    fn query_time_range_filters_correctly() {
        let store = EmbeddingStore::new(100);
        store.push(make_chunk("cam-0", 1000));
        store.push(make_chunk("cam-0", 5000));
        store.push(make_chunk("cam-0", 10000));
        let results = store.query(&QueryFilter {
            time_start_ms: Some(3000),
            time_end_ms: Some(8000),
            camera_ids: None,
            top_k: 10,
            query_embedding: None,
        });
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].chunk.start_ts_ms, 5000);
        assert_eq!(results[1].chunk.start_ts_ms, 1000);
    }

    #[test]
    fn query_camera_filter_excludes_other_cameras() {
        let store = EmbeddingStore::new(100);
        store.push(make_chunk("cam-0", 1000));
        store.push(make_chunk("cam-1", 2000));
        store.push(make_chunk("cam-0", 3000));
        let results = store.query(&QueryFilter {
            time_start_ms: None,
            time_end_ms: None,
            camera_ids: Some(vec!["cam-1".into()]),
            top_k: 10,
            query_embedding: None,
        });
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].chunk.camera_id, "cam-1");
    }

    #[test]
    fn push_evicts_oldest_when_over_max_size() {
        let store = EmbeddingStore::new(3);
        store.push(make_chunk("cam-0", 1000));
        store.push(make_chunk("cam-0", 2000));
        store.push(make_chunk("cam-0", 3000));
        store.push(make_chunk("cam-0", 4000));
        assert_eq!(store.len(), 3);
        let results = store.query(&no_filter(10));
        assert!(results.iter().all(|c| c.chunk.start_ts_ms >= 2000));
    }

    #[test]
    fn query_returns_most_recent_chunks_first_without_embedding() {
        let store = EmbeddingStore::new(100);
        store.push(make_chunk("cam-0", 1000));
        store.push(make_chunk("cam-0", 5000));
        store.push(make_chunk("cam-0", 3000));
        let results = store.query(&no_filter(10));
        assert_eq!(results[0].chunk.start_ts_ms, 5000);
        assert_eq!(results[2].chunk.start_ts_ms, 1000);
    }

    #[test]
    fn query_embedding_ranks_by_cosine_similarity() {
        let store = EmbeddingStore::new(100);
        // Chunk A: old but highly similar to query [1, 0, 0]
        store.push(make_chunk_with_embedding("cam-0", 1000, vec![1.0, 0.0, 0.0]));
        // Chunk B: newer but orthogonal to query
        store.push(make_chunk_with_embedding("cam-0", 9000, vec![0.0, 1.0, 0.0]));

        let results = store.query(&QueryFilter {
            time_start_ms: None,
            time_end_ms: None,
            camera_ids: None,
            top_k: 10,
            query_embedding: Some(vec![1.0, 0.0, 0.0]),
        });
        // Chunk A should rank first: cosine_sim=1.0 vs 0.0, enough to overcome recency gap
        assert_eq!(results[0].chunk.start_ts_ms, 1000);
    }

    #[test]
    fn query_embedding_cross_camera_ranking() {
        let store = EmbeddingStore::new(100);
        store.push(make_chunk_with_embedding("cam-entrance", 5000, vec![1.0, 0.0, 0.0]));
        store.push(make_chunk_with_embedding("cam-parking", 5000, vec![0.0, 1.0, 0.0]));
        store.push(make_chunk_with_embedding("cam-checkout", 5000, vec![0.9, 0.1, 0.0]));

        let results = store.query(&QueryFilter {
            time_start_ms: None,
            time_end_ms: None,
            camera_ids: None, // all cameras
            top_k: 10,
            query_embedding: Some(vec![1.0, 0.0, 0.0]),
        });
        // cam-entrance (exact match) should rank first, cam-checkout second, cam-parking last
        assert_eq!(results[0].chunk.camera_id, "cam-entrance");
        assert_eq!(results[2].chunk.camera_id, "cam-parking");
    }

    #[test]
    fn cosine_similarity_identical_vectors() {
        assert!((cosine_similarity(&[1.0, 0.0], &[1.0, 0.0]) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_similarity_orthogonal_vectors() {
        assert!(cosine_similarity(&[1.0, 0.0], &[0.0, 1.0]).abs() < 1e-6);
    }

    #[test]
    fn cosine_similarity_zero_vector() {
        assert_eq!(cosine_similarity(&[0.0, 0.0], &[1.0, 0.0]), 0.0);
    }
}
