//! Leader: accepts ingest connections from any number of followers, prints
//! a ticket on startup, and logs every chunk it receives.
//!
//! Exposes an axum HTTP server (default `127.0.0.1:8080`) for the UI:
//! - `GET /api/cameras`        — list registered followers + status
//! - `GET /api/live/:camera_id` — JPEG snapshot fetched on-demand from the
//!   follower over the existing iroh connection.
//!
//! When built with `--features cactus`, the leader also loads a Gemma
//! instance at startup (see `src/cactus/`) and exposes `POST /api/query`
//! that runs chat completion against it. Without the feature, that
//! endpoint returns an echo response so the UI still works.

#[cfg(feature = "cactus")]
mod cactus;
#[cfg(feature = "cactus")]
mod query;
mod store;

use std::{
    collections::HashMap,
    net::SocketAddr,
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, RwLock,
    },
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, Result};
use axum::{
    extract::{Path as AxPath, State},
    http::{header, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use clap::Parser;
use common::{read_frame, write_frame, FollowerMsg, LeaderMsg, Ticket, INGEST_ALPN};
use iroh::{
    endpoint::Connection,
    protocol::{AcceptError, ProtocolHandler, Router as IrohRouter},
    Endpoint, SecretKey, Watcher,
};
use serde::{Deserialize, Serialize};
use tokio::sync::{mpsc, oneshot};

#[cfg(feature = "cactus")]
use crate::cactus::CactusModel;
#[cfg(feature = "cactus")]
use crate::query::CactusQueryHandler;
use crate::store::{EmbeddingStore, QueryFilter};
use tower_http::cors::{Any, CorsLayer};
use tracing::{debug, error, info, warn};

#[derive(Parser, Debug)]
#[command(about = "iroh leader: accepts data from followers")]
struct Args {
    /// Path to a file holding the leader's 32-byte secret key as hex.
    /// Created with mode 0600 on first run if missing. Pin this file to keep
    /// your node id stable across restarts.
    #[arg(long, env = "LEADER_KEY_FILE", default_value = ".leader.key")]
    key_file: PathBuf,

    /// Path where the dialable ticket is written on startup. Followers in the
    /// same directory will pick it up automatically.
    #[arg(long, env = "LEADER_TICKET_FILE", default_value = ".leader.ticket")]
    ticket_file: PathBuf,

    /// Filter logs (default `info`).
    #[arg(long, env = "RUST_LOG", default_value = "info")]
    log: String,

    /// Address to bind the HTTP server the UI talks to.
    #[arg(long, env = "LEADER_HTTP_ADDR", default_value = "127.0.0.1:8080")]
    http_addr: SocketAddr,

    /// How long to wait for a follower's frame response before giving up.
    #[arg(long, default_value_t = 2000)]
    frame_timeout_ms: u64,

    /// Path to the Cactus-converted Gemma weights directory. When the
    /// `cactus` feature is on, this model is loaded at startup and used
    /// to answer `POST /api/query`. Ignored without the feature.
    #[arg(
        long,
        env = "GEMMA_MODEL_PATH",
        default_value = "/Users/danielargento/cactus/weights/gemma-4-e2b-it"
    )]
    model_path: PathBuf,

    /// Max embedding chunks to retain in memory (older chunks evicted first).
    #[arg(long, env = "STORE_MAX_SIZE", default_value_t = 10_000)]
    store_max_size: usize,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Pull values from `.env` if present; real env always wins.
    let _ = dotenvy::dotenv();

    let args = Args::parse();
    tracing_subscriber::fmt()
        .with_env_filter(&args.log)
        .with_target(false)
        .init();

    let secret_key = load_or_create_key(&args.key_file)?;

    let endpoint = Endpoint::builder()
        .secret_key(secret_key)
        .discovery_n0()
        .alpns(vec![INGEST_ALPN.to_vec()])
        .bind()
        .await?;

    let id = endpoint.node_id();
    info!(node_id = %id, key_file = %args.key_file.display(), "leader endpoint bound");

    // Wait for a relay URL to be established so the ticket works across networks.
    let relay = endpoint.home_relay().initialized().await;
    info!(%relay, "relay established");

    // Now grab the full NodeAddr (includes the relay URL + direct addrs).
    let addr = endpoint.node_addr().initialized().await;

    let ticket = Ticket::new(addr);
    let ticket_str = ticket.to_string();
    std::fs::write(&args.ticket_file, &ticket_str)
        .with_context(|| format!("write ticket file {}", args.ticket_file.display()))?;

    println!("\n  leader ready");
    println!("  endpoint id: {id}");
    println!("  ticket file: {}", args.ticket_file.display());
    println!("  http on:     http://{}", args.http_addr);
    println!("  ticket:\n\n{ticket_str}\n");
    println!("  == HOW TO CONNECT ==");
    println!("  1. From this computer (Local):");
    println!("     cargo run --release -p follower -- --camera-id cam-local");
    println!("  2. From another computer (Remote):");
    println!("     cargo run --release -p follower -- {ticket_str} --camera-id cam-partner\n");

    let store = Arc::new(EmbeddingStore::new(args.store_max_size));

    // Load Gemma up front. If it fails, log and keep serving — the rest
    // of the leader (camera registry, live snapshots) still works.
    #[cfg(feature = "cactus")]
    let llm = {
        let path = args.model_path.clone();
        info!(path = %path.display(), "loading gemma-4 via cactus");
        match tokio::task::spawn_blocking(move || CactusModel::load(&path)).await {
            Ok(Ok(m)) => {
                info!("gemma-4 loaded");
                Some(Arc::new(m))
            }
            Ok(Err(e)) => {
                warn!(error = %e, "cactus load failed; /api/query will return an error");
                None
            }
            Err(e) => {
                warn!(error = %e, "cactus load task panicked");
                None
            }
        }
    };

    let gemini_api_key = std::env::var("GEMINI_API_KEY").ok();
    if gemini_api_key.is_some() {
        info!("GEMINI_API_KEY set — semantic query embedding enabled");
    } else {
        info!("GEMINI_API_KEY not set — query will use recency-only ranking");
    }

    // Shared state used by both the iroh ingest handler and the HTTP server.
    let app_state = AppState {
        registry: Arc::new(RwLock::new(HashMap::new())),
        next_req_id: Arc::new(AtomicU64::new(1)),
        chunks_total: Arc::new(AtomicU64::new(0)),
        frame_timeout: Duration::from_millis(args.frame_timeout_ms),
        store,
        #[cfg(feature = "cactus")]
        llm,
        gemini_api_key,
    };

    let handler = IngestHandler {
        state: app_state.clone(),
    };
    let iroh_router = IrohRouter::builder(endpoint)
        .accept(INGEST_ALPN, handler)
        .spawn();

    // Spawn the HTTP server in the background.
    let http_state = app_state.clone();
    let http_addr = args.http_addr;
    let http_task = tokio::spawn(async move {
        if let Err(e) = serve_http(http_addr, http_state).await {
            error!(%e, "http server exited with error");
        }
    });

    wait_for_shutdown().await?;
    info!("shutting down");
    let _ = std::fs::remove_file(&args.ticket_file);
    http_task.abort();
    iroh_router.shutdown().await?;
    Ok(())
}

/// Resolves on Ctrl-C (SIGINT) or SIGTERM so the ticket file gets cleaned up
/// regardless of how the process is asked to exit.
async fn wait_for_shutdown() -> Result<()> {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{signal, SignalKind};
        let mut term = signal(SignalKind::terminate())?;
        tokio::select! {
            _ = tokio::signal::ctrl_c() => {}
            _ = term.recv() => {}
        }
        Ok(())
    }
    #[cfg(not(unix))]
    {
        tokio::signal::ctrl_c().await?;
        Ok(())
    }
}

/// Load a hex-encoded 32-byte secret key from `path`, or generate a fresh one
/// and persist it there (mode 0600 on unix).
fn load_or_create_key(path: &Path) -> Result<SecretKey> {
    if path.exists() {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("read key file {}", path.display()))?;
        let bytes = data_encoding::HEXLOWER_PERMISSIVE
            .decode(text.trim().as_bytes())
            .with_context(|| format!("key file {} is not valid hex", path.display()))?;
        let arr: [u8; 32] = bytes
            .as_slice()
            .try_into()
            .with_context(|| format!("key file {} must decode to 32 bytes", path.display()))?;
        info!(path = %path.display(), "loaded secret key");
        Ok(SecretKey::from_bytes(&arr))
    } else {
        let sk = SecretKey::generate(&mut rand_core_06::OsRng);
        let encoded = data_encoding::HEXLOWER.encode(&sk.to_bytes());
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent).ok();
            }
        }
        std::fs::write(path, encoded)
            .with_context(|| format!("write key file {}", path.display()))?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))
                .with_context(|| format!("chmod 600 {}", path.display()))?;
        }
        info!(path = %path.display(), "generated new secret key");
        Ok(sk)
    }
}

// ──────────────────────────── shared state ────────────────────────────

/// Live registration entry per camera. Created on `Hello`, removed on
/// follower disconnect. The `request_tx` lets the HTTP layer push frame
/// requests at the iroh task that owns the bidi stream.
#[derive(Clone)]
struct CameraEntry {
    request_tx: mpsc::Sender<FrameReq>,
    follower_node_id: String,
    last_seen_ms: Arc<AtomicU64>,
    chunks_total: Arc<AtomicU64>,
    connected_at_ms: u64,
}

struct FrameReq {
    req_id: u64,
    response_tx: oneshot::Sender<FrameOutcome>,
}

enum FrameOutcome {
    Ok(FrameSnapshot),
    Err(String),
}

struct FrameSnapshot {
    jpeg: Vec<u8>,
    #[allow(dead_code)]
    width: u32,
    #[allow(dead_code)]
    height: u32,
    #[allow(dead_code)]
    ts_ms: u64,
}

#[derive(Clone)]
struct AppState {
    registry: Arc<RwLock<HashMap<String, CameraEntry>>>,
    next_req_id: Arc<AtomicU64>,
    chunks_total: Arc<AtomicU64>,
    frame_timeout: Duration,
    store: Arc<EmbeddingStore>,
    #[cfg(feature = "cactus")]
    llm: Option<Arc<CactusModel>>,
    gemini_api_key: Option<String>,
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

// ──────────────────────────── iroh handler ────────────────────────────

#[derive(Debug, Clone)]
struct IngestHandler {
    state: AppState,
}

// AppState contains an RwLock + Arc<AtomicU64>; not Debug. Hand-roll Debug
// for the State field of IngestHandler.
impl std::fmt::Debug for AppState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AppState").finish_non_exhaustive()
    }
}

impl ProtocolHandler for IngestHandler {
    async fn accept(&self, conn: Connection) -> Result<(), AcceptError> {
        let remote = conn
            .remote_node_id()
            .map(|id| id.to_string())
            .unwrap_or_else(|_| "<unknown>".to_string());
        info!(%remote, "follower connected");

        if let Err(err) = self.serve(conn, remote.clone()).await {
            warn!(%remote, %err, "follower session ended with error");
        } else {
            info!(%remote, "follower disconnected");
        }
        Ok(())
    }
}

impl IngestHandler {
    async fn serve(&self, conn: Connection, remote: String) -> Result<()> {
        loop {
            let (send, recv) = match conn.accept_bi().await {
                Ok(s) => s,
                Err(_) => return Ok(()), // connection closed
            };

            let state = self.state.clone();
            let remote = remote.clone();
            tokio::spawn(async move {
                if let Err(e) = serve_stream(send, recv, state, remote.clone()).await {
                    warn!(%remote, %e, "stream task ended with error");
                }
            });
        }
    }
}

/// Owns one bidi stream and multiplexes:
/// - inbound `FollowerMsg` (Hello / Chunk / FrameResponse / FrameError / Bye)
/// - outbound `LeaderMsg` (Ack / FrameRequest), driven by:
///   - chunk acks fired from the same loop
///   - frame requests pushed by HTTP handlers via `mpsc`
async fn serve_stream(
    mut send: iroh::endpoint::SendStream,
    mut recv: iroh::endpoint::RecvStream,
    state: AppState,
    remote: String,
) -> Result<()> {
    let (req_tx, mut req_rx) = mpsc::channel::<FrameReq>(64);
    let (outbound_tx, mut outbound_rx) = mpsc::channel::<LeaderMsg>(128);

    let writer_task = tokio::spawn(async move {
        while let Some(msg) = outbound_rx.recv().await {
            if let Err(e) = write_frame(&mut send, &msg).await {
                error!(%e, "writer task send failed - network connection might have dropped concurrently");
                break;
            }
        }
        let _ = send.finish();
    });

    let (inbound_tx, mut inbound_rx) = mpsc::channel(64);
    let reader_task = tokio::spawn(async move {
        loop {
            let res = read_frame::<_, FollowerMsg>(&mut recv).await;
            let is_err_or_eof = res.is_err() || matches!(res, Ok(None));
            if inbound_tx.send(res).await.is_err() || is_err_or_eof {
                break;
            }
        }
    });

    let mut pending: HashMap<u64, oneshot::Sender<FrameOutcome>> = HashMap::new();
    let mut camera_id: Option<String> = None;
    let mut entry: Option<CameraEntry> = None;

    loop {
        tokio::select! {
            // Inbound traffic from the follower.
            opt = inbound_rx.recv() => {
                let Some(msg_res) = opt else { break; };
                let msg = match msg_res {
                    Ok(Some(m)) => m,
                    Ok(None) => {
                        debug!("clean EOF from follower stream");
                        break;
                    }
                    Err(e) => {
                        let cid_str = camera_id.as_deref().unwrap_or("<unknown>");
                        error!(camera_id = %cid_str, %remote, %e, "stream read failed - remote connection likely severed or frame excessively large");
                        break;
                    }
                };
                match msg {
                    FollowerMsg::Hello { camera_id: cid } => {
                        info!(camera_id = %cid, %remote, "hello");
                        let new_entry = CameraEntry {
                            request_tx: req_tx.clone(),
                            follower_node_id: remote.clone(),
                            last_seen_ms: Arc::new(AtomicU64::new(now_ms())),
                            chunks_total: Arc::new(AtomicU64::new(0)),
                            connected_at_ms: now_ms(),
                        };
                        state
                            .registry
                            .write()
                            .expect("registry poisoned")
                            .insert(cid.clone(), new_entry.clone());
                        camera_id = Some(cid);
                        entry = Some(new_entry);
                    }
                    FollowerMsg::Chunk(chunk) => {
                        let n = state.chunks_total.fetch_add(1, Ordering::Relaxed) + 1;
                        if let Some(e) = entry.as_ref() {
                            e.chunks_total.fetch_add(1, Ordering::Relaxed);
                            e.last_seen_ms.store(now_ms(), Ordering::Relaxed);
                        }
                        state.store.push(chunk.clone());
                        info!(
                            total = n,
                            camera = %chunk.camera_id,
                            chunk = %chunk.chunk_id,
                            dim = chunk.embedding.len(),
                            video_dim = chunk.video_dim,
                            audio_dim = chunk.audio_dim,
                            caption = chunk.caption.as_deref().unwrap_or(""),
                            "recv chunk",
                        );
                        let ack = LeaderMsg::Ack { chunk_id: chunk.chunk_id };
                        if outbound_tx.send(ack).await.is_err() {
                            break;
                        }
                    }
                    FollowerMsg::FrameResponse { req_id, ts_ms, width, height, jpeg } => {
                        if let Some(e) = entry.as_ref() {
                            e.last_seen_ms.store(now_ms(), Ordering::Relaxed);
                        }
                        if let Some(tx) = pending.remove(&req_id) {
                            let _ = tx.send(FrameOutcome::Ok(FrameSnapshot {
                                jpeg, width, height, ts_ms,
                            }));
                        } else {
                            debug!(req_id, "frame response with no waiter (timed out?)");
                        }
                    }
                    FollowerMsg::FrameError { req_id, message } => {
                        if let Some(tx) = pending.remove(&req_id) {
                            let _ = tx.send(FrameOutcome::Err(message));
                        }
                    }
                    FollowerMsg::Bye => {
                        if let Some(cid) = &camera_id {
                            info!(camera = %cid, "bye");
                        }
                        break;
                    }
                }
            }
            // Outbound: HTTP layer asked for a frame.
            Some(req) = req_rx.recv() => {
                pending.insert(req.req_id, req.response_tx);
                if outbound_tx.send(LeaderMsg::FrameRequest { req_id: req.req_id }).await.is_err() {
                    break;
                }
            }
        }
    }

    writer_task.abort();
    reader_task.abort();

    // Cleanup: drop the registration so HTTP requests stop landing here, and
    // cancel any pending oneshots so callers see an error instead of hanging.
    if let Some(cid) = &camera_id {
        let mut reg = state.registry.write().expect("registry poisoned");
        if let Some(existing) = reg.get(cid) {
            if Arc::ptr_eq(
                &existing.last_seen_ms,
                &entry.as_ref().unwrap().last_seen_ms,
            ) {
                reg.remove(cid);
                info!(camera = %cid, "deregistered");
            }
        }
    }
    for (_, tx) in pending.drain() {
        let _ = tx.send(FrameOutcome::Err("follower disconnected".into()));
    }
    Ok(())
}

// ──────────────────────────── HTTP layer ─────────────────────────────

async fn serve_http(addr: SocketAddr, state: AppState) -> Result<()> {
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    let app = Router::new()
        .route("/api/cameras", get(list_cameras))
        .route("/api/live/:camera_id", get(live_jpg))
        .route("/api/query", post(query))
        .layer(cors)
        .with_state(state);

    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .with_context(|| format!("bind http {addr}"))?;
    info!(%addr, "http server listening");
    axum::serve(listener, app).await.context("axum serve")?;
    Ok(())
}

#[derive(Serialize)]
struct CameraJson {
    id: String,
    follower_node_id: String,
    status: &'static str,
    last_seen_ms: u64,
    chunks_per_min: f64,
}

async fn list_cameras(State(state): State<AppState>) -> Json<Vec<CameraJson>> {
    let now = now_ms();
    let reg = state.registry.read().expect("registry poisoned");
    let mut out: Vec<CameraJson> = reg
        .iter()
        .map(|(id, e)| {
            let last_seen = e.last_seen_ms.load(Ordering::Relaxed);
            let age_ms = now.saturating_sub(last_seen);
            let status = if age_ms < 30_000 {
                "online"
            } else if age_ms < 120_000 {
                "degraded"
            } else {
                "offline"
            };
            let elapsed_min =
                ((now.saturating_sub(e.connected_at_ms)) as f64 / 60_000.0).max(1.0 / 60.0);
            let chunks = e.chunks_total.load(Ordering::Relaxed) as f64;
            CameraJson {
                id: id.clone(),
                follower_node_id: e.follower_node_id.clone(),
                status,
                last_seen_ms: last_seen,
                chunks_per_min: chunks / elapsed_min,
            }
        })
        .collect();
    out.sort_by(|a, b| a.id.cmp(&b.id));
    Json(out)
}

async fn live_jpg(State(state): State<AppState>, AxPath(camera_id): AxPath<String>) -> Response {
    let req_tx = {
        let reg = state.registry.read().expect("registry poisoned");
        match reg.get(&camera_id) {
            Some(e) => e.request_tx.clone(),
            None => {
                return (
                    StatusCode::NOT_FOUND,
                    format!("camera '{camera_id}' not online"),
                )
                    .into_response();
            }
        }
    };

    let req_id = state.next_req_id.fetch_add(1, Ordering::Relaxed);
    let (otx, orx) = oneshot::channel();
    if req_tx
        .send(FrameReq {
            req_id,
            response_tx: otx,
        })
        .await
        .is_err()
    {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            "follower request channel closed",
        )
            .into_response();
    }

    match tokio::time::timeout(state.frame_timeout, orx).await {
        Ok(Ok(FrameOutcome::Ok(snap))) => (
            [
                (header::CONTENT_TYPE, "image/jpeg"),
                (header::CACHE_CONTROL, "no-store, no-cache, must-revalidate"),
            ],
            snap.jpeg,
        )
            .into_response(),
        Ok(Ok(FrameOutcome::Err(msg))) => (StatusCode::BAD_GATEWAY, msg).into_response(),
        Ok(Err(_recv_err)) => (
            StatusCode::SERVICE_UNAVAILABLE,
            "follower closed before frame arrived",
        )
            .into_response(),
        Err(_) => (StatusCode::GATEWAY_TIMEOUT, "frame request timed out").into_response(),
    }
}

// ──────────────────────────── query (LLM) ────────────────────────────

#[derive(Deserialize)]
struct QueryReq {
    query: String,
    #[serde(default)]
    cameras: Option<Vec<String>>,
    #[serde(default)]
    top_k: Option<u32>,
}

#[derive(Serialize)]
struct QueryResp {
    answer: String,
    citations: Vec<serde_json::Value>,
    hits: Vec<serde_json::Value>,
}

#[cfg(feature = "cactus")]
async fn query(State(state): State<AppState>, Json(req): Json<QueryReq>) -> Response {
    let Some(model) = state.llm.clone() else {
        return (StatusCode::SERVICE_UNAVAILABLE, "gemma model not loaded").into_response();
    };
    if req.query.trim().is_empty() {
        return (StatusCode::BAD_REQUEST, "empty query").into_response();
    }

    let handler = CactusQueryHandler::new(model, state.gemini_api_key.clone());
    let now = now_ms();

    // Embed the query text into the same Gemini 3072-dim space as the stored
    // video embeddings for semantic nearest-neighbor retrieval. Falls back to
    // recency-only when no API key is configured.
    let query_embedding = handler.embed_query(&req.query).await;

    // Skip the LLM-based query parser — Gemma 4 E2B on CPU spends minutes
    // "thinking" before emitting JSON, which makes the UI hang. Use sane
    // defaults: last 30 minutes, all cameras, top_k=20 (overridable via
    // request body).
    let filter = QueryFilter {
        time_start_ms: Some(now.saturating_sub(30 * 60 * 1000)),
        time_end_ms: Some(now),
        camera_ids: req.cameras.clone(),
        top_k: req.top_k.map(|k| k as usize).unwrap_or(20),
        query_embedding,
    };
    let chunks = state.store.query(&filter);
    let n_chunks = chunks.len();
    info!(n_chunks, query = %req.query, "query: starting synthesis");

    let answer = match handler.synthesize_answer(&req.query, &chunks).await {
        Ok(a) => a,
        Err(e) => {
            error!(%e, "synthesize_answer failed");
            return (StatusCode::INTERNAL_SERVER_ERROR, format!("synthesis error: {e}"))
                .into_response();
        }
    };

    info!(n_chunks, "query answered");
    Json(QueryResp { answer, citations: Vec::new(), hits: Vec::new() }).into_response()
}

#[cfg(not(feature = "cactus"))]
async fn query(_state: State<AppState>, Json(req): Json<QueryReq>) -> Response {
    if req.query.trim().is_empty() {
        return (StatusCode::BAD_REQUEST, "empty query").into_response();
    }
    Json(QueryResp {
        answer: "Build with --features cactus to enable RAG query support.".into(),
        citations: Vec::new(),
        hits: Vec::new(),
    })
    .into_response()
}

#[cfg(test)]
mod tests {
    use crate::store::{EmbeddingStore, QueryFilter};
    use common::EmbeddingChunk;
    use std::sync::Arc;

    fn make_chunk(camera_id: &str, ts_ms: u64) -> EmbeddingChunk {
        EmbeddingChunk {
            chunk_id: format!("{camera_id}-{ts_ms}"),
            camera_id: camera_id.into(),
            start_ts_ms: ts_ms,
            end_ts_ms: ts_ms + 5000,
            embedding: vec![0.1, 0.2],
            video_dim: 0,
            audio_dim: 0,
            caption: None,
            representative_jpeg: None,
        }
    }

    #[test]
    fn store_accessible_and_chunks_persist() {
        let store = Arc::new(EmbeddingStore::new(1000));
        store.push(make_chunk("cam-0", 1000));
        store.push(make_chunk("cam-0", 6000));
        let results = store.query(&QueryFilter {
            time_start_ms: None,
            time_end_ms: None,
            camera_ids: None,
            top_k: 10,
            query_embedding: None,
        });
        assert_eq!(results.len(), 2);
    }
}
