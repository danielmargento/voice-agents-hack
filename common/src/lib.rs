//! Shared types & wire protocol for the iroh-based leader/follower system.
//!
//! The transport is QUIC via iroh. Each protocol is identified by an ALPN
//! constant. On a given bidirectional stream we exchange length-prefixed
//! `postcard`-encoded frames (4-byte little-endian length + payload).

use std::{fmt, str::FromStr};

use anyhow::{bail, Context, Result};
use iroh::NodeAddr;
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};

/// ALPN for the long-lived ingest stream (follower -> leader pushes data).
pub const INGEST_ALPN: &[u8] = b"cactus/ingest/v1";

/// Hard cap on a single frame to protect against malformed peers. 16 MiB is
/// far larger than any embedding/caption payload we expect.
pub const MAX_FRAME_BYTES: usize = 16 * 1024 * 1024;

/// One unit of work shipped from a follower to the leader.
///
/// In the real system this carries the multimodal embedding for a short
/// video/audio chunk. The fields here mirror the PRD's `EmbeddingChunk`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingChunk {
    /// Stable id (e.g. `blake3(camera_id || start_ts)`). Used for dedupe.
    pub chunk_id: String,
    pub camera_id: String,
    /// Unix epoch milliseconds.
    pub start_ts_ms: u64,
    pub end_ts_ms: u64,
    /// Concatenated `[video || audio]` embedding, L2-normalized upstream.
    pub embedding: Vec<f32>,
    /// Dimensionality of the video portion of the embedding vector.
    pub video_dim: usize,
    /// Dimensionality of the audio portion of the embedding vector (0 when
    /// audio is unavailable). `embedding.len() == video_dim + audio_dim`.
    pub audio_dim: usize,
    /// Optional one-sentence caption for hybrid retrieval.
    pub caption: Option<String>,
}

/// A raw video + audio segment shipped from a follower to the leader for
/// persistent storage (replay). Each segment corresponds to one chunk
/// window and carries every captured JPEG frame plus the PCM audio.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VideoSegment {
    /// Unique id, typically `<camera_id>-<seq>`.
    pub segment_id: String,
    pub camera_id: String,
    /// Unix epoch milliseconds.
    pub start_ts_ms: u64,
    pub end_ts_ms: u64,
    /// Ordered JPEG-encoded frames captured during this window.
    pub jpeg_frames: Vec<Vec<u8>>,
    /// Mono f32 PCM audio at `audio_sample_rate` Hz.
    pub audio_samples: Vec<f32>,
    /// Sample rate of `audio_samples` (typically 16 000).
    pub audio_sample_rate: u32,
}

/// Frames sent follower -> leader on `INGEST_ALPN`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FollowerMsg {
    /// Sent once at connection start so the leader can register the follower.
    Hello {
        camera_id: String,
    },
    Chunk(EmbeddingChunk),
    /// Raw video + audio segment for persistent storage / replay.
    Video(VideoSegment),
    /// JPEG snapshot returned in response to a `LeaderMsg::FrameRequest`.
    /// Carries the latest captured frame from the follower's webcam, encoded
    /// as JPEG so the leader can serve it directly to HTTP clients.
    FrameResponse {
        req_id: u64,
        ts_ms: u64,
        width: u32,
        height: u32,
        jpeg: Vec<u8>,
    },
    /// Sent in place of `FrameResponse` when the follower has no frame
    /// available (camera not yet ready, encode failed, etc).
    FrameError {
        req_id: u64,
        message: String,
    },
    /// Graceful shutdown signal.
    Bye,
}

/// Frames sent leader -> follower on `INGEST_ALPN`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LeaderMsg {
    /// Acknowledges receipt + persistence of a chunk by id.
    Ack { chunk_id: String },
    /// Acknowledges receipt + storage of a video segment.
    VideoAck { segment_id: String },
    /// Asks the follower to send back its latest webcam frame as JPEG.
    /// The follower's response carries the same `req_id` so the leader can
    /// route it back to the waiting HTTP request.
    FrameRequest { req_id: u64 },
}

/// Connection ticket: everything a follower needs to reach the leader.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Ticket {
    pub leader: NodeAddr,
}

impl Ticket {
    pub fn new(leader: NodeAddr) -> Self {
        Self { leader }
    }

    fn to_bytes(&self) -> Vec<u8> {
        postcard::to_allocvec(self).expect("postcard ticket encode is infallible")
    }

    fn from_bytes(bytes: &[u8]) -> Result<Self> {
        postcard::from_bytes(bytes).context("decode ticket")
    }
}

impl fmt::Display for Ticket {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut text = data_encoding::BASE32_NOPAD.encode(&self.to_bytes());
        text.make_ascii_lowercase();
        f.write_str(&text)
    }
}

impl FromStr for Ticket {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self> {
        let bytes = data_encoding::BASE32_NOPAD
            .decode(s.to_ascii_uppercase().as_bytes())
            .context("ticket is not valid base32")?;
        Self::from_bytes(&bytes)
    }
}

/// Write a single length-prefixed `postcard` frame.
pub async fn write_frame<W, T>(w: &mut W, value: &T) -> Result<()>
where
    W: AsyncWrite + Unpin,
    T: Serialize,
{
    let bytes = postcard::to_allocvec(value).context("encode frame")?;
    if bytes.len() > MAX_FRAME_BYTES {
        bail!("frame too large: {} bytes", bytes.len());
    }
    w.write_all(&(bytes.len() as u32).to_le_bytes()).await?;
    w.write_all(&bytes).await?;
    w.flush().await?;
    Ok(())
}

/// Read a single length-prefixed `postcard` frame. Returns `Ok(None)` on a
/// clean EOF before any bytes of the next frame.
pub async fn read_frame<R, T>(r: &mut R) -> Result<Option<T>>
where
    R: AsyncRead + Unpin,
    T: for<'de> Deserialize<'de>,
{
    let mut len_buf = [0u8; 4];
    match r.read_exact(&mut len_buf).await {
        Ok(_) => {}
        Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(None),
        Err(e) => return Err(e.into()),
    }
    let len = u32::from_le_bytes(len_buf) as usize;
    if len > MAX_FRAME_BYTES {
        bail!("incoming frame too large: {len} bytes (hex: {len_buf:02x?})");
    }
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf).await?;
    let value = postcard::from_bytes(&buf).context("decode frame")?;
    Ok(Some(value))
}
