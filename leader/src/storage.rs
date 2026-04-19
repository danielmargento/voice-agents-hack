//! Persistent video storage for follower recordings.
//!
//! Directory layout:
//!
//! ```text
//! <recordings_dir>/
//! └── <camera_id>/
//!     ├── <start_ts_ms>_<end_ts_ms>.mp4
//!     ├── <start_ts_ms>_<end_ts_ms>.mp4
//!     └── ...
//! ```
//!
//! Each `.mp4` is produced by muxing the received JPEG frames (as an
//! MJPEG video track) and PCM audio (re-encoded to AAC) via an `ffmpeg`
//! subprocess. When `ffmpeg` is not available we fall back to storing
//! the raw frames + audio as individual files under a segment directory.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use common::VideoSegment;
use serde::Serialize;
use tokio::process::Command;
use tracing::{info, warn};

/// Manages the on-disk recording directory.
#[derive(Clone)]
pub struct RecordingStore {
    base_dir: PathBuf,
}

/// Metadata about one stored recording.
#[derive(Debug, Clone, Serialize)]
pub struct RecordingInfo {
    pub camera_id: String,
    pub filename: String,
    pub start_ts_ms: u64,
    pub end_ts_ms: u64,
    pub size_bytes: u64,
}

impl RecordingStore {
    /// Create a new store rooted at `base_dir`. Creates the directory if
    /// it doesn't exist.
    pub fn new(base_dir: impl Into<PathBuf>) -> Result<Self> {
        let base_dir = base_dir.into();
        std::fs::create_dir_all(&base_dir)
            .with_context(|| format!("create recordings dir {}", base_dir.display()))?;
        info!(dir = %base_dir.display(), "recording store ready");
        Ok(Self { base_dir })
    }

    /// Persist a video segment from a follower. Returns the path to the
    /// written MP4 (or fallback directory).
    pub async fn store_segment(&self, seg: &VideoSegment) -> Result<PathBuf> {
        if seg.jpeg_frames.is_empty() {
            anyhow::bail!("video segment has no frames");
        }

        let cam_dir = self.base_dir.join(sanitize_path_component(&seg.camera_id));
        std::fs::create_dir_all(&cam_dir)?;

        let stem = format!("{}_{}", seg.start_ts_ms, seg.end_ts_ms);
        let mp4_path = cam_dir.join(format!("{stem}.mp4"));

        // Write frames + audio to a temp dir, then mux via ffmpeg.
        let tmp = tempfile::tempdir().context("create temp dir for mux")?;

        // --- Write JPEG frames ---
        for (i, jpeg) in seg.jpeg_frames.iter().enumerate() {
            let frame_path = tmp.path().join(format!("frame_{i:04}.jpg"));
            std::fs::write(&frame_path, jpeg)
                .with_context(|| format!("write frame {i}"))?;
        }

        // --- Write raw audio as f32le PCM ---
        let has_audio = !seg.audio_samples.is_empty();
        let audio_path = tmp.path().join("audio.pcm");
        if has_audio {
            let pcm_bytes: Vec<u8> = seg
                .audio_samples
                .iter()
                .flat_map(|s| s.to_le_bytes())
                .collect();
            std::fs::write(&audio_path, &pcm_bytes).context("write audio pcm")?;
        }

        // --- Try ffmpeg mux ---
        match mux_ffmpeg(tmp.path(), &mp4_path, seg, has_audio, &audio_path).await {
            Ok(()) => {
                info!(
                    camera = %seg.camera_id,
                    path = %mp4_path.display(),
                    frames = seg.jpeg_frames.len(),
                    audio_samples = seg.audio_samples.len(),
                    "segment stored as MP4",
                );
                Ok(mp4_path)
            }
            Err(e) => {
                warn!(%e, "ffmpeg mux failed, falling back to raw storage");
                self.store_raw_fallback(&cam_dir, &stem, seg)
            }
        }
    }

    /// Fallback: store JPEG frames + WAV audio as individual files when
    /// ffmpeg is unavailable.
    fn store_raw_fallback(
        &self,
        cam_dir: &Path,
        stem: &str,
        seg: &VideoSegment,
    ) -> Result<PathBuf> {
        let seg_dir = cam_dir.join(stem);
        std::fs::create_dir_all(&seg_dir)?;

        for (i, jpeg) in seg.jpeg_frames.iter().enumerate() {
            std::fs::write(seg_dir.join(format!("frame_{i:04}.jpg")), jpeg)?;
        }

        if !seg.audio_samples.is_empty() {
            write_wav(
                &seg_dir.join("audio.wav"),
                &seg.audio_samples,
                seg.audio_sample_rate,
            )?;
        }

        // Write a small manifest so we know timestamps.
        let manifest = serde_json::json!({
            "camera_id": seg.camera_id,
            "segment_id": seg.segment_id,
            "start_ts_ms": seg.start_ts_ms,
            "end_ts_ms": seg.end_ts_ms,
            "frame_count": seg.jpeg_frames.len(),
            "audio_sample_rate": seg.audio_sample_rate,
            "audio_samples": seg.audio_samples.len(),
        });
        std::fs::write(
            seg_dir.join("manifest.json"),
            serde_json::to_string_pretty(&manifest)?,
        )?;

        info!(
            camera = %seg.camera_id,
            path = %seg_dir.display(),
            "segment stored as raw files (ffmpeg unavailable)",
        );
        Ok(seg_dir)
    }

    /// List camera IDs that have at least one recording.
    pub fn list_cameras(&self) -> Result<Vec<String>> {
        let mut cameras = Vec::new();
        let entries = std::fs::read_dir(&self.base_dir).context("read recordings dir")?;
        for entry in entries {
            let entry = entry?;
            if entry.file_type()?.is_dir() {
                if let Some(name) = entry.file_name().to_str() {
                    cameras.push(name.to_string());
                }
            }
        }
        cameras.sort();
        Ok(cameras)
    }

    /// List all recordings for a specific camera, sorted by start time.
    pub fn list_recordings(&self, camera_id: &str) -> Result<Vec<RecordingInfo>> {
        let cam_dir = self.base_dir.join(sanitize_path_component(camera_id));
        if !cam_dir.is_dir() {
            return Ok(Vec::new());
        }

        let mut recordings = Vec::new();
        let entries = std::fs::read_dir(&cam_dir)?;
        for entry in entries {
            let entry = entry?;
            let name = entry.file_name().to_string_lossy().to_string();
            if name.ends_with(".mp4") {
                let size = entry.metadata().map(|m| m.len()).unwrap_or(0);
                let (start, end) = parse_segment_filename(&name);
                recordings.push(RecordingInfo {
                    camera_id: camera_id.to_string(),
                    filename: name,
                    start_ts_ms: start,
                    end_ts_ms: end,
                    size_bytes: size,
                });
            }
        }
        recordings.sort_by_key(|r| r.start_ts_ms);
        Ok(recordings)
    }

    /// Resolve a recording filename to its full path. Returns `None` if
    /// the file doesn't exist or the filename is suspicious.
    pub fn recording_path(&self, camera_id: &str, filename: &str) -> Option<PathBuf> {
        let clean_cam = sanitize_path_component(camera_id);
        let clean_file = sanitize_path_component(filename);
        let path = self.base_dir.join(clean_cam).join(clean_file);
        if path.is_file() {
            Some(path)
        } else {
            None
        }
    }
}

/// Run ffmpeg to mux JPEG frames + PCM audio into an MP4.
async fn mux_ffmpeg(
    tmp_dir: &Path,
    output: &Path,
    seg: &VideoSegment,
    has_audio: bool,
    audio_path: &Path,
) -> Result<()> {
    let n_frames = seg.jpeg_frames.len();
    let duration_s = (seg.end_ts_ms.saturating_sub(seg.start_ts_ms)) as f64 / 1000.0;
    let fps = if duration_s > 0.0 {
        (n_frames as f64 / duration_s).max(1.0)
    } else {
        1.0
    };

    let frames_glob = tmp_dir.join("frame_%04d.jpg");

    let mut cmd = Command::new("ffmpeg");
    cmd.arg("-y") // overwrite
        .arg("-framerate")
        .arg(format!("{fps:.2}"))
        .arg("-i")
        .arg(&frames_glob);

    if has_audio {
        cmd.arg("-f")
            .arg("f32le")
            .arg("-ar")
            .arg(seg.audio_sample_rate.to_string())
            .arg("-ac")
            .arg("1")
            .arg("-i")
            .arg(audio_path);
    }

    cmd.arg("-c:v")
        .arg("libx264")
        .arg("-pix_fmt")
        .arg("yuv420p")
        .arg("-preset")
        .arg("ultrafast");

    if has_audio {
        cmd.arg("-c:a").arg("aac").arg("-b:a").arg("128k");
    }

    cmd.arg("-shortest")
        .arg("-movflags")
        .arg("+faststart")
        .arg(output);

    let result = cmd
        .output()
        .await
        .context("spawn ffmpeg (is it installed?)")?;

    if !result.status.success() {
        let stderr = String::from_utf8_lossy(&result.stderr);
        anyhow::bail!("ffmpeg exited {}: {stderr}", result.status);
    }

    Ok(())
}

/// Write a simple WAV file from f32 samples (for the raw fallback).
fn write_wav(path: &Path, samples: &[f32], sample_rate: u32) -> Result<()> {
    use std::io::Write;

    let num_samples = samples.len() as u32;
    let byte_rate = sample_rate * 4; // 32-bit float, mono
    let data_size = num_samples * 4;
    let file_size = 36 + data_size;

    let mut f = std::fs::File::create(path)?;
    // RIFF header
    f.write_all(b"RIFF")?;
    f.write_all(&file_size.to_le_bytes())?;
    f.write_all(b"WAVE")?;
    // fmt chunk (IEEE float)
    f.write_all(b"fmt ")?;
    f.write_all(&16u32.to_le_bytes())?; // chunk size
    f.write_all(&3u16.to_le_bytes())?; // format: IEEE float
    f.write_all(&1u16.to_le_bytes())?; // channels
    f.write_all(&sample_rate.to_le_bytes())?;
    f.write_all(&byte_rate.to_le_bytes())?;
    f.write_all(&4u16.to_le_bytes())?; // block align
    f.write_all(&32u16.to_le_bytes())?; // bits per sample
    // data chunk
    f.write_all(b"data")?;
    f.write_all(&data_size.to_le_bytes())?;
    for s in samples {
        f.write_all(&s.to_le_bytes())?;
    }
    Ok(())
}

/// Strip any path-separator or traversal characters to prevent directory
/// traversal attacks.
fn sanitize_path_component(s: &str) -> String {
    s.replace(['/', '\\', '\0'], "")
        .replace("..", "")
        .trim()
        .to_string()
}

/// Parse `<start>_<end>.mp4` into (start_ts, end_ts). Returns (0, 0) on
/// parse failure.
fn parse_segment_filename(name: &str) -> (u64, u64) {
    let stem = name.trim_end_matches(".mp4");
    let mut parts = stem.splitn(2, '_');
    let start = parts
        .next()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let end = parts
        .next()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    (start, end)
}
