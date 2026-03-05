//! Shared monitoring types and traits for the echo-system plugin ecosystem.
//!
//! This module defines the contract between pulse-null core and its monitoring
//! plugins (praxis-echo, vigil-echo, pulse-echo). Core calls through trait
//! objects; plugins implement the traits.

use std::path::Path;

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Pipeline types (praxis-echo domain)
// ---------------------------------------------------------------------------

/// Threshold configuration for pipeline documents.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineThresholds {
    pub learning_soft: usize,
    pub learning_hard: usize,
    pub thoughts_soft: usize,
    pub thoughts_hard: usize,
    pub curiosity_soft: usize,
    pub curiosity_hard: usize,
    pub reflections_soft: usize,
    pub reflections_hard: usize,
    pub praxis_soft: usize,
    pub praxis_hard: usize,
}

impl Default for PipelineThresholds {
    fn default() -> Self {
        Self {
            learning_soft: 5,
            learning_hard: 8,
            thoughts_soft: 5,
            thoughts_hard: 10,
            curiosity_soft: 3,
            curiosity_hard: 7,
            reflections_soft: 15,
            reflections_hard: 20,
            praxis_soft: 5,
            praxis_hard: 10,
        }
    }
}

/// Status of a document relative to its thresholds.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ThresholdStatus {
    Green,
    Yellow,
    Red,
}

impl std::fmt::Display for ThresholdStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Green => write!(f, "green"),
            Self::Yellow => write!(f, "yellow"),
            Self::Red => write!(f, "red"),
        }
    }
}

/// Health report for a single pipeline document.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentHealth {
    pub count: usize,
    pub soft: usize,
    pub hard: usize,
    pub status: ThresholdStatus,
}

/// Full pipeline health report across all documents.
#[derive(Debug, Clone)]
pub struct PipelineHealth {
    pub learning: DocumentHealth,
    pub thoughts: DocumentHealth,
    pub curiosity: DocumentHealth,
    pub reflections: DocumentHealth,
    pub praxis: DocumentHealth,
    pub warnings: Vec<String>,
}

/// Document entry counts for freeze detection.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct DocumentCounts {
    pub learning: usize,
    pub thoughts: usize,
    pub curiosity: usize,
    pub reflections: usize,
    pub praxis: usize,
}

/// Persistent pipeline state tracked across sessions.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PipelineState {
    pub last_updated: Option<String>,
    pub session_count: u32,
    pub sessions_without_movement: u32,
    pub last_counts: DocumentCounts,
}

impl PipelineState {
    /// Update counts and detect pipeline freezes.
    ///
    /// `now_iso` should be an ISO 8601 timestamp string (e.g. from `chrono::Utc::now().to_rfc3339()`).
    pub fn update_counts(&mut self, new_counts: &DocumentCounts, now_iso: &str) {
        if *new_counts == self.last_counts {
            self.sessions_without_movement += 1;
        } else {
            self.sessions_without_movement = 0;
        }
        self.last_counts = new_counts.clone();
        self.session_count += 1;
        self.last_updated = Some(now_iso.to_string());
    }
}

// ---------------------------------------------------------------------------
// Cognitive types (vigil-echo domain)
// ---------------------------------------------------------------------------

/// Cognitive health status levels.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CognitiveStatus {
    Healthy,
    Watch,
    Concern,
    Alert,
}

impl std::fmt::Display for CognitiveStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Healthy => write!(f, "HEALTHY"),
            Self::Watch => write!(f, "WATCH"),
            Self::Concern => write!(f, "CONCERN"),
            Self::Alert => write!(f, "ALERT"),
        }
    }
}

/// Signal trend direction.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Trend {
    Improving,
    Stable,
    Declining,
}

impl std::fmt::Display for Trend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Improving => write!(f, "improving"),
            Self::Stable => write!(f, "stable"),
            Self::Declining => write!(f, "declining"),
        }
    }
}

/// Full cognitive health assessment.
#[derive(Debug, Clone)]
pub struct CognitiveHealth {
    pub status: CognitiveStatus,
    pub vocabulary_trend: Trend,
    pub question_trend: Trend,
    pub evidence_trend: Trend,
    pub progress_trend: Trend,
    pub suggestions: Vec<String>,
    pub sufficient_data: bool,
}

/// A single frame of cognitive signals extracted from LLM output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalFrame {
    pub timestamp: String,
    pub task_id: String,
    pub vocabulary_diversity: f64,
    pub question_count: usize,
    pub evidence_references: usize,
    pub thought_progress: bool,
}

// ---------------------------------------------------------------------------
// Outcome types (pulse-echo domain)
// ---------------------------------------------------------------------------

/// A record of what happened during task execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutcomeRecord {
    pub task_id: String,
    pub timestamp: String,
    pub domain: String,
    pub task_type: String,
    pub description: String,
    pub outcome: String,
    pub tokens_used: u32,
    pub tool_rounds: u32,
}

// ---------------------------------------------------------------------------
// Traits
// ---------------------------------------------------------------------------

/// Pipeline document monitoring and archiving.
///
/// Implemented by praxis-echo. Used by pulse-null core for:
/// - Banner rendering (startup health bars)
/// - System prompt injection (pipeline state for LLM context)
/// - Post-execution hooks (update state, auto-archive)
/// - CLI commands (pipeline health, archive)
/// - Dashboard endpoint (JSON health)
pub trait PipelineMonitor: Send + Sync {
    /// Calculate pipeline health from document files on disk.
    fn calculate(&self, root_dir: &Path, thresholds: &PipelineThresholds) -> PipelineHealth;

    /// Render pipeline health as text for prompt injection.
    fn render_for_prompt(
        &self,
        health: &PipelineHealth,
        sessions_frozen: u32,
        freeze_threshold: u32,
    ) -> String;

    /// Extract document counts from a health report (for freeze detection).
    fn counts_from_health(&self, health: &PipelineHealth) -> DocumentCounts;

    /// Load persistent pipeline state from disk.
    fn load_state(&self, root_dir: &Path) -> PipelineState;

    /// Save persistent pipeline state to disk.
    fn save_state(
        &self,
        root_dir: &Path,
        state: &PipelineState,
    ) -> Result<(), Box<dyn std::error::Error>>;

    /// Check all documents against hard limits and auto-archive overflow.
    /// Returns list of document names that were archived.
    fn check_and_archive(
        &self,
        root_dir: &Path,
        thresholds: &PipelineThresholds,
        health: &PipelineHealth,
    ) -> Vec<String>;

    /// List archived files, optionally filtered by document name.
    fn list_archives(
        &self,
        root_dir: &Path,
        document: Option<&str>,
    ) -> Result<Vec<String>, Box<dyn std::error::Error>>;

    /// Manually archive a specific document by name.
    fn archive_by_name(
        &self,
        root_dir: &Path,
        document: &str,
    ) -> Result<String, Box<dyn std::error::Error>>;
}

/// Metacognitive signal tracking and health assessment.
///
/// Implemented by vigil-echo. Used by pulse-null core for:
/// - Banner rendering (cognitive health status + trend arrows)
/// - System prompt injection (assessment text for LLM context)
/// - Post-execution hooks (extract and record signals, detect changes)
/// - Dashboard endpoint (JSON signals)
pub trait CognitiveMonitor: Send + Sync {
    /// Perform a cognitive health assessment from signal history.
    fn assess(&self, root_dir: &Path, window_size: usize, min_samples: usize) -> CognitiveHealth;

    /// Render cognitive health as text for prompt injection.
    fn render_for_prompt(&self, health: &CognitiveHealth) -> String;

    /// Extract cognitive signals from LLM output text.
    fn extract(&self, content: &str, task_id: &str) -> SignalFrame;

    /// Append a signal frame to the rolling window on disk.
    fn record(
        &self,
        root_dir: &Path,
        frame: SignalFrame,
        window_size: usize,
    ) -> Result<(), Box<dyn std::error::Error>>;
}

/// Task execution outcome recording.
///
/// Implemented by pulse-echo. Used by pulse-null core for:
/// - Post-execution hooks (build and record outcomes after tasks/intents)
pub trait OutcomeTracker: Send + Sync {
    /// Build an outcome record from task execution results.
    fn build_outcome(
        &self,
        task_id: &str,
        task_name: &str,
        response_text: &str,
        tool_rounds: u32,
        input_tokens: u32,
        output_tokens: u32,
    ) -> OutcomeRecord;

    /// Record an outcome to persistent storage.
    fn record_outcome(
        &self,
        docs_dir: &Path,
        outcome: OutcomeRecord,
        max_outcomes: usize,
    ) -> Result<(), Box<dyn std::error::Error>>;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn threshold_status_display() {
        assert_eq!(ThresholdStatus::Green.to_string(), "green");
        assert_eq!(ThresholdStatus::Yellow.to_string(), "yellow");
        assert_eq!(ThresholdStatus::Red.to_string(), "red");
    }

    #[test]
    fn cognitive_status_display() {
        assert_eq!(CognitiveStatus::Healthy.to_string(), "HEALTHY");
        assert_eq!(CognitiveStatus::Watch.to_string(), "WATCH");
        assert_eq!(CognitiveStatus::Concern.to_string(), "CONCERN");
        assert_eq!(CognitiveStatus::Alert.to_string(), "ALERT");
    }

    #[test]
    fn trend_display() {
        assert_eq!(Trend::Improving.to_string(), "improving");
        assert_eq!(Trend::Stable.to_string(), "stable");
        assert_eq!(Trend::Declining.to_string(), "declining");
    }

    #[test]
    fn pipeline_thresholds_default() {
        let t = PipelineThresholds::default();
        assert_eq!(t.learning_soft, 5);
        assert_eq!(t.learning_hard, 8);
        assert_eq!(t.curiosity_hard, 7);
    }

    #[test]
    fn pipeline_state_update_counts_detects_freeze() {
        let mut state = PipelineState::default();
        let counts = DocumentCounts {
            learning: 3,
            thoughts: 2,
            curiosity: 1,
            reflections: 5,
            praxis: 2,
        };

        state.update_counts(&counts, "2026-03-05T12:00:00Z");
        assert_eq!(state.sessions_without_movement, 0);
        assert_eq!(state.session_count, 1);

        // Same counts — should increment freeze counter
        state.update_counts(&counts, "2026-03-05T13:00:00Z");
        assert_eq!(state.sessions_without_movement, 1);
        assert_eq!(state.session_count, 2);

        // Different counts — should reset freeze counter
        let new_counts = DocumentCounts {
            learning: 4,
            ..counts
        };
        state.update_counts(&new_counts, "2026-03-05T14:00:00Z");
        assert_eq!(state.sessions_without_movement, 0);
        assert_eq!(state.session_count, 3);
    }

    #[test]
    fn document_counts_default() {
        let counts = DocumentCounts::default();
        assert_eq!(counts.learning, 0);
        assert_eq!(counts.thoughts, 0);
    }

    #[test]
    fn threshold_status_equality() {
        assert_eq!(ThresholdStatus::Red, ThresholdStatus::Red);
        assert_ne!(ThresholdStatus::Red, ThresholdStatus::Green);
    }

    #[test]
    fn signal_frame_serializes() {
        let frame = SignalFrame {
            timestamp: "2026-03-05T12:00:00Z".to_string(),
            task_id: "test".to_string(),
            vocabulary_diversity: 0.72,
            question_count: 3,
            evidence_references: 5,
            thought_progress: true,
        };
        let json = serde_json::to_string(&frame).unwrap();
        let back: SignalFrame = serde_json::from_str(&json).unwrap();
        assert_eq!(back.task_id, "test");
        assert!((back.vocabulary_diversity - 0.72).abs() < f64::EPSILON);
    }

    #[test]
    fn outcome_record_serializes() {
        let record = OutcomeRecord {
            task_id: "task-1".to_string(),
            timestamp: "2026-03-05T12:00:00Z".to_string(),
            domain: "research".to_string(),
            task_type: "research".to_string(),
            description: "Deep dive".to_string(),
            outcome: "success".to_string(),
            tokens_used: 1500,
            tool_rounds: 3,
        };
        let json = serde_json::to_string(&record).unwrap();
        let back: OutcomeRecord = serde_json::from_str(&json).unwrap();
        assert_eq!(back.task_id, "task-1");
        assert_eq!(back.tokens_used, 1500);
    }
}
