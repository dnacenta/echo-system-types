//! Plugin trait — the shared contract for pulse-null plugins.
//!
//! Plugins are constructed via async factory functions, not through the trait.
//! By the time a [`Plugin`] exists, it is fully initialized and ready for
//! [`start()`](Plugin::start).
//!
//! # Factory Pattern
//!
//! Each plugin crate exports a factory function:
//!
//! ```rust,ignore
//! pub async fn create(
//!     config: &serde_json::Value,
//!     ctx: &PluginContext,
//! ) -> Result<Box<dyn Plugin>, Box<dyn Error + Send + Sync>>
//! ```
//!
//! This avoids two-phase initialization — the plugin is either fully
//! constructed or the factory returns an error.
//!
//! # Object Safety
//!
//! The trait is object-safe and designed for `Box<dyn Plugin>`. Async methods
//! use `Pin<Box<dyn Future>>` instead of `async_trait`.

use std::any::Any;
use std::future::Future;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;

use crate::llm::LmProvider;
use crate::tool::Tool;
use crate::{HealthStatus, PluginMeta, ScheduledTask, SetupPrompt};

/// Async result type for plugin lifecycle operations.
pub type PluginResult<'a> =
    Pin<Box<dyn Future<Output = Result<(), Box<dyn std::error::Error + Send + Sync>>> + Send + 'a>>;

/// Context passed to plugin factories during construction.
///
/// Contains everything a plugin needs to initialize:
/// filesystem root, entity identity, and LLM access.
pub struct PluginContext {
    /// Root directory of the entity (e.g., `/home/echo`).
    pub entity_root: PathBuf,
    /// Entity name from config (e.g., `"Echo"`).
    pub entity_name: String,
    /// LLM provider for plugin use (summarization, analysis, etc.).
    pub provider: Arc<dyn LmProvider>,
}

/// Declares what role a plugin fills in the system.
///
/// Some roles are constrained:
/// - [`Memory`](PluginRole::Memory): exactly one required.
/// - All others: zero or more.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PluginRole {
    /// Persistent memory system. **Required — exactly one.**
    Memory,
    /// Pipeline state monitoring.
    Pipeline,
    /// Cognitive signal monitoring.
    Cognitive,
    /// Outcome tracking.
    Outcome,
    /// Communication interface (HTTP, voice, chat, etc.).
    Interface,
    /// General-purpose extension. No constraints.
    Extension,
}

/// The core plugin contract.
///
/// Plugins are constructed via async factory functions — by the time this
/// trait is available, the plugin is fully initialized and ready for
/// [`start()`](Plugin::start).
///
/// # Dyn-compatible
///
/// Uses `Pin<Box<dyn Future>>` for async methods. No `async_trait` dependency.
///
/// # Extension via `as_any()`
///
/// Host-specific capabilities (e.g., HTTP routes via axum) are not part of
/// this trait. Plugins that expose such capabilities implement [`as_any()`](Plugin::as_any)
/// to allow the host to downcast to the concrete type.
pub trait Plugin: Send + Sync {
    /// Plugin identity (name, version, description).
    fn meta(&self) -> PluginMeta;

    /// What role this plugin fills in the system.
    fn role(&self) -> PluginRole;

    /// Start the plugin. Called once after construction.
    fn start(&mut self) -> PluginResult<'_>;

    /// Stop the plugin gracefully.
    fn stop(&mut self) -> PluginResult<'_>;

    /// Report current health.
    fn health(&self) -> Pin<Box<dyn Future<Output = HealthStatus> + Send + '_>>;

    /// Optional: contribute scheduled tasks.
    fn scheduled_tasks(&self) -> Vec<ScheduledTask> {
        Vec::new()
    }

    /// Optional: setup wizard prompts for first-time configuration.
    fn setup_prompts(&self) -> Vec<SetupPrompt> {
        Vec::new()
    }

    /// Optional: contribute tools to the entity's tool registry.
    fn tools(&self) -> Vec<Box<dyn Tool>> {
        Vec::new()
    }

    /// Downcast support for host-specific extensions.
    ///
    /// Plugins that expose capabilities beyond this trait (e.g., axum routes)
    /// return `self` here so the host can downcast to the concrete type.
    fn as_any(&self) -> &dyn Any;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plugin_role_equality() {
        assert_eq!(PluginRole::Memory, PluginRole::Memory);
        assert_ne!(PluginRole::Memory, PluginRole::Pipeline);
    }

    #[test]
    fn plugin_role_is_copy() {
        let role = PluginRole::Memory;
        let copy = role;
        assert_eq!(role, copy);
    }

    #[test]
    fn plugin_role_debug() {
        let debug = format!("{:?}", PluginRole::Interface);
        assert_eq!(debug, "Interface");
    }
}
