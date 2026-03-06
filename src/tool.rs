//! Tool trait — the shared contract for LLM-callable tools.
//!
//! Plugins contribute tools via [`Plugin::tools()`](crate::plugin::Plugin::tools).
//! The host registers them and exposes them to the LLM.

use std::fmt;
use std::future::Future;
use std::pin::Pin;

/// Async result type for tool execution.
pub type ToolResult<'a> = Pin<Box<dyn Future<Output = Result<String, ToolError>> + Send + 'a>>;

/// Errors from tool execution.
#[derive(Debug)]
pub enum ToolError {
    /// The requested resource was not found.
    NotFound(String),
    /// Tool execution failed.
    ExecutionFailed(String),
    /// Access denied.
    PermissionDenied(String),
}

impl fmt::Display for ToolError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ToolError::NotFound(msg) => write!(f, "not found: {msg}"),
            ToolError::ExecutionFailed(msg) => write!(f, "execution failed: {msg}"),
            ToolError::PermissionDenied(msg) => write!(f, "permission denied: {msg}"),
        }
    }
}

impl std::error::Error for ToolError {}

/// A tool that can be invoked by an LLM.
///
/// Tools are contributed by plugins via [`Plugin::tools()`](crate::plugin::Plugin::tools)
/// and registered with the host's tool registry. The LLM calls tools by name.
///
/// # Object Safety
///
/// This trait is object-safe and designed for use as `Box<dyn Tool>`.
pub trait Tool: Send + Sync {
    /// Tool name — must match what the LLM calls.
    fn name(&self) -> &str;

    /// Human-readable description for the LLM.
    fn description(&self) -> &str;

    /// JSON Schema for the tool's input parameters.
    fn input_schema(&self) -> serde_json::Value;

    /// Execute the tool with the given input.
    fn execute(&self, input: serde_json::Value) -> ToolResult<'_>;
}
