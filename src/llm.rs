//! LLM interaction types — the shared contract for model-agnostic design.
//!
//! These types define the conversation model, content blocks, and provider trait
//! that pulse-null and its plugins use to interact with language models.

use std::future::Future;
use std::pin::Pin;

/// Result type for LLM provider invocations
pub type LlmResult<'a> = Pin<
    Box<
        dyn Future<Output = Result<LlmResponse, Box<dyn std::error::Error + Send + Sync>>>
            + Send
            + 'a,
    >,
>;

/// A content block in a message or response.
/// Claude API uses tagged unions — each block has a "type" field.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type")]
pub enum ContentBlock {
    #[serde(rename = "text")]
    Text { text: String },

    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },

    #[serde(rename = "tool_result")]
    ToolResult {
        tool_use_id: String,
        content: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        is_error: Option<bool>,
    },
}

/// Message content can be a simple string or structured content blocks.
/// The Claude API accepts both formats.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Blocks(Vec<ContentBlock>),
}

/// Why the model stopped generating.
#[derive(Debug, Clone, PartialEq)]
pub enum StopReason {
    EndTurn,
    ToolUse,
    MaxTokens,
    StopSequence,
    Other(String),
}

/// Response from an LLM invocation
#[derive(Debug, Clone)]
pub struct LlmResponse {
    pub content: Vec<ContentBlock>,
    pub stop_reason: StopReason,
    pub model: String,
    pub input_tokens: Option<u32>,
    pub output_tokens: Option<u32>,
}

impl LlmResponse {
    /// Extract all text content from the response, concatenated.
    pub fn text(&self) -> String {
        self.content
            .iter()
            .filter_map(|block| match block {
                ContentBlock::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("")
    }

    /// Check if the response contains any tool_use blocks.
    pub fn has_tool_use(&self) -> bool {
        self.content
            .iter()
            .any(|block| matches!(block, ContentBlock::ToolUse { .. }))
    }
}

/// A message in a conversation
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: MessageContent,
}

/// Conversation role
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    User,
    Assistant,
}

/// Trait for LLM providers — the core abstraction for model-agnostic design
pub trait LmProvider: Send + Sync {
    /// Send a message and get a response.
    /// `tools` is an optional slice of tool definitions (JSON objects).
    fn invoke(
        &self,
        system_prompt: &str,
        messages: &[Message],
        max_tokens: u32,
        tools: Option<&[serde_json::Value]>,
    ) -> LlmResult<'_>;

    /// Provider name
    fn name(&self) -> &str;

    /// Whether this provider supports tool use
    fn supports_tools(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn content_block_text_serializes() {
        let block = ContentBlock::Text {
            text: "hello".into(),
        };
        let json = serde_json::to_string(&block).unwrap();
        assert!(json.contains("\"type\":\"text\""));
        assert!(json.contains("\"text\":\"hello\""));
    }

    #[test]
    fn content_block_tool_use_serializes() {
        let block = ContentBlock::ToolUse {
            id: "t1".into(),
            name: "file_read".into(),
            input: serde_json::json!({"path": "/tmp/test"}),
        };
        let json = serde_json::to_string(&block).unwrap();
        assert!(json.contains("\"type\":\"tool_use\""));
        assert!(json.contains("\"name\":\"file_read\""));
    }

    #[test]
    fn content_block_tool_result_serializes() {
        let block = ContentBlock::ToolResult {
            tool_use_id: "t1".into(),
            content: "file contents".into(),
            is_error: None,
        };
        let json = serde_json::to_string(&block).unwrap();
        assert!(json.contains("\"type\":\"tool_result\""));
        assert!(!json.contains("is_error")); // skipped when None
    }

    #[test]
    fn message_content_text_roundtrip() {
        let content = MessageContent::Text("hello".into());
        let json = serde_json::to_string(&content).unwrap();
        let back: MessageContent = serde_json::from_str(&json).unwrap();
        matches!(back, MessageContent::Text(s) if s == "hello");
    }

    #[test]
    fn message_content_blocks_roundtrip() {
        let content = MessageContent::Blocks(vec![ContentBlock::Text {
            text: "hello".into(),
        }]);
        let json = serde_json::to_string(&content).unwrap();
        let back: MessageContent = serde_json::from_str(&json).unwrap();
        matches!(back, MessageContent::Blocks(b) if b.len() == 1);
    }

    #[test]
    fn llm_response_text_extraction() {
        let response = LlmResponse {
            content: vec![
                ContentBlock::Text {
                    text: "hello ".into(),
                },
                ContentBlock::ToolUse {
                    id: "t1".into(),
                    name: "test".into(),
                    input: serde_json::Value::Null,
                },
                ContentBlock::Text {
                    text: "world".into(),
                },
            ],
            stop_reason: StopReason::EndTurn,
            model: "test".into(),
            input_tokens: None,
            output_tokens: None,
        };
        assert_eq!(response.text(), "hello world");
        assert!(response.has_tool_use());
    }

    #[test]
    fn llm_response_no_tool_use() {
        let response = LlmResponse {
            content: vec![ContentBlock::Text {
                text: "just text".into(),
            }],
            stop_reason: StopReason::EndTurn,
            model: "test".into(),
            input_tokens: None,
            output_tokens: None,
        };
        assert!(!response.has_tool_use());
    }

    #[test]
    fn stop_reason_equality() {
        assert_eq!(StopReason::EndTurn, StopReason::EndTurn);
        assert_ne!(StopReason::EndTurn, StopReason::ToolUse);
        assert_eq!(
            StopReason::Other("custom".into()),
            StopReason::Other("custom".into())
        );
    }

    #[test]
    fn message_serializes() {
        let msg = Message {
            role: Role::User,
            content: MessageContent::Text("hi".into()),
        };
        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains("\"role\":\"user\""));
    }
}
