// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::OnceLock;

use dynamo_runtime::config::{
    env_is_truthy, environment_names::llm::agent_trace as env_agent_trace,
};

use super::DEFAULT_TOOL_EVENTS_TOPIC;

const DEFAULT_CAPACITY: usize = 1024;
const DEFAULT_JSONL_BUFFER_BYTES: usize = 1024 * 1024;
const DEFAULT_JSONL_FLUSH_INTERVAL_MS: u64 = 1000;

#[derive(Clone, Debug)]
pub struct AgentTracePolicy {
    pub enabled: bool,
    pub jsonl_path: Option<String>,
    pub capacity: usize,
    pub jsonl_buffer_bytes: usize,
    pub jsonl_flush_interval_ms: u64,
    pub tool_events_enabled: bool,
    pub tool_events_topic: String,
    pub tool_events_namespace: Option<String>,
    pub tool_events_zmq_endpoint: Option<String>,
    pub tool_events_zmq_topic: Option<String>,
}

static POLICY: OnceLock<AgentTracePolicy> = OnceLock::new();

fn load_from_env() -> AgentTracePolicy {
    let jsonl_path = std::env::var(env_agent_trace::DYN_AGENT_TRACE_JSONL)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty());
    let tool_events_topic = std::env::var(env_agent_trace::DYN_AGENT_TRACE_TOOL_EVENTS_TOPIC)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| DEFAULT_TOOL_EVENTS_TOPIC.to_string());
    let tool_events_namespace = std::env::var(env_agent_trace::DYN_AGENT_TRACE_NAMESPACE)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty());
    let tool_events_zmq_endpoint =
        std::env::var(env_agent_trace::DYN_AGENT_TRACE_TOOL_EVENTS_ZMQ_ENDPOINT)
            .ok()
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty());
    let tool_events_zmq_topic =
        std::env::var(env_agent_trace::DYN_AGENT_TRACE_TOOL_EVENTS_ZMQ_TOPIC)
            .ok()
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty());
    let tool_events_enabled = env_is_truthy(env_agent_trace::DYN_AGENT_TRACE_TOOL_EVENTS)
        || tool_events_zmq_endpoint.is_some();
    let capacity = std::env::var(env_agent_trace::DYN_AGENT_TRACE_CAPACITY)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(DEFAULT_CAPACITY);
    let jsonl_buffer_bytes = std::env::var(env_agent_trace::DYN_AGENT_TRACE_JSONL_BUFFER_BYTES)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(DEFAULT_JSONL_BUFFER_BYTES);
    let jsonl_flush_interval_ms =
        std::env::var(env_agent_trace::DYN_AGENT_TRACE_JSONL_FLUSH_INTERVAL_MS)
            .ok()
            .and_then(|value| value.parse::<u64>().ok())
            .unwrap_or(DEFAULT_JSONL_FLUSH_INTERVAL_MS);

    AgentTracePolicy {
        enabled: jsonl_path.is_some() || tool_events_enabled,
        jsonl_path,
        capacity,
        jsonl_buffer_bytes,
        jsonl_flush_interval_ms,
        tool_events_enabled,
        tool_events_topic,
        tool_events_namespace,
        tool_events_zmq_endpoint,
        tool_events_zmq_topic,
    }
}

pub fn policy() -> &'static AgentTracePolicy {
    POLICY.get_or_init(load_from_env)
}

pub fn is_enabled() -> bool {
    policy().enabled
}
