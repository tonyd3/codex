use std::cmp::Reverse;
use std::collections::HashMap;
use std::collections::HashSet;
use std::path::Path;
use std::path::PathBuf;

use codex_core::INTERACTIVE_SESSION_SOURCES;
use codex_core::RolloutRecorder;
use codex_core::ThreadItem;
use codex_core::ThreadSortKey;
use codex_core::config::Config;
use codex_protocol::models::ContentItem;
use codex_protocol::models::MessagePhase;
use codex_protocol::models::ResponseItem;
use codex_protocol::protocol::EventMsg;
use codex_protocol::protocol::ExecCommandEndEvent;
use codex_protocol::protocol::ExecCommandStatus;
use codex_protocol::protocol::RolloutItem;

pub(crate) const DEFAULT_IMPROVE_SESSION_LIMIT: usize = 50;
const AGGREGATE_LIMIT: usize = 8;
const FRICTION_LIMIT: usize = 8;
const PREVIEW_CHAR_LIMIT: usize = 110;
const PHRASE_CHAR_LIMIT: usize = 120;

pub(crate) async fn build_improve_prompt(
    config: Config,
    current_rollout_path: Option<PathBuf>,
    limit: usize,
) -> Result<String, String> {
    let sessions = collect_recent_sessions(&config, current_rollout_path.as_deref(), limit).await?;
    if sessions.is_empty() {
        return Err("No previous interactive sessions were found to analyze.".to_string());
    }

    let mut analyzed_sessions = Vec::new();
    for session in sessions {
        let Ok(initial_history) =
            RolloutRecorder::get_rollout_history(session.path.as_path()).await
        else {
            continue;
        };
        let items = initial_history.get_rollout_items();
        analyzed_sessions.push(SessionAnalysis::from_rollout(session, items));
    }

    if analyzed_sessions.is_empty() {
        return Err("Unable to read any previous interactive sessions.".to_string());
    }

    let digest = ImproveDigest::from_sessions(analyzed_sessions);
    Ok(digest.render_prompt(limit))
}

async fn collect_recent_sessions(
    config: &Config,
    current_rollout_path: Option<&Path>,
    limit: usize,
) -> Result<Vec<ThreadItem>, String> {
    let fetch_limit = limit.saturating_mul(2).max(limit.saturating_add(1));
    let current_page = RolloutRecorder::list_threads(
        config,
        fetch_limit,
        None,
        ThreadSortKey::CreatedAt,
        INTERACTIVE_SESSION_SOURCES,
        None,
        config.model_provider_id.as_str(),
        None,
    )
    .await
    .map_err(|err| format!("failed to list recent sessions: {err}"))?;
    let archived_page = RolloutRecorder::list_archived_threads(
        config,
        fetch_limit,
        None,
        ThreadSortKey::CreatedAt,
        INTERACTIVE_SESSION_SOURCES,
        None,
        config.model_provider_id.as_str(),
        None,
    )
    .await
    .map_err(|err| format!("failed to list archived sessions: {err}"))?;

    let mut seen_paths = HashSet::new();
    let mut items = current_page.items;
    items.extend(archived_page.items);
    items.retain(|item| {
        if current_rollout_path.is_some_and(|path| path == item.path.as_path()) {
            return false;
        }
        seen_paths.insert(item.path.clone())
    });
    items.sort_by(|left, right| {
        right
            .created_at
            .cmp(&left.created_at)
            .then_with(|| right.path.cmp(&left.path))
    });
    items.truncate(limit);
    Ok(items)
}

#[derive(Clone, Debug, Default)]
struct SessionAnalysis {
    created_at: String,
    preview: String,
    cwd: Option<String>,
    user_messages: usize,
    assistant_messages: usize,
    final_answers: usize,
    commentary_messages: usize,
    tool_calls: usize,
    exec_commands: usize,
    failed_exec_commands: usize,
    declined_exec_commands: usize,
    stream_errors: usize,
    turn_aborts: usize,
    approvals: usize,
    compactions: usize,
    tool_counts: HashMap<String, usize>,
    command_counts: HashMap<String, usize>,
    failed_command_counts: HashMap<String, usize>,
    assistant_phrase_samples: Vec<String>,
}

impl SessionAnalysis {
    fn from_rollout(session: ThreadItem, items: Vec<RolloutItem>) -> Self {
        let mut analysis = Self {
            created_at: session
                .created_at
                .unwrap_or_else(|| "unknown time".to_string()),
            preview: session
                .first_user_message
                .map(|text| shorten(text.as_str(), PREVIEW_CHAR_LIMIT))
                .unwrap_or_else(|| "(no user preview recorded)".to_string()),
            cwd: session.cwd.map(|cwd| shorten_path(cwd.as_path())),
            ..Self::default()
        };

        let mut compaction_events = 0usize;
        for item in items {
            match item {
                RolloutItem::ResponseItem(response) => analysis.record_response_item(response),
                RolloutItem::Compacted(_) => {
                    analysis.compactions += 1;
                }
                RolloutItem::EventMsg(event) => {
                    if matches!(event, EventMsg::ContextCompacted(_)) {
                        compaction_events += 1;
                    }
                    analysis.record_event(event);
                }
                RolloutItem::SessionMeta(_) | RolloutItem::TurnContext(_) => {}
            }
        }

        if analysis.compactions == 0 {
            analysis.compactions = compaction_events;
        }

        analysis
    }

    fn record_response_item(&mut self, response: ResponseItem) {
        match response {
            ResponseItem::Message {
                role,
                content,
                phase,
                ..
            } => {
                let text = extract_message_text(content.as_slice());
                match role.as_str() {
                    "user" => {
                        self.user_messages += 1;
                        if self.preview == "(no user preview recorded)"
                            && let Some(text) = text.as_deref()
                        {
                            self.preview = shorten(text, PREVIEW_CHAR_LIMIT);
                        }
                    }
                    "assistant" => {
                        self.assistant_messages += 1;
                        if phase == Some(MessagePhase::Commentary) {
                            self.commentary_messages += 1;
                        } else {
                            self.final_answers += 1;
                        }
                        if let Some(text) = text.as_deref()
                            && let Some(sample) = assistant_phrase_sample(text)
                        {
                            self.assistant_phrase_samples.push(sample);
                        }
                    }
                    _ => {}
                }
            }
            ResponseItem::FunctionCall { name, .. } => {
                self.tool_calls += 1;
                increment_count(&mut self.tool_counts, name);
            }
            ResponseItem::CustomToolCall { name, .. } => {
                self.tool_calls += 1;
                increment_count(&mut self.tool_counts, format!("custom:{name}"));
            }
            ResponseItem::ToolSearchCall { .. } => {
                self.tool_calls += 1;
                increment_count(&mut self.tool_counts, "tool_search");
            }
            ResponseItem::WebSearchCall { .. } => {
                self.tool_calls += 1;
                increment_count(&mut self.tool_counts, "web_search");
            }
            ResponseItem::ImageGenerationCall { .. } => {
                self.tool_calls += 1;
                increment_count(&mut self.tool_counts, "image_generation");
            }
            ResponseItem::LocalShellCall { .. } => {
                self.tool_calls += 1;
                increment_count(&mut self.tool_counts, "local_shell");
            }
            ResponseItem::Reasoning { .. }
            | ResponseItem::FunctionCallOutput { .. }
            | ResponseItem::CustomToolCallOutput { .. }
            | ResponseItem::ToolSearchOutput { .. }
            | ResponseItem::GhostSnapshot { .. }
            | ResponseItem::Compaction { .. }
            | ResponseItem::Other => {}
        }
    }

    fn record_event(&mut self, event: EventMsg) {
        match event {
            EventMsg::ExecCommandEnd(exec) => self.record_exec_event(exec),
            EventMsg::StreamError(_) => {
                self.stream_errors += 1;
            }
            EventMsg::TurnAborted(_) => {
                self.turn_aborts += 1;
            }
            EventMsg::ExecApprovalRequest(_)
            | EventMsg::ApplyPatchApprovalRequest(_)
            | EventMsg::RequestPermissions(_) => {
                self.approvals += 1;
            }
            EventMsg::AgentMessage(message) => {
                if self.assistant_messages == 0 {
                    self.assistant_messages += 1;
                    if message.phase == Some(MessagePhase::Commentary) {
                        self.commentary_messages += 1;
                    } else {
                        self.final_answers += 1;
                    }
                    if let Some(sample) = assistant_phrase_sample(message.message.as_str()) {
                        self.assistant_phrase_samples.push(sample);
                    }
                }
            }
            EventMsg::UserMessage(message) => {
                if self.user_messages == 0 {
                    self.user_messages += 1;
                    if self.preview == "(no user preview recorded)" && !message.message.is_empty() {
                        self.preview = shorten(message.message.as_str(), PREVIEW_CHAR_LIMIT);
                    }
                }
            }
            _ => {}
        }
    }

    fn record_exec_event(&mut self, exec: ExecCommandEndEvent) {
        self.exec_commands += 1;
        let command = summarize_exec_command(exec.command.as_slice());
        increment_count(&mut self.command_counts, command.clone());
        if exec.status != ExecCommandStatus::Completed || exec.exit_code != 0 {
            self.failed_exec_commands += 1;
            if exec.status == ExecCommandStatus::Declined {
                self.declined_exec_commands += 1;
            }
            increment_count(&mut self.failed_command_counts, command);
        }
    }

    fn friction_score(&self) -> usize {
        self.failed_exec_commands * 4
            + self.stream_errors * 4
            + self.turn_aborts * 3
            + self.declined_exec_commands * 2
            + self.approvals
            + self.exec_commands.saturating_sub(8)
    }

    fn summary_line(&self) -> String {
        let mut line = format!(
            "- {} | {} | users {}, assistant {}, commentary {}, tools {}, exec {}, failed exec {}, approvals {}, compactions {}",
            self.created_at,
            self.preview,
            self.user_messages,
            self.assistant_messages,
            self.commentary_messages,
            self.tool_calls,
            self.exec_commands,
            self.failed_exec_commands,
            self.approvals,
            self.compactions,
        );
        if let Some(cwd) = self.cwd.as_deref() {
            line.push_str(&format!(" | cwd {cwd}"));
        }
        line
    }

    fn friction_line(&self) -> String {
        let mut reasons = Vec::new();
        if self.failed_exec_commands > 0 {
            reasons.push(format!("failed exec {}", self.failed_exec_commands));
        }
        if self.stream_errors > 0 {
            reasons.push(format!("stream errors {}", self.stream_errors));
        }
        if self.turn_aborts > 0 {
            reasons.push(format!("turn aborts {}", self.turn_aborts));
        }
        if self.approvals > 0 {
            reasons.push(format!("approvals {}", self.approvals));
        }
        if reasons.is_empty() && self.exec_commands > 0 {
            reasons.push(format!("exec commands {}", self.exec_commands));
        }
        let top_tools = format_count_entries(top_counts(&self.tool_counts, 3));
        let top_failed = format_count_entries(top_counts(&self.failed_command_counts, 2));
        let mut line = format!(
            "- score {} | {} | {}",
            self.friction_score(),
            self.created_at,
            self.preview
        );
        line.push_str(&format!(" | {}", reasons.join(", ")));
        if !top_tools.is_empty() {
            line.push_str(&format!(" | top tools: {top_tools}"));
        }
        if !top_failed.is_empty() {
            line.push_str(&format!(" | failed commands: {top_failed}"));
        }
        line
    }
}

struct ImproveDigest {
    sessions: Vec<SessionAnalysis>,
    total_user_messages: usize,
    total_assistant_messages: usize,
    total_commentary_messages: usize,
    total_final_answers: usize,
    total_tool_calls: usize,
    total_exec_commands: usize,
    total_failed_exec_commands: usize,
    total_stream_errors: usize,
    total_turn_aborts: usize,
    total_approvals: usize,
    total_compactions: usize,
    top_tools: Vec<(String, usize)>,
    top_commands: Vec<(String, usize)>,
    top_failed_commands: Vec<(String, usize)>,
    top_cwds: Vec<(String, usize)>,
    repeated_phrases: Vec<(String, usize)>,
}

impl ImproveDigest {
    fn from_sessions(sessions: Vec<SessionAnalysis>) -> Self {
        let total_user_messages = sessions.iter().map(|session| session.user_messages).sum();
        let total_assistant_messages = sessions
            .iter()
            .map(|session| session.assistant_messages)
            .sum();
        let total_commentary_messages = sessions
            .iter()
            .map(|session| session.commentary_messages)
            .sum();
        let total_final_answers = sessions.iter().map(|session| session.final_answers).sum();
        let total_tool_calls = sessions.iter().map(|session| session.tool_calls).sum();
        let total_exec_commands = sessions.iter().map(|session| session.exec_commands).sum();
        let total_failed_exec_commands = sessions
            .iter()
            .map(|session| session.failed_exec_commands)
            .sum();
        let total_stream_errors = sessions.iter().map(|session| session.stream_errors).sum();
        let total_turn_aborts = sessions.iter().map(|session| session.turn_aborts).sum();
        let total_approvals = sessions.iter().map(|session| session.approvals).sum();
        let total_compactions = sessions.iter().map(|session| session.compactions).sum();

        let mut tool_counts = HashMap::new();
        let mut command_counts = HashMap::new();
        let mut failed_command_counts = HashMap::new();
        let mut cwd_counts = HashMap::new();
        let mut phrase_counts = HashMap::new();

        for session in &sessions {
            merge_counts(&mut tool_counts, &session.tool_counts);
            merge_counts(&mut command_counts, &session.command_counts);
            merge_counts(&mut failed_command_counts, &session.failed_command_counts);
            if let Some(cwd) = session.cwd.as_deref() {
                increment_count(&mut cwd_counts, cwd.to_string());
            }
            for sample in &session.assistant_phrase_samples {
                if let Some(key) = repetition_key(sample.as_str()) {
                    let entry = phrase_counts
                        .entry(key)
                        .or_insert_with(|| (sample.clone(), 0usize));
                    entry.1 += 1;
                }
            }
        }

        let mut repeated_phrases = phrase_counts
            .into_values()
            .filter(|(_, count)| *count > 1)
            .collect::<Vec<_>>();
        repeated_phrases.sort_by_key(|(sample, count)| (Reverse(*count), sample.clone()));
        repeated_phrases.truncate(AGGREGATE_LIMIT);

        Self {
            sessions,
            total_user_messages,
            total_assistant_messages,
            total_commentary_messages,
            total_final_answers,
            total_tool_calls,
            total_exec_commands,
            total_failed_exec_commands,
            total_stream_errors,
            total_turn_aborts,
            total_approvals,
            total_compactions,
            top_tools: top_counts(&tool_counts, AGGREGATE_LIMIT),
            top_commands: top_counts(&command_counts, AGGREGATE_LIMIT),
            top_failed_commands: top_counts(&failed_command_counts, AGGREGATE_LIMIT),
            top_cwds: top_counts(&cwd_counts, 5),
            repeated_phrases,
        }
    }

    fn render_prompt(&self, requested_limit: usize) -> String {
        let mut lines = vec![
            format!(
                "Analyze this digest of my last {} interactive Codex sessions and tell me how Codex should improve.",
                self.sessions.len()
            ),
            String::new(),
            "Focus on:".to_string(),
            "- Concrete friction patterns and the smallest prompt or default-behavior change that would address each one.".to_string(),
            "- Explicit suggestions to try next, including prompt wording, default behaviors, and automation when the evidence is strong. Make the suggestions succinct and actionable.".to_string(),
            "- Repeated explanations or commentary that should be cut, shortened, or standardized."
                .to_string(),
            "- Evidence-driven recommendations only. If support is weak, say so.".to_string(),
            "- Only suggest a new skill, slash command, hook, or helper script when the same workflow or friction shows up in at least two sessions.".to_string(),
            "- Keep the response concise: use at most 3 bullets per section.".to_string(),
            "- In every section, make each bullet start with the change.".to_string(),
            "- Do not restate the digest; cite only the minimum evidence needed to justify each recommendation.".to_string(),
            String::new(),
            "Respond with these sections:".to_string(),
            "1. Friction to address".to_string(),
            "2. Suggestions to try".to_string(),
            "3. Repetition to remove".to_string(),
            "4. Ship next".to_string(),
            String::new(),
            "## Aggregate stats".to_string(),
            format!("- Sessions analyzed: {}", self.sessions.len()),
            format!(
                "- Messages: user {}, assistant {}, commentary {}, final answers {}",
                self.total_user_messages,
                self.total_assistant_messages,
                self.total_commentary_messages,
                self.total_final_answers,
            ),
            format!(
                "- Workload: tool calls {}, exec commands {}, failed exec commands {}",
                self.total_tool_calls, self.total_exec_commands, self.total_failed_exec_commands,
            ),
            format!(
                "- Friction signals: stream errors {}, turn aborts {}, approvals {}, compactions {}",
                self.total_stream_errors,
                self.total_turn_aborts,
                self.total_approvals,
                self.total_compactions,
            ),
        ];
        if self.sessions.len() < requested_limit {
            lines.push(format!(
                "- Sample size note: only {} earlier sessions were available.",
                self.sessions.len()
            ));
        }

        lines.push(String::new());
        lines.push("## Most used tools".to_string());
        lines.extend(render_count_section(&self.top_tools));

        lines.push(String::new());
        lines.push("## Most used shell commands".to_string());
        lines.extend(render_count_section(&self.top_commands));

        lines.push(String::new());
        lines.push("## Most failed shell commands".to_string());
        lines.extend(render_count_section(&self.top_failed_commands));

        lines.push(String::new());
        lines.push("## Common working directories".to_string());
        lines.extend(render_count_section(&self.top_cwds));

        lines.push(String::new());
        lines.push("## Repeated assistant phrases".to_string());
        if self.repeated_phrases.is_empty() {
            lines.push(
                "- No strongly repeated assistant phrasing was detected in the local digest."
                    .to_string(),
            );
        } else {
            for (sample, count) in &self.repeated_phrases {
                lines.push(format!("- {sample} (seen {count} times)"));
            }
        }

        lines.push(String::new());
        lines.push("## Highest-friction sessions".to_string());
        let mut friction_sessions = self.sessions.clone();
        friction_sessions.sort_by_key(|session| {
            (
                Reverse(session.friction_score()),
                Reverse(session.failed_exec_commands),
                session.created_at.clone(),
            )
        });
        let mut added_friction = 0usize;
        for session in friction_sessions
            .into_iter()
            .filter(|session| session.friction_score() > 0)
            .take(FRICTION_LIMIT)
        {
            lines.push(session.friction_line());
            added_friction += 1;
        }
        if added_friction == 0 {
            lines
                .push("- No strong friction signal stood out in the sampled sessions.".to_string());
        }

        lines.push(String::new());
        lines.push("## Session summaries".to_string());
        for session in &self.sessions {
            lines.push(session.summary_line());
        }

        lines.join("\n")
    }
}

fn extract_message_text(content: &[ContentItem]) -> Option<String> {
    let mut text = String::new();
    for item in content {
        match item {
            ContentItem::InputText { text: item_text }
            | ContentItem::OutputText { text: item_text } => {
                text.push_str(item_text);
            }
            ContentItem::InputImage { .. } => {}
        }
    }
    let text = collapse_whitespace(text.as_str());
    (!text.is_empty()).then_some(text)
}

fn assistant_phrase_sample(text: &str) -> Option<String> {
    let first_line = text
        .lines()
        .map(str::trim)
        .find(|line| !line.is_empty() && !line.starts_with("```"))?;
    let shortened = shorten(first_line, PHRASE_CHAR_LIMIT);
    repetition_key(shortened.as_str()).map(|_| shortened)
}

fn repetition_key(text: &str) -> Option<String> {
    let mut out = String::new();
    let mut previous_space = false;
    for ch in text.chars() {
        if ch.is_ascii_alphanumeric() {
            out.push(ch.to_ascii_lowercase());
            previous_space = false;
            continue;
        }
        if !previous_space && !out.is_empty() {
            out.push(' ');
            previous_space = true;
        }
    }
    let mut normalized_words = Vec::new();
    for word in out.split_whitespace() {
        let expanded = match word {
            "ll" if !normalized_words.is_empty() => "will",
            "ve" if !normalized_words.is_empty() => "have",
            "re" if !normalized_words.is_empty() => "are",
            "m" if !normalized_words.is_empty() => "am",
            _ => word,
        };
        normalized_words.push(expanded);
    }
    let normalized = normalized_words.join(" ");
    if normalized.len() < 24 || normalized.split_whitespace().count() < 4 {
        return None;
    }
    Some(normalized)
}

fn summarize_exec_command(command: &[String]) -> String {
    if let Some(shell_script) = unwrap_shell_script(command) {
        let segment = first_shell_segment(shell_script);
        let summary = take_command_words(segment, 4);
        if !summary.is_empty() {
            return summary;
        }
    }

    let summary = take_command_words(command.join(" ").as_str(), 4);
    if summary.is_empty() {
        "unknown_command".to_string()
    } else {
        summary
    }
}

fn unwrap_shell_script(command: &[String]) -> Option<&str> {
    if command.len() >= 3 && matches!(command[1].as_str(), "-lc" | "-c") {
        return Some(command[2].as_str());
    }
    None
}

fn first_shell_segment(script: &str) -> &str {
    let mut best = script.len();
    for needle in ["&&", "||", ";", "|", "\n"] {
        if let Some(index) = script.find(needle) {
            best = best.min(index);
        }
    }
    script[..best].trim()
}

fn take_command_words(text: &str, limit: usize) -> String {
    text.split_whitespace()
        .take(limit)
        .collect::<Vec<_>>()
        .join(" ")
}

fn collapse_whitespace(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn shorten(text: &str, max_chars: usize) -> String {
    let collapsed = collapse_whitespace(text);
    let mut shortened = collapsed.chars().take(max_chars + 1).collect::<String>();
    if shortened.chars().count() <= max_chars {
        return shortened;
    }
    shortened = shortened.chars().take(max_chars).collect();
    format!("{shortened}...")
}

fn shorten_path(path: &Path) -> String {
    shorten(path.to_string_lossy().as_ref(), 72)
}

fn increment_count(map: &mut HashMap<String, usize>, key: impl Into<String>) {
    *map.entry(key.into()).or_default() += 1;
}

fn merge_counts(target: &mut HashMap<String, usize>, counts: &HashMap<String, usize>) {
    for (name, count) in counts {
        *target.entry(name.clone()).or_default() += count;
    }
}

fn top_counts(counts: &HashMap<String, usize>, limit: usize) -> Vec<(String, usize)> {
    let mut entries = counts
        .iter()
        .map(|(name, count)| (name.clone(), *count))
        .collect::<Vec<_>>();
    entries.sort_by_key(|(name, count)| (Reverse(*count), name.clone()));
    entries.truncate(limit);
    entries
}

fn format_count_entries(entries: Vec<(String, usize)>) -> String {
    entries
        .into_iter()
        .map(|(name, count)| format!("{name} x{count}"))
        .collect::<Vec<_>>()
        .join(", ")
}

fn render_count_section(entries: &[(String, usize)]) -> Vec<String> {
    if entries.is_empty() {
        return vec!["- None recorded in the sampled sessions.".to_string()];
    }
    entries
        .iter()
        .map(|(name, count)| format!("- {name}: {count}"))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use codex_core::ARCHIVED_SESSIONS_SUBDIR;
    use codex_core::config::ConfigBuilder;
    use codex_protocol::ThreadId;
    use codex_protocol::protocol::AgentMessageEvent;
    use codex_protocol::protocol::GitInfo;
    use codex_protocol::protocol::RolloutLine;
    use codex_protocol::protocol::SessionMeta;
    use codex_protocol::protocol::SessionMetaLine;
    use codex_protocol::protocol::SessionSource;
    use pretty_assertions::assert_eq;
    use tempfile::TempDir;
    use uuid::Uuid;

    fn rollout_path(root: &Path, subdir: &str, filename_ts: &str, thread_id: &str) -> PathBuf {
        let filename = format!("rollout-{filename_ts}-{thread_id}.jsonl");
        if subdir == ARCHIVED_SESSIONS_SUBDIR {
            root.join(subdir).join(filename)
        } else {
            let year = &filename_ts[0..4];
            let month = &filename_ts[5..7];
            let day = &filename_ts[8..10];
            root.join(subdir)
                .join(year)
                .join(month)
                .join(day)
                .join(filename)
        }
    }

    struct RolloutFixture<'a> {
        subdir: &'a str,
        filename_ts: &'a str,
        timestamp: &'a str,
        preview: &'a str,
        assistant_commentary: &'a str,
        exec_command: &'a [&'a str],
        exec_status: ExecCommandStatus,
        exit_code: i32,
    }

    fn build_rollout_lines(
        thread_id: ThreadId,
        cwd: &Path,
        fixture: &RolloutFixture<'_>,
    ) -> Vec<String> {
        let session_meta = SessionMeta {
            id: thread_id,
            forked_from_id: None,
            timestamp: fixture.timestamp.to_string(),
            cwd: cwd.to_path_buf(),
            originator: "codex".to_string(),
            cli_version: "0.0.0".to_string(),
            source: SessionSource::Cli,
            agent_nickname: None,
            agent_role: None,
            model_provider: Some("openai".to_string()),
            base_instructions: None,
            dynamic_tools: None,
            memory_mode: None,
        };
        let items = vec![
            RolloutItem::SessionMeta(SessionMetaLine {
                meta: session_meta,
                git: Some(GitInfo {
                    commit_hash: None,
                    branch: Some("main".to_string()),
                    repository_url: None,
                }),
            }),
            RolloutItem::ResponseItem(ResponseItem::Message {
                id: None,
                role: "user".to_string(),
                content: vec![ContentItem::InputText {
                    text: fixture.preview.to_string(),
                }],
                end_turn: None,
                phase: None,
            }),
            RolloutItem::EventMsg(EventMsg::UserMessage(
                codex_protocol::protocol::UserMessageEvent {
                    message: fixture.preview.to_string(),
                    images: None,
                    local_images: Vec::new(),
                    text_elements: Vec::new(),
                },
            )),
            RolloutItem::ResponseItem(ResponseItem::Message {
                id: None,
                role: "assistant".to_string(),
                content: vec![ContentItem::OutputText {
                    text: fixture.assistant_commentary.to_string(),
                }],
                end_turn: None,
                phase: Some(MessagePhase::Commentary),
            }),
            RolloutItem::EventMsg(EventMsg::AgentMessage(AgentMessageEvent {
                message: fixture.assistant_commentary.to_string(),
                phase: Some(MessagePhase::Commentary),
            })),
            RolloutItem::ResponseItem(ResponseItem::FunctionCall {
                id: None,
                name: "exec_command".to_string(),
                namespace: None,
                arguments: "{}".to_string(),
                call_id: "call-1".to_string(),
            }),
            RolloutItem::EventMsg(EventMsg::ExecCommandEnd(ExecCommandEndEvent {
                call_id: "call-1".to_string(),
                process_id: None,
                turn_id: "turn-1".to_string(),
                command: fixture
                    .exec_command
                    .iter()
                    .map(ToString::to_string)
                    .collect(),
                cwd: cwd.to_path_buf(),
                parsed_cmd: Vec::new(),
                source: codex_protocol::protocol::ExecCommandSource::Agent,
                interaction_input: None,
                stdout: String::new(),
                stderr: String::new(),
                aggregated_output: String::new(),
                exit_code: fixture.exit_code,
                duration: std::time::Duration::from_millis(100),
                formatted_output: String::new(),
                status: fixture.exec_status.clone(),
            })),
        ];

        items
            .into_iter()
            .map(|item| {
                serde_json::to_string(&RolloutLine {
                    timestamp: fixture.timestamp.to_string(),
                    item,
                })
                .expect("rollout line")
            })
            .collect()
    }

    fn write_rollout(root: &Path, fixture: RolloutFixture<'_>) -> PathBuf {
        let uuid = Uuid::new_v4();
        let thread_id = ThreadId::from_string(uuid.to_string().as_str()).expect("thread id");
        let rollout_path = rollout_path(
            root,
            fixture.subdir,
            fixture.filename_ts,
            uuid.to_string().as_str(),
        );
        std::fs::create_dir_all(rollout_path.parent().expect("parent")).expect("create dir");
        let lines = build_rollout_lines(thread_id, root, &fixture);
        std::fs::write(&rollout_path, lines.join("\n") + "\n").expect("write rollout");
        rollout_path
    }

    async fn test_config(temp_home: &TempDir) -> Config {
        ConfigBuilder::default()
            .codex_home(temp_home.path().to_path_buf())
            .build()
            .await
            .expect("config")
    }

    #[tokio::test]
    async fn build_improve_prompt_includes_archived_sessions_and_excludes_current_rollout() {
        let temp_home = tempfile::tempdir().expect("tempdir");
        let config = test_config(&temp_home).await;
        let current_rollout = write_rollout(
            temp_home.path(),
            RolloutFixture {
                subdir: "sessions",
                filename_ts: "2026-03-12T12-00-00",
                timestamp: "2026-03-12T12:00:00Z",
                preview: "Current session should be excluded",
                assistant_commentary: "I will inspect the command plumbing first.",
                exec_command: &["bash", "-lc", "rg improve_command"],
                exec_status: ExecCommandStatus::Completed,
                exit_code: 0,
            },
        );
        let archived_rollout = write_rollout(
            temp_home.path(),
            RolloutFixture {
                subdir: "archived_sessions",
                filename_ts: "2026-03-10T08-00-00",
                timestamp: "2026-03-10T08:00:00Z",
                preview: "Investigate repeated CI flakes",
                assistant_commentary: "I will inspect the command plumbing first.",
                exec_command: &["bash", "-lc", "cargo test -p codex-tui"],
                exec_status: ExecCommandStatus::Failed,
                exit_code: 101,
            },
        );
        let _recent_rollout = write_rollout(
            temp_home.path(),
            RolloutFixture {
                subdir: "sessions",
                filename_ts: "2026-03-11T09-00-00",
                timestamp: "2026-03-11T09:00:00Z",
                preview: "Implement a new slash command",
                assistant_commentary: "I will inspect the command plumbing first.",
                exec_command: &["bash", "-lc", "cargo test -p codex-core"],
                exec_status: ExecCommandStatus::Failed,
                exit_code: 1,
            },
        );

        let prompt = build_improve_prompt(
            config,
            Some(current_rollout.clone()),
            DEFAULT_IMPROVE_SESSION_LIMIT,
        )
        .await
        .expect("prompt");

        assert!(prompt.contains("Investigate repeated CI flakes"));
        assert!(prompt.contains("Implement a new slash command"));
        assert!(!prompt.contains("Current session should be excluded"));
        assert!(prompt.contains("Repeated assistant phrases"));
        assert!(prompt.contains("I will inspect the command plumbing first."));
        assert!(prompt.contains("Most failed shell commands"));
        assert!(prompt.contains("cargo test -p codex-tui"));
        assert!(prompt.contains("Friction to address"));
        assert!(prompt.contains("Suggestions to try"));
        assert!(prompt.contains("Repetition to remove"));
        assert!(prompt.contains("Ship next"));
        assert!(prompt.contains("Make the suggestions succinct and actionable."));
        assert!(!prompt.contains("Project areas"));
        assert!(!prompt.contains("Interaction style"));
        assert!(!prompt.contains(archived_rollout.to_string_lossy().as_ref()));
    }

    #[test]
    fn repetition_key_collapses_minor_formatting_changes() {
        let left = repetition_key("I will inspect the command plumbing first.");
        let right = repetition_key("I'll inspect the command plumbing first!");
        assert_eq!(left, right);
    }
}
