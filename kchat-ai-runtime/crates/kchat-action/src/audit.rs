//! Audit log — structured records of actions and policy outcomes.
//!
//! Audit records structured actions and policy outcomes, with raw content
//! retained only under explicit tenant policy.

use kchat_core::ids::{TaskId, TenantId, ToolId, UserId};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// Outcome of an audited action.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AuditOutcome {
    /// Action was authorized and committed
    Committed,
    /// Action was dry-run only (not committed)
    DryRun,
    /// Action was denied by policy
    Denied,
    /// Action was cancelled by user
    Cancelled,
    /// Action failed during execution
    Failed,
}

/// A single audit entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEntry {
    /// Timestamp (epoch millis)
    pub timestamp_ms: u128,
    /// Task ID
    pub task_id: TaskId,
    /// Tenant ID
    pub tenant_id: TenantId,
    /// User ID (actor)
    pub user_id: UserId,
    /// Tool ID
    pub tool_id: Option<ToolId>,
    /// Action performed
    pub action: String,
    /// Outcome
    pub outcome: AuditOutcome,
    /// Confirmation class
    pub confirmation_class: Option<String>,
    /// Reason code (for denials or failures)
    pub reason_code: Option<String>,
    /// Whether step-up auth was required
    pub step_up_auth: bool,
    /// Commit token hash (if committed)
    pub commit_token_hash: Option<String>,
    /// Data scope
    pub data_scope: Option<String>,
}

impl AuditEntry {
    pub fn committed(
        task_id: TaskId,
        tenant_id: TenantId,
        user_id: UserId,
        action: impl Into<String>,
    ) -> Self {
        Self {
            timestamp_ms: current_millis(),
            task_id,
            tenant_id,
            user_id,
            tool_id: None,
            action: action.into(),
            outcome: AuditOutcome::Committed,
            confirmation_class: None,
            reason_code: None,
            step_up_auth: false,
            commit_token_hash: None,
            data_scope: None,
        }
    }

    pub fn denied(
        task_id: TaskId,
        tenant_id: TenantId,
        user_id: UserId,
        action: impl Into<String>,
        reason: impl Into<String>,
    ) -> Self {
        Self {
            timestamp_ms: current_millis(),
            task_id,
            tenant_id,
            user_id,
            tool_id: None,
            action: action.into(),
            outcome: AuditOutcome::Denied,
            confirmation_class: None,
            reason_code: Some(reason.into()),
            step_up_auth: false,
            commit_token_hash: None,
            data_scope: None,
        }
    }
}

/// In-memory audit log. In production, this is persisted to SQLCipher.
pub struct AuditLog {
    entries: Mutex<VecDeque<AuditEntry>>,
    max_entries: usize,
}

impl AuditLog {
    pub fn new(max_entries: usize) -> Self {
        Self {
            entries: Mutex::new(VecDeque::with_capacity(max_entries)),
            max_entries,
        }
    }

    /// Record an audit entry.
    pub fn record(&self, entry: AuditEntry) {
        let mut entries = self.entries.lock();
        if entries.len() >= self.max_entries {
            entries.pop_front();
        }
        entries.push_back(entry);
    }

    /// Drain all entries (for persistence flush).
    pub fn drain(&self) -> Vec<AuditEntry> {
        let mut entries = self.entries.lock();
        entries.drain(..).collect()
    }

    /// Number of entries.
    pub fn len(&self) -> usize {
        self.entries.lock().len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.lock().is_empty()
    }

    /// Get entries for a specific tenant.
    pub fn for_tenant(&self, tenant_id: &TenantId) -> Vec<AuditEntry> {
        self.entries
            .lock()
            .iter()
            .filter(|e| &e.tenant_id == tenant_id)
            .cloned()
            .collect()
    }
}

impl Default for AuditLog {
    fn default() -> Self {
        Self::new(10_000)
    }
}

fn current_millis() -> u128 {
    use std::time::SystemTime;
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_record_and_drain() {
        let log = AuditLog::new(100);
        let task = TaskId::new();
        let tenant = TenantId::new();
        let user = UserId::new();

        log.record(AuditEntry::committed(task, tenant, user, "search"));
        log.record(AuditEntry::denied(task, tenant, user, "delete", "permission_denied"));

        assert_eq!(log.len(), 2);

        let entries = log.drain();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].outcome, AuditOutcome::Committed);
        assert_eq!(entries[1].outcome, AuditOutcome::Denied);
        assert!(log.is_empty());
    }

    #[test]
    fn test_tenant_filter() {
        let log = AuditLog::new(100);
        let task = TaskId::new();
        let tenant1 = TenantId::new();
        let tenant2 = TenantId::new();
        let user = UserId::new();

        log.record(AuditEntry::committed(task, tenant1, user, "action1"));
        log.record(AuditEntry::committed(task, tenant2, user, "action2"));

        let t1_entries = log.for_tenant(&tenant1);
        assert_eq!(t1_entries.len(), 1);
        assert_eq!(t1_entries[0].action, "action1");
    }

    #[test]
    fn test_max_entries_eviction() {
        let log = AuditLog::new(2);
        let task = TaskId::new();
        let tenant = TenantId::new();
        let user = UserId::new();

        log.record(AuditEntry::committed(task, tenant, user, "a1"));
        log.record(AuditEntry::committed(task, tenant, user, "a2"));
        log.record(AuditEntry::committed(task, tenant, user, "a3"));

        let entries = log.drain();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].action, "a2"); // oldest evicted
        assert_eq!(entries[1].action, "a3");
    }
}
