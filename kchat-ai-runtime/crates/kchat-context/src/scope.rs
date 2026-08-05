//! Scope types — explicit scopes for indexing and retrieval.
//!
//! Local chat and artifacts are indexed under explicit scopes: user, account,
//! workspace, conversation, participant, source, record ACL, retention class,
//! and time. Retrieval checks authorization before search, filters candidates
//! during search, and checks again when constructing the prompt.

use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Stable identifier for a scope.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ScopeId(pub Uuid);

impl ScopeId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }

    pub fn from_uuid(u: Uuid) -> Self {
        Self(u)
    }
}

impl Default for ScopeId {
    fn default() -> Self {
        Self::new()
    }
}

/// A scope under which evidence is indexed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Scope {
    pub id: ScopeId,
    pub scope_type: ScopeType,
    /// Parent scope (for hierarchical authorization)
    pub parent: Option<ScopeId>,
    /// ACL version — incremented when permissions change
    pub acl_version: u32,
    /// Retention class (e.g. "permanent", "30_days", "session")
    pub retention_class: String,
    /// Authorized user IDs
    pub authorized_users: Vec<Uuid>,
    /// Authorized role names
    pub authorized_roles: Vec<String>,
}

/// Type of scope.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScopeType {
    User,
    Account,
    Workspace,
    Conversation,
    Participant,
    Source,
    RecordAcl,
}

/// Filter for retrieval — specifies which scopes are authorized for search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScopeFilter {
    /// Only return evidence from these scopes
    pub allowed_scopes: Vec<ScopeId>,
    /// Exclude evidence from these scopes
    pub denied_scopes: Vec<ScopeId>,
    /// Only return evidence visible to this user
    pub user_id: Uuid,
    /// Only return evidence visible to these roles
    pub roles: Vec<String>,
}

impl ScopeFilter {
    /// Check if a scope is authorized for this filter.
    ///
    /// This is the FIRST authorization check — before search.
    pub fn is_scope_authorized(&self, scope: &Scope) -> bool {
        // Check denied list
        if self.denied_scopes.contains(&scope.id) {
            return false;
        }

        // Check allowed list (empty = allow all non-denied)
        if !self.allowed_scopes.is_empty() && !self.allowed_scopes.contains(&scope.id) {
            return false;
        }

        // Check user authorization
        if !scope.authorized_users.is_empty() && !scope.authorized_users.contains(&self.user_id) {
            // Check role authorization
            if scope.authorized_roles.is_empty()
                || !scope.authorized_roles.iter().any(|r| self.roles.contains(r))
            {
                return false;
            }
        }

        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scope_authorized_for_user() {
        let user = Uuid::new_v4();
        let scope = Scope {
            id: ScopeId::new(),
            scope_type: ScopeType::Workspace,
            parent: None,
            acl_version: 1,
            retention_class: "permanent".into(),
            authorized_users: vec![user],
            authorized_roles: vec![],
        };

        let filter = ScopeFilter {
            allowed_scopes: vec![scope.id],
            denied_scopes: vec![],
            user_id: user,
            roles: vec![],
        };

        assert!(filter.is_scope_authorized(&scope));
    }

    #[test]
    fn test_scope_denied_for_unauthorized_user() {
        let user1 = Uuid::new_v4();
        let user2 = Uuid::new_v4();
        let scope = Scope {
            id: ScopeId::new(),
            scope_type: ScopeType::Workspace,
            parent: None,
            acl_version: 1,
            retention_class: "permanent".into(),
            authorized_users: vec![user1],
            authorized_roles: vec![],
        };

        let filter = ScopeFilter {
            allowed_scopes: vec![],
            denied_scopes: vec![],
            user_id: user2,
            roles: vec![],
        };

        assert!(!filter.is_scope_authorized(&scope));
    }

    #[test]
    fn test_scope_authorized_via_role() {
        let user = Uuid::new_v4();
        let scope = Scope {
            id: ScopeId::new(),
            scope_type: ScopeType::Workspace,
            parent: None,
            acl_version: 1,
            retention_class: "permanent".into(),
            authorized_users: vec![],
            authorized_roles: vec!["admin".into()],
        };

        let filter = ScopeFilter {
            allowed_scopes: vec![],
            denied_scopes: vec![],
            user_id: user,
            roles: vec!["admin".into()],
        };

        assert!(filter.is_scope_authorized(&scope));
    }

    #[test]
    fn test_scope_in_denied_list() {
        let user = Uuid::new_v4();
        let scope = Scope {
            id: ScopeId::new(),
            scope_type: ScopeType::Conversation,
            parent: None,
            acl_version: 1,
            retention_class: "session".into(),
            authorized_users: vec![user],
            authorized_roles: vec![],
        };

        let filter = ScopeFilter {
            allowed_scopes: vec![],
            denied_scopes: vec![scope.id],
            user_id: user,
            roles: vec![],
        };

        assert!(!filter.is_scope_authorized(&scope));
    }
}
