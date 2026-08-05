//! Authorization broker — RBAC and confirmation policy.
//!
//! Tenant ID, actor ID, user roles, and data scopes must come from the
//! authenticated application or server context. They must never come from
//! a model-produced invocation. Every operation is reauthorized after
//! planning and immediately before execution.

use kchat_core::ids::{TenantId, ToolId, UserId};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Confirmation class for an operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConfirmationClass {
    /// Read-only, low sensitivity — may auto-run with limits
    ReadOnly,
    /// Local reversible mutation — dry-run, preview, confirmation, commit
    LocalMutation,
    /// External mutation — reauthorize server-side, show target and effect
    ExternalMutation,
    /// Finance, HR, admin, export, bulk — step-up auth and policy approval
    SensitiveAction,
}

/// Permission granted to a user for a specific tool or operation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Permission {
    pub tool_id: ToolId,
    pub actions: HashSet<String>,
    pub data_scopes: HashSet<String>,
    pub confirmation_class: ConfirmationClass,
}

/// Authenticated context — derived from the session, never from the model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthContext {
    pub user_id: UserId,
    pub tenant_id: TenantId,
    pub roles: Vec<String>,
    pub permissions: Vec<Permission>,
    pub session_token: String,
}

impl AuthContext {
    /// Check if the user has a specific role.
    pub fn has_role(&self, role: &str) -> bool {
        self.roles.iter().any(|r| r == role)
    }

    /// Check if the user has permission for a tool action.
    pub fn has_permission(&self, tool_id: &ToolId, action: &str) -> bool {
        self.permissions.iter().any(|p| {
            &p.tool_id == tool_id && p.actions.contains(action)
        })
    }

    /// Get the confirmation class for a tool.
    pub fn confirmation_class_for(&self, tool_id: &ToolId) -> Option<ConfirmationClass> {
        self.permissions.iter().find(|p| &p.tool_id == tool_id).map(|p| p.confirmation_class)
    }

    /// Check if the user has access to a data scope.
    pub fn has_data_scope(&self, tool_id: &ToolId, scope: &str) -> bool {
        self.permissions.iter().any(|p| {
            &p.tool_id == tool_id && p.data_scopes.contains(scope)
        })
    }
}

/// RBAC broker — reauthorizes operations after planning and before execution.
pub struct RbacBroker {
    /// Additional role-to-permissions mapping (server-side policy)
    role_permissions: HashMap<String, Vec<Permission>>,
}

impl RbacBroker {
    pub fn new() -> Self {
        Self {
            role_permissions: HashMap::new(),
        }
    }

    /// Add server-side role permissions.
    pub fn add_role_permissions(&mut self, role: &str, permissions: Vec<Permission>) {
        self.role_permissions.insert(role.into(), permissions);
    }

    /// Reauthorize a tool plan step before execution.
    ///
    /// This is called AFTER the model produces a plan and IMMEDIATELY BEFORE
    /// execution. The server derives tenant, actor, roles, and source scopes
    /// from the authenticated session. It ignores those fields if the client
    /// or model supplies them.
    pub fn reauthorize(
        &self,
        auth: &AuthContext,
        tool_id: &ToolId,
        action: &str,
        data_scope: &str,
    ) -> Result<ConfirmationClass, AuthError> {
        // 1. Check permission exists in auth context
        if !auth.has_permission(tool_id, action) {
            // 2. Check server-side role permissions
            let has_role_perm = auth.roles.iter().any(|role| {
                self.role_permissions
                    .get(role)
                    .map(|perms| {
                        perms.iter().any(|p| {
                            &p.tool_id == tool_id && p.actions.contains(action)
                        })
                    })
                    .unwrap_or(false)
            });

            if !has_role_perm {
                return Err(AuthError::PermissionDenied {
                    tool_id: tool_id.to_string(),
                    action: action.into(),
                });
            }
        }

        // 3. Check data scope
        if !auth.has_data_scope(tool_id, data_scope) {
            // Check role-based scope
            let has_role_scope = auth.roles.iter().any(|role| {
                self.role_permissions
                    .get(role)
                    .map(|perms| {
                        perms.iter().any(|p| {
                            &p.tool_id == tool_id && p.data_scopes.contains(data_scope)
                        })
                    })
                    .unwrap_or(false)
            });

            if !has_role_scope {
                return Err(AuthError::ScopeDenied {
                    tool_id: tool_id.to_string(),
                    scope: data_scope.into(),
                });
            }
        }

        // 4. Get confirmation class
        let confirmation_class = auth
            .confirmation_class_for(tool_id)
            .or_else(|| {
                auth.roles.iter().find_map(|role| {
                    self.role_permissions.get(role).and_then(|perms| {
                        perms.iter().find(|p| &p.tool_id == tool_id).map(|p| p.confirmation_class)
                    })
                })
            })
            .ok_or(AuthError::NoConfirmationClass {
                tool_id: tool_id.to_string(),
            })?;

        Ok(confirmation_class)
    }

    /// Check if step-up authentication is required for a confirmation class.
    pub fn requires_step_up_auth(&self, class: ConfirmationClass) -> bool {
        matches!(class, ConfirmationClass::SensitiveAction)
    }

    /// Check if dry-run is required before commit.
    pub fn requires_dry_run(&self, class: ConfirmationClass) -> bool {
        matches!(
            class,
            ConfirmationClass::LocalMutation
                | ConfirmationClass::ExternalMutation
                | ConfirmationClass::SensitiveAction
        )
    }

    /// Full reauthorization with step-up auth and dry-run enforcement.
    /// Returns the confirmation class if authorized, or an error explaining
    /// what additional steps are needed.
    ///
    /// - `step_up_performed`: whether the user has completed step-up auth
    /// - `dry_run_completed`: whether a dry-run has been performed
    pub fn reauthorize_full(
        &self,
        auth: &AuthContext,
        tool_id: &ToolId,
        action: &str,
        data_scope: &str,
        step_up_performed: bool,
        dry_run_completed: bool,
    ) -> Result<ConfirmationClass, AuthError> {
        let class = self.reauthorize(auth, tool_id, action, data_scope)?;

        // Enforce step-up auth for sensitive actions
        if self.requires_step_up_auth(class) && !step_up_performed {
            return Err(AuthError::StepUpAuthRequired {
                tool_id: tool_id.to_string(),
            });
        }

        // Enforce dry-run before commit for mutation operations
        if self.requires_dry_run(class) && !dry_run_completed {
            return Err(AuthError::DryRunRequired {
                tool_id: tool_id.to_string(),
            });
        }

        Ok(class)
    }
}

impl Default for RbacBroker {
    fn default() -> Self {
        Self::new()
    }
}

/// Authorization errors.
#[derive(Debug, thiserror::Error)]
pub enum AuthError {
    #[error("permission denied for tool {tool_id} action {action}")]
    PermissionDenied { tool_id: String, action: String },

    #[error("data scope denied for tool {tool_id} scope {scope}")]
    ScopeDenied { tool_id: String, scope: String },

    #[error("no confirmation class found for tool {tool_id}")]
    NoConfirmationClass { tool_id: String },

    #[error("step-up authentication required for tool {tool_id}")]
    StepUpAuthRequired { tool_id: String },

    #[error("dry-run required before commit for tool {tool_id}")]
    DryRunRequired { tool_id: String },
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    fn make_auth_context() -> AuthContext {
        let tool_id = ToolId::new();
        let mut actions = HashSet::new();
        actions.insert("read".into());
        actions.insert("write".into());

        let mut scopes = HashSet::new();
        scopes.insert("workspace_123".into());

        AuthContext {
            user_id: UserId::new(),
            tenant_id: TenantId::new(),
            roles: vec!["member".into()],
            permissions: vec![Permission {
                tool_id,
                actions,
                data_scopes: scopes,
                confirmation_class: ConfirmationClass::LocalMutation,
            }],
            session_token: "test-token".into(),
        }
    }

    #[test]
    fn test_permission_granted() {
        let auth = make_auth_context();
        let tool_id = &auth.permissions[0].tool_id;
        let broker = RbacBroker::new();

        let result = broker.reauthorize(&auth, tool_id, "write", "workspace_123");
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), ConfirmationClass::LocalMutation);
    }

    #[test]
    fn test_permission_denied() {
        let auth = make_auth_context();
        let tool_id = &auth.permissions[0].tool_id;
        let broker = RbacBroker::new();

        let result = broker.reauthorize(&auth, tool_id, "delete", "workspace_123");
        assert!(result.is_err());
    }

    #[test]
    fn test_scope_denied() {
        let auth = make_auth_context();
        let tool_id = &auth.permissions[0].tool_id;
        let broker = RbacBroker::new();

        let result = broker.reauthorize(&auth, tool_id, "write", "workspace_999");
        assert!(result.is_err());
    }

    #[test]
    fn test_step_up_auth_for_sensitive() {
        let broker = RbacBroker::new();
        assert!(broker.requires_step_up_auth(ConfirmationClass::SensitiveAction));
        assert!(!broker.requires_step_up_auth(ConfirmationClass::ReadOnly));
    }

    #[test]
    fn test_dry_run_required_for_mutations() {
        let broker = RbacBroker::new();
        assert!(broker.requires_dry_run(ConfirmationClass::LocalMutation));
        assert!(broker.requires_dry_run(ConfirmationClass::ExternalMutation));
        assert!(broker.requires_dry_run(ConfirmationClass::SensitiveAction));
        assert!(!broker.requires_dry_run(ConfirmationClass::ReadOnly));
    }

    #[test]
    fn test_role_based_permissions() {
        let mut broker = RbacBroker::new();
        let tool_id = ToolId::new();

        let mut actions = HashSet::new();
        actions.insert("admin".into());
        let mut scopes = HashSet::new();
        scopes.insert("all".into());

        broker.add_role_permissions("admin", vec![Permission {
            tool_id,
            actions,
            data_scopes: scopes,
            confirmation_class: ConfirmationClass::SensitiveAction,
        }]);

        let auth = AuthContext {
            user_id: UserId::new(),
            tenant_id: TenantId::new(),
            roles: vec!["admin".into()],
            permissions: vec![],
            session_token: "test".into(),
        };

        let result = broker.reauthorize(&auth, &tool_id, "admin", "all");
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), ConfirmationClass::SensitiveAction);
    }

    #[test]
    fn test_reauthorize_full_step_up_required() {
        let broker = RbacBroker::new();
        let tool_id = ToolId::new();
        let mut actions = HashSet::new();
        actions.insert("execute".into());
        let mut scopes = HashSet::new();
        scopes.insert("workspace_1".into());

        let auth = AuthContext {
            user_id: UserId::new(),
            tenant_id: TenantId::new(),
            roles: vec![],
            permissions: vec![Permission {
                tool_id,
                actions,
                data_scopes: scopes,
                confirmation_class: ConfirmationClass::SensitiveAction,
            }],
            session_token: "test".into(),
        };

        // Without step-up auth → should fail
        let result = broker.reauthorize_full(&auth, &tool_id, "execute", "workspace_1", false, true);
        assert!(matches!(result, Err(AuthError::StepUpAuthRequired { .. })));

        // With step-up auth → should succeed
        let result = broker.reauthorize_full(&auth, &tool_id, "execute", "workspace_1", true, true);
        assert!(result.is_ok());
    }

    #[test]
    fn test_reauthorize_full_dry_run_required() {
        let broker = RbacBroker::new();
        let tool_id = ToolId::new();
        let mut actions = HashSet::new();
        actions.insert("write".into());
        let mut scopes = HashSet::new();
        scopes.insert("workspace_1".into());

        let auth = AuthContext {
            user_id: UserId::new(),
            tenant_id: TenantId::new(),
            roles: vec![],
            permissions: vec![Permission {
                tool_id,
                actions,
                data_scopes: scopes,
                confirmation_class: ConfirmationClass::LocalMutation,
            }],
            session_token: "test".into(),
        };

        // Without dry-run → should fail
        let result = broker.reauthorize_full(&auth, &tool_id, "write", "workspace_1", true, false);
        assert!(matches!(result, Err(AuthError::DryRunRequired { .. })));

        // With dry-run → should succeed
        let result = broker.reauthorize_full(&auth, &tool_id, "write", "workspace_1", true, true);
        assert!(result.is_ok());
    }
}
