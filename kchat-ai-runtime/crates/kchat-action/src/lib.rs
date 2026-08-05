//! kchat-action: Action plane — artifact AST, tool plan validation,
//! authorization broker, dry-run/commit/audit.
//!
//! The model emits a `ToolPlan`, not an executable request. Each app has a
//! signed manifest containing tool schemas, requested capabilities, data
//! scopes, network destinations, side effects, confirmation class, and
//! publisher identity.
//!
//! Tenant ID, actor ID, user roles, and data scopes must come from the
//! authenticated application or server context. They must never come from a
//! model-produced invocation. Every operation is reauthorized after planning
//! and immediately before execution.

pub mod artifact;
pub mod auth;
pub mod toolplan;
pub mod audit;

pub use artifact::{ArtifactAst, ArtifactNode, ArtifactNodeId, ArtifactType, ArtifactOperation, OperationValidator};
pub use auth::{AuthContext, ConfirmationClass, Permission, RbacBroker};
pub use toolplan::{ToolManifest, ToolPlan, ToolPlanStep, ToolPlanValidator};
pub use audit::{AuditEntry, AuditLog, AuditOutcome};
