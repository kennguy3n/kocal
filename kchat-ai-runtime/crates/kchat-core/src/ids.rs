//! Strongly-typed identifiers used across all KChat workload planes.
//!
//! All IDs are UUID-based, serialized as strings for FFI. New IDs use UUIDv4.
//! Deterministic IDs (e.g. for policy packs) may use UUIDv5 with a known namespace.

use serde::{Deserialize, Serialize};
use uuid::Uuid;

macro_rules! id_type {
    ($name:ident, $doc:expr) => {
        #[doc = $doc]
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
        pub struct $name(pub Uuid);

        impl $name {
            pub fn new() -> Self {
                Self(Uuid::new_v4())
            }

            pub fn from_uuid(u: Uuid) -> Self {
                Self(u)
            }

            pub fn as_uuid(&self) -> Uuid {
                self.0
            }

            pub fn as_str(&self) -> String {
                self.0.to_string()
            }
        }

        impl Default for $name {
            fn default() -> Self {
                Self::new()
            }
        }

        impl std::fmt::Display for $name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "{}", self.0)
            }
        }

        impl std::str::FromStr for $name {
            type Err = uuid::Error;
            fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
                Ok(Self(Uuid::parse_str(s)?))
            }
        }
    };
}

id_type!(UserId, "Authenticated KChat user identifier.");
id_type!(TenantId, "Tenant (workspace/organization) identifier.");
id_type!(TaskId, "Unique AI task identifier for tracking and audit.");
id_type!(ArtifactId, "Stable artifact node identifier (document/slide/sheet/base).");
id_type!(ModelPackId, "Signed model pack identifier.");
id_type!(PolicyPackId, "Signed policy/skill pack identifier.");
id_type!(ToolId, "Microapp tool identifier from a signed extension manifest.");
