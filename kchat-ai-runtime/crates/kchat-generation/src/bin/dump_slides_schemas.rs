//! Dump all slide template schemas from SlidesTemplateRegistry to JSON.
//! Run: cargo run -p kchat-generation --bin dump_slides_schemas -- <output.json>

use kchat_generation::slides_templates::TEMPLATE_REGISTRY;

fn main() {
    let out_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "slides_schemas.json".to_string());

    let registry = &*TEMPLATE_REGISTRY;
    let templates: Vec<serde_json::Value> = registry
        .all()
        .iter()
        .map(|t| {
            serde_json::json!({
                "id": t.id,
                "label": t.label,
                "family": format!("{:?}", t.family),
                "layout_hint": t.layout_hint,
                "max_bullets": t.max_bullets,
                "supports_chart": t.supports_chart,
                "slots": t.slots.iter().map(|s| serde_json::json!({
                    "id": s.id,
                    "slot_type": format!("{:?}", s.slot_type),
                    "label": s.label,
                    "required": s.required,
                })).collect::<Vec<_>>(),
                "slot_schema": t.slot_schema(),
            })
        })
        .collect();

    let json = serde_json::json!({
        "count": templates.len(),
        "templates": templates,
    });

    std::fs::write(&out_path, serde_json::to_string_pretty(&json).unwrap())
        .expect("write slides_schemas.json");
    println!("Wrote {} templates to {}", templates.len(), out_path);
}
