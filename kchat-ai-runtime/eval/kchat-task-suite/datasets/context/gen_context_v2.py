#!/usr/bin/env python3
"""Generate context_dataset_v2.json with 80+ docs and 40+ queries.

Features:
- 80+ documents across 14 languages, multiple scopes
- 40+ queries with graded relevance, multi-hop, cross-language
- ACL tests (denied scopes)
- Multi-hop queries (expect 2+ docs)
- Cross-language queries (query in one language, doc in another)
"""
import json, os

docs = []
queries = []
did = 0
qid = 0

def doc(scope, lang, tags, content):
    global did
    did += 1
    docs.append({
        "id": f"doc_{did:03d}",
        "scope": scope,
        "language": lang,
        "tags": tags,
        "content": content,
    })
    return f"doc_{did:03d}"

def q(query, expected, allowed, denied=None, desc="", lang=None):
    global qid
    qid += 1
    queries.append({
        "id": f"q_{qid:03d}",
        "query": query,
        "expected_doc_ids": expected if isinstance(expected, list) else [expected],
        "allowed_scopes": allowed if isinstance(allowed, list) else [allowed],
        "denied_scopes": denied or [],
        "description": desc,
        "language": lang or "en",
    })

# ============================================================
# ENGINEERING SCOPE (20 docs)
# ============================================================
d1 = doc("workspace_engineering", "en", ["auth","oauth","security"],
"Authentication and Token Management\n\nOur system uses OAuth 2.0 with PKCE for all client-side authentication. Access tokens expire after 15 minutes and are refreshed via the /auth/refresh endpoint using a secure HTTP-only refresh token cookie. Token revocation is handled through a Redis-backed denylist with a 5-minute TTL. For service-to-service auth, we use mTLS with SPIFFE/SPIRE for workload identity. The SSO provider supports SAML 2.0 and OpenID Connect 1.0.")

d2 = doc("workspace_engineering", "en", ["database","messaging","schema"],
"Database Schema for Messaging Service\n\nThe messaging service uses PostgreSQL 15 with the following core tables:\n- messages (id UUID PK, conversation_id FK, sender_id FK, body TEXT, created_at TIMESTAMPTZ, deleted_at TIMESTAMPTZ)\n- conversations (id UUID PK, type ENUM('direct','group'), title VARCHAR(200), created_at TIMESTAMPTZ)\n- conversation_members (conversation_id FK, user_id FK, role ENUM('member','admin'), joined_at TIMESTAMPTZ)\n- message_attachments (id UUID PK, message_id FK, file_url TEXT, file_type VARCHAR(50), file_size BIGINT)\n\nIndexes: messages(conversation_id, created_at DESC), conversation_members(user_id). Partitioning by month on messages.created_at.")

d3 = doc("workspace_engineering", "en", ["devops","kubernetes","ci-cd"],
"CI/CD Pipeline Configuration\n\nWe use GitHub Actions for CI and ArgoCD for CD. The pipeline stages are:\n1. lint (cargo clippy --workspace, 2 min)\n2. test (cargo test --workspace, 8 min)\n3. build (cargo build --release, 5 min)\n4. security-scan (trivy, cargo audit, 3 min)\n5. deploy-staging (argocd app sync, 2 min)\n6. integration-tests (pytest, 10 min)\n7. deploy-production (manual approval, argocd app sync)\n\nKubernetes clusters: 3 nodes (m5.large) in us-east-1, 2 nodes in ap-southeast-1. Auto-scaling: 3-10 nodes based on CPU > 70%.")

d4 = doc("workspace_engineering", "en", ["api","rate-limiting","gateway"],
"API Rate Limiting Policy\n\nRate limits are enforced at the API gateway (Kong) layer:\n- Free tier: 100 req/min, 1000 req/hour\n- Pro tier: 1000 req/min, 10000 req/hour\n- Enterprise: 10000 req/min, unlimited hourly\n\nLimits are per-API-key and per-IP (whichever is lower). 429 responses include Retry-After header. A token bucket algorithm with burst capacity of 2x the steady-state rate is used. Rate limit headers: X-RateLimit-Limit, X-RateLimit-Remaining, X-RateLimit-Reset.")

d5 = doc("workspace_engineering", "en", ["architecture","ai","runtime"],
"AI Runtime Architecture\n\nThe on-device AI runtime follows a 4-plane architecture:\n1. Safety Plane (deterministic, WASM-compatible): NFKC normalization, PII detection, URL screening, policy packs\n2. Context Plane (encrypted): SQLCipher FTS5, per-scope XChaCha20-Poly1305, dense embeddings (768-dim)\n3. Generation Plane: llama.cpp backend with Metal/Vulkan, JSON Schema grammar constraints, LoRA hot-swap\n4. Action Plane: typed artifact AST, ToolPlan validation, RBAC broker, commit tokens\n\nDevice tiers: Low, Medium, High. All tiers use the same 1.7B generative model (Bonsai-1.7B 1-bit) with task-specialized LoRA adapters. All tiers run the safety plane.")

d6 = doc("workspace_engineering", "vi", ["security","encryption","infrastructure"],
"Bảo mật và Mã hóa Infrastructure\n\nHệ thống sử dụng AES-256-GCM cho mã hóa at-rest và TLS 1.3 cho in-transit. Khóa chính được quản lý qua AWS KMS với rotation 90 ngày. Mỗi scope có khóa riêng (XChaCha20-Poly1305) được derive từ master key bằng HKDF-SHA256. Certificate rotation qua cert-manager 12 tháng. HSM được dùng cho khóa production.")

d7 = doc("workspace_engineering", "ja", ["performance","mobile","targets"],
"パフォーマンス目標 — モバイル\n\nモバイルデバイスのパフォーマンス目標:\n- TTFT (Time To First Token): Low tier < 3000ms, Medium < 1500ms, High < 800ms\n- デコード速度: Low N/A, Medium > 15 tok/s, High > 30 tok/s\n- メモリ使用量: Low < 512MB, Medium < 1.5GB, High < 3GB\n- バッテリー消費: 1時間の使用で < 15%\n- 起動時間: < 2秒\n\n測定は iPhone 12 (Low), iPhone 14 (Medium), iPhone 15 Pro (High) で実施。")

d8 = doc("workspace_engineering", "zh", ["architecture","microservices","grpc"],
"微服务架构与 gRPC\n\n系统采用 gRPC + Protocol Buffers 进行服务间通信。核心微服务:\n- auth-service (端口 50051): 认证、令牌管理\n- message-service (端口 50052): 消息收发、会话管理\n- media-service (端口 50053): 文件上传、缩略图\n- push-service (端口 50054): 推送通知\n- search-service (端口 50055): 全文搜索\n\n使用 Envoy 作为服务网格,支持负载均衡、熔断(阈值: 5% 错误率/10秒)、重试(最多3次)。服务发现通过 Consul。")

d9 = doc("workspace_engineering", "hi", ["marketing","aso","optimization"],
"ऐप स्टोर ऑप्टिमाइज़ेशन (ASO) रणनीति\n\nASO के लिए प्रमुख बिंदु:\n- शीर्षक: 30 अक्षर, ब्रांड नाम + प्रमुख कीवर्ड\n- उपशीर्षक: 30 अक्षर, द्वितीयक कीवर्ड\n- कीवर्ड फील्ड (iOS): 100 अक्षर, कॉमा-सेपरेटेड\n- विवरण: 4000 अक्षर, पहले 250 अक्षर सबसे महत्वपूर्ण\n- स्क्रीनशॉट: पहले 3 स्क्रीनशॉट 60% रूपांतरण प्रभाव\n- A/B परीक्षण: StoreKit प्रोमोशनल बैनर के साथ\n\nलक्ष्य: 10 प्रमुख खोज शब्दों में शीर्ष 5 रैंकिंग।")

d10 = doc("workspace_engineering", "en", ["observability","monitoring","tracing"],
"Observability Stack\n\nWe use OpenTelemetry for distributed tracing, Prometheus for metrics, and Loki for log aggregation. The three pillars:\n1. Metrics: RED (Rate, Errors, Duration) per service, histograms for latency, gauges for queue depth\n2. Traces: 1% sampling in prod, 100% in staging. Span attributes include user_id, scope, model_tier\n3. Logs: structured JSON, shipped via Fluent Bit to Loki. Retention: 7 days hot, 30 days warm, 1 year cold (S3)\n\nDashboards: Grafana with 4 panels — Service Health, Latency P50/P95/P99, Error Budget Burn, Model Performance.")

d11 = doc("workspace_engineering", "en", ["testing","qa","automation"],
"Test Strategy and Coverage\n\nTest pyramid:\n- Unit tests: 3000+ tests, run on every PR (cargo test, 8 min)\n- Integration tests: 200+ tests, run on merge to main (docker-compose, 15 min)\n- E2E tests: 50+ tests, run nightly (Playwright, 45 min)\n- Load tests: k6 scripts, run weekly (1000 concurrent users, 10 min)\n\nCoverage targets: 80% line coverage for core crates, 70% for bindings. Coverage measured via cargo-tarpaulin. Flaky test policy: 3 consecutive failures → quarantine → fix within 48h.")

d12 = doc("workspace_engineering", "en", ["incident","postmortem","oncall"],
"Incident Response and On-Call\n\nSeverity levels:\n- SEV1 (critical): full outage, data loss. Response: 5 min. Page all on-call engineers.\n- SEV2 (major): partial outage, degraded service. Response: 15 min. Page primary on-call.\n- SEV3 (minor): limited impact, workaround exists. Response: 1 hour. Slack notification.\n- SEV4 (low): cosmetic, no user impact. Response: next business day.\n\nPostmortem: required for SEV1/SEV2 within 48h. Blameless format. Action items tracked in Jira with 2-week SLA. Monthly review of all open action items.")

d13 = doc("workspace_engineering", "en", ["data","migration","etl"],
"Data Migration Plan: MongoDB to PostgreSQL\n\nMigration timeline (8 weeks):\nWeek 1-2: Schema design, dual-write proxy setup\nWeek 3-4: Backfill historical data (50M records, 48 hours via Spark)\nWeek 5-6: Read traffic cutover (10% → 50% → 100%)\nWeek 7: Write traffic cutover, MongoDB read-only\nWeek 8: Validation, MongoDB decommission\n\nRollback plan: dual-write proxy can redirect back to MongoDB within 5 minutes. Data consistency checks: hourly checksums on 1% sample. Zero-downtime requirement: met via dual-read phase.")

d14 = doc("workspace_engineering", "en", ["security","vulnerability","cve"],
"Vulnerability Management\n\nScanning: Trivy (container images), cargo-audit (Rust deps), npm audit (frontend), Snyk (all languages). Schedule: daily on main branch, weekly on release branches.\n\nSLA for remediation:\n- Critical (CVSS 9+): 24 hours\n- High (CVSS 7-8.9): 7 days\n- Medium (CVSS 4-6.9): 30 days\n- Low (CVSS <4): next release\n\nDisclosure: coordinated via security@kchat.com. Bug bounty: $500-$5000 via HackerOne. CVE assignment for confirmed criticals.")

d15 = doc("workspace_engineering", "en", ["caching","redis","performance"],
"Caching Strategy\n\nMulti-layer caching:\n1. CDN (Cloudflare): static assets, 1 year TTL, cache-key includes content hash\n2. Edge (Cloudflare Workers): API responses, 60s TTL, stale-while-revalidate\n3. Application (Redis cluster): session data (15 min), user profiles (5 min), feature flags (60s)\n4. Database (PostgreSQL): materialized views for analytics, refreshed hourly\n\nCache invalidation: tag-based (Cloudflare), event-driven (Redis pub/sub), scheduled (materialized views). Hit rate targets: CDN > 95%, Redis > 90%.")

d16 = doc("workspace_engineering", "en", ["i18n","localization","translation"],
"Internationalization (i18n) Guidelines\n\nSupported languages: en, vi, zh, ja, ko, es, fr, de, ar, hi, th, id, pt, tl (14 total).\n\nRules:\n- Use ICU MessageFormat for all user-facing strings\n- Never hardcode strings in source — use .ftl (Fluent) resource files\n- RTL support for ar (right-to-left layout)\n- Date/time: use Intl.DateTimeFormat, never hardcoded formats\n- Numbers/currency: use Intl.NumberFormat\n- Pluralization: CLDR rules per language\n- Translation pipeline: source en → Crowdin → 14 target languages → PR review → merge\n- Quality gate: native speaker review for all user-facing strings")

d17 = doc("workspace_engineering", "en", ["api","graphql","federation"],
"GraphQL Federation Setup\n\nWe use Apollo Federation 2 with 5 subgraphs:\n- auth-subgraph (User, AuthSession)\n- messaging-subgraph (Conversation, Message, Attachment)\n- media-subgraph (File, Thumbnail)\n- notification-subgraph (Notification, DeviceToken)\n- analytics-subgraph (Event, Metric)\n\nGateway: Apollo Router (Rust). Query complexity limit: 1000. Depth limit: 10. Persisted queries: enabled in production. DataLoader: batched and cached per-request. Schema registry: Apollo Studio, schema checks on every PR.")

d18 = doc("workspace_engineering", "en", ["deployment","blue-green","canary"],
"Deployment Strategies\n\nBlue-Green: used for database migrations. Two identical environments (blue/green). Switch via DNS (5 min TTL). Rollback: DNS switch back.\n\nCanary: used for application code. Traffic splitting: 5% → 25% → 50% → 100%, 1 hour per stage. Auto-rollback if error rate > 2% or P95 latency > baseline + 20%. Metrics: error rate, latency, conversion rate.\n\nFeature flags: LaunchDarkly. 200+ flags in production. Stale flag cleanup: quarterly. Flag naming: team_feature_purpose (e.g., messaging_reactions_v2).")

d19 = doc("workspace_engineering", "en", ["websocket","realtime","connection"],
"WebSocket Realtime Connection\n\nRealtime updates use WebSocket via a dedicated gateway (ws.kchat.com). Connection lifecycle:\n- Connect: wss://ws.kchat.com/v1/connect?token=<JWT>\n- Heartbeat: ping/pong every 30s, timeout 60s\n- Reconnection: exponential backoff (1s, 2s, 4s, 8s, 16s, max 30s)\n- Message format: JSON with type field (message_new, message_update, typing, presence)\n\nScaling: sticky sessions via Redis pub/sub fan-out. 50K concurrent connections per gateway node. Auto-scale: 3-20 nodes based on connection count. Geographic: 3 regions (us-east, eu-west, ap-southeast).")

d20 = doc("workspace_engineering", "en", ["accessibility","a11y","wcag"],
"Accessibility (a11y) Standards\n\nTarget: WCAG 2.2 AA compliance.\n\nKey requirements:\n- Color contrast: 4.5:1 for normal text, 3:1 for large text\n- Keyboard navigation: all interactive elements reachable via Tab, no keyboard traps\n- Screen reader: ARIA labels on all icon-only buttons, live regions for dynamic content\n- Focus management: visible focus rings, logical tab order, focus trap in modals\n- Touch targets: minimum 44x44px on mobile\n- Reduced motion: respect prefers-reduced-motion\n\nTesting: axe-core in CI, manual NVDA/VoiceOver testing per release, external audit annually.")

# ============================================================
# MARKETING SCOPE (15 docs)
# ============================================================
d21 = doc("workspace_marketing", "en", ["marketing","campaign","budget"],
"Q3 Marketing Campaign Budget\n\nTotal Q3 budget: $250,000\nAllocation:\n- Digital ads (Google, Meta, TikTok): $120,000 (48%)\n- Content marketing: $40,000 (16%)\n- Influencer partnerships: $50,000 (20%)\n- Events & sponsorships: $25,000 (10%)\n- Tools & analytics: $15,000 (6%)\n\nRegional split: North America 40%, Southeast Asia 30%, Europe 20%, Other 10%. KPIs: CAC < $25, LTV/CAC > 3, brand awareness +15% in target markets.")

d22 = doc("workspace_marketing", "en", ["brand","design","guidelines"],
"Brand Design Guidelines v3\n\nBrand colors:\n- Primary: #4F46E5 (Indigo)\n- Secondary: #10B981 (Emerald)\n- Accent: #F59E0B (Amber)\n- Neutral: #1F2937, #6B7280, #E5E7EB, #F9FAFB\n\nTypography: Inter (primary), Noto Sans (CJK fallback). Logo: minimum 24px height, clear space = 1x logo height. Icon style: Lucide icons, 1.5px stroke. Photography: authentic, candid, natural lighting. Illustration: flat, geometric, brand palette only.")

d23 = doc("workspace_marketing", "de", ["marketing","strategy","budget"],
"Marketing Strategie DACH Region\n\nQ3 Budget DACH: €75,000\n- Google Ads: €25,000\n- LinkedIn Ads: €15,000\n- Content (Blog, Whitepaper): €20,000\n- Events (München Meetup, Berlin Tech Day): €15,000\n\nZielgruppe: B2B SaaS Unternehmen, 50-500 Mitarbeiter. Key Messages: 'On-Device AI für maximale Privatsphäre', 'DSGVO-konform von Grund auf'. KPIs: MQLs +30%, CPL < €50, Pipeline €500K.")

d24 = doc("workspace_marketing", "es", ["marketing","latam","campaign"],
"Campaña Marketing LATAM Q3\n\nPresupuesto: $45,000 USD\n- Meta Ads (Instagram, WhatsApp): $18,000\n- Google Ads: $12,000\n- Influencers (México, Colombia, Brasil): $10,000\n- Contenido (Blog, YouTube): $5,000\n\nMercados clave: México, Colombia, Argentina, Chile. Mensaje: 'IA privada en tu dispositivo, sin servidores'. KPIs: CAC < $15, instalaciones orgánicas +40%, retención D30 > 35%.")

d25 = doc("workspace_marketing", "en", ["content","seo","blog"],
"Content Strategy and SEO\n\nBlog cadence: 2 posts/week (Tuesday, Thursday). Topics mapped to customer journey:\n- Awareness: 'What is on-device AI?', 'Privacy-first messaging apps'\n- Consideration: 'KChat vs WhatsApp: Privacy comparison', 'How AI works offline'\n- Decision: 'KChat enterprise deployment guide', 'Security audit results'\n\nSEO targets: 50 keywords in top 10, 10 keywords in top 3. Backlink goal: 20 DR40+ links per quarter. Content length: 1500-2500 words. Internal linking: 3+ per article. Schema markup: Article, FAQ, HowTo.")

d26 = doc("workspace_marketing", "en", ["social","community","engagement"],
"Social Media and Community Management\n\nChannels:\n- Twitter/X: 3 posts/day, tech-focused, 50K followers\n- LinkedIn: 1 post/day, B2B focused, 25K followers\n- Discord: community server, 10K members, 3 community managers\n- Reddit: r/kchat (5K), r/privacy (participation only)\n\nEngagement targets: response time < 4 hours, sentiment score > 0.7. Community programs: monthly dev challenge, quarterly contributor spotlight. Crisis protocol: SEV1 within 1 hour, SEV2 within 4 hours.")

d27 = doc("workspace_marketing", "en", ["analytics","attribution","funnel"],
"Marketing Analytics and Attribution\n\nAttribution model: data-driven (Shapley values), 30-day lookback window.\nFunnel stages:\n1. Impression → Click (CTR target: 2%)\n2. Click → Install (CVR target: 15%)\n3. Install → Sign up (CVR target: 60%)\n4. Sign up → Active (D7 retention target: 40%)\n5. Active → Paid (conversion target: 5%)\n\nTools: AppsFlyer (mobile attribution), GA4 (web), Mixpanel (product analytics), Amplitude (cohort analysis). Weekly dashboard: CAC by channel, LTV by cohort, payback period.")

d28 = doc("workspace_marketing", "fr", ["marketing","campaign","france"],
"Campagne Marketing France Q3\n\nBudget: €40,000\n- Google Ads: €12,000\n- LinkedIn Ads: €10,000\n- Content (Blog FR, Whitepaper): €10,000\n- Events (Paris Tech Summit, Lyon Meetup): €8,000\n\nPublic cible: entreprises tech, 50-300 employés. Message clé: 'IA sur appareil, conformité RGPD native'. KPIs: MQLs +25%, CPL < €60, NPS > 50.")

d29 = doc("workspace_marketing", "pt", ["marketing","brazil","campaign"],
"Campanha Marketing Brasil Q3\n\nOrçamento: R$150,000\n- Meta Ads (Instagram, WhatsApp): R$60,000\n- Google Ads: R$40,000\n- Influencers (Brasil): R$30,000\n- Conteúdo (Blog, YouTube): R$20,000\n\nMercado: Brasil, Portugal. Mensagem: 'IA privada no seu dispositivo, sem servidores'. KPIs: CAC < R$80, instalações orgânicas +50%, retenção D30 > 30%.")

d30 = doc("workspace_marketing", "en", ["email","lifecycle","automation"],
"Email Lifecycle Automation\n\nJourney stages:\n1. Welcome series (5 emails, days 0,1,3,7,14): onboarding tips, feature highlights\n2. Activation: inactivity trigger at day 3 → re-engagement email\n3. Weekly digest: every Monday, personalized content based on usage\n4. Win-back: day 30 inactivity → 3-email series with discount\n5. Upsell: trigger when user hits free tier limits\n\nTools: Customer.io for behavioral emails, SendGrid for transactional. Open rate target: 35%, CTR target: 8%. A/B testing: subject lines (50/50 split), send time optimization.")

d31 = doc("workspace_marketing", "en", ["pr","press","media"],
"PR and Media Relations\n\nPress kit: kchat.com/press (logos, screenshots, executive bios, fact sheet). Media contacts: 200+ journalists in tech, privacy, enterprise SaaS.\n\nPR cadence:\n- Product launches: embargoed press releases 48h before\n- Funding rounds: coordinated with TechCrunch, The Verge, Axios\n- Thought leadership: 2 op-eds per quarter from CEO/CTO\n- Awards: apply to Webby, Fast Company Innovation, Time100\n\nCrisis comms: holding statement within 1 hour, full response within 4 hours. Spokesperson: CEO for SEV1, VP Comms for SEV2-3.")

d32 = doc("workspace_marketing", "ko", ["marketing","korea","campaign"],
"한국 마케팅 캠페인 Q3\n\n예산: ₩50,000,000\n- 네이버 광고: ₩20,000,000\n- 카카오광고: ₩15,000,000\n- 인플루언서: ₩10,000,000\n- 콘텐츠(블로그, 유튜브): ₩5,000,000\n\n타겟: 20-40대 IT 직장인. 핵심 메시지: '기기 내 AI, 완벽한 프라이버시'. KPIs: CAC < ₩15,000, 유기적 설치 +35%, D30 유지 > 40%.")

d33 = doc("workspace_marketing", "ja", ["marketing","japan","campaign"],
"日本マーケティングキャンペーン Q3\n\n予算: ¥5,000,000\n- Google広告: ¥2,000,000\n- X(Twitter)広告: ¥1,500,000\n- インフルエンサー: ¥1,000,000\n- コンテンツ(ブログ、YouTube): ¥500,000\n\nターゲット: 20-40代ITエンジニア。主要メッセージ: 'オンデバイスAI、完全なプライバシー'。KPIs: CAC < ¥3,000、オーガニックインストール +30%、D30維持率 > 45%。")

d34 = doc("workspace_marketing", "th", ["marketing","thailand","campaign"],
"แคมเปญการตลาดไทย Q3\n\nงบประมาณ: ฿1,500,000\n- Meta Ads (Instagram, Facebook): ฿600,000\n- Google Ads: ฿400,000\n- อินฟลูเอนเซอร์: ฿300,000\n- คอนเทนต์ (บล็อก, YouTube): ฿200,000\n\nกลุ่มเป้าหมาย: วัยทำงาน 20-40 ปี สนใจเทคโนโลยี. ข้อความหลัก: 'AI บนอุปกรณ์ ความเป็นส่วนตัวสูงสุด'. KPIs: CAC < ฿500, ดาวน์โหลดออร์แกนิก +40%, อัตราการรักษา D30 > 35%.")

d35 = doc("workspace_marketing", "id", ["marketing","indonesia","campaign"],
"Kampanye Marketing Indonesia Q3\n\nBudget: Rp 500,000,000\n- Meta Ads (Instagram, WhatsApp): Rp 200,000,000\n- Google Ads: Rp 150,000,000\n- Influencer: Rp 100,000,000\n- Konten (Blog, YouTube): Rp 50,000,000\n\nTarget: profesional muda 20-35 tahun. Pesan utama: 'AI di perangkat, privasi penuh'. KPIs: CAC < Rp 50,000, instalasi organik +45%, retensi D30 > 30%.")

# ============================================================
# LEGAL SCOPE (15 docs)
# ============================================================
d36 = doc("workspace_legal", "en", ["legal","gdpr","compliance"],
"GDPR Compliance Framework\n\nLegal basis for processing:\n- Consent (Art. 6(1)(a)): explicit opt-in for marketing\n- Contract (Art. 6(1)(b)): service provision\n- Legitimate interest (Art. 6(1)(f)): security, fraud prevention\n\nData Subject Rights:\n- Access (Art. 15): 30-day response via privacy@kchat.com\n- Rectification (Art. 16): in-app profile editing\n- Erasure (Art. 17): 'right to be forgotten', 30-day completion\n- Portability (Art. 20): JSON export via settings\n- Objection (Art. 21): opt-out of profiling\n\nDPO: legal@kchat.com. Lead supervisory authority: CNIL (France). Records of processing: maintained in OneTrust.")

d37 = doc("workspace_legal", "en", ["legal","employment","contract"],
"Employment Contract Template\n\nStandard terms:\n- Probation: 3 months, either party may terminate with 2 weeks notice\n- Notice period: 30 days (after probation)\n- Non-compete: 12 months post-employment, limited to direct competitors\n- IP assignment: all work product assigned to company\n- Confidentiality: survives 5 years post-employment\n- Remote work: permitted, must maintain core hours (10am-3pm CET)\n\nBenefits: 25 days PTO, health insurance, €2000 learning budget, equipment provided. Equity: per offer letter, 4-year vest with 1-year cliff.")

d38 = doc("workspace_legal", "ar", ["legal","nda","confidentiality"],
"اتفاقية عدم الإفصاح (NDA)\n\nتسري هذه الاتفاقية لمدة 5 سنوات من تاريخ التوقيع.\nالمعلومات السرية تشمل:\n- الكود المصدري والبنية التحتية التقنية\n- بيانات العملاء والمستخدمين\n- الاستراتيجيات التجارية والمالية\n- خطط التسويق والعلاقات\n\nالاستثناءات: المعلومات المتاحة للعامة، المعلومات المعروفة مسبقاً، المعلومات المطلوبة قانونياً. عقوبة الإخلال: تعويض يحدد بحكم المحكمة.")

d39 = doc("workspace_legal", "fr", ["legal","license","software"],
"Contrat de Licence Logiciel\n\nLicence: perpétuelle, non-exclusive, non-transférable.\nChamp: utilisation interne, nombre d'utilisateurs selon forfait.\n\nGaranties:\n- Conformité aux spécifications: 90 jours\n- Absence de virus: garantie pour la durée du contrat\n- Support: inclus pour la 1ère année, renouvelable\n\nLimitation de responsabilité: plafond = frais de licence des 12 derniers mois. Propriété intellectuelle: concédant conserve tous les droits. Résiliation: manquement non réparé sous 30 jours. Loi applicable: droit français. Tribunal: Paris.")

d40 = doc("workspace_legal", "en", ["legal","dpa","data-processing"],
"Data Processing Agreement (DPA)\n\nPer GDPR Art. 28:\n- Processor: KChat Inc. processes personal data on behalf of Controller\n- Processing scope: messaging, AI inference, analytics as defined in MSA\n- Sub-processors: AWS, Cloudflare, Twilio (listed at kchat.com/subprocessors)\n- Data location: EU (Frankfurt) for EU controllers, US (Virginia) for US\n- Breach notification: 72 hours to Controller\n- Deletion on termination: within 90 days, certificate of destruction provided\n- Audit: Controller may audit once/year with 30 days notice\n\nInternational transfers: SCC (2021 modules) + TIA. Transfer tools: encryption, pseudonymization.")

d41 = doc("workspace_legal", "en", ["legal","terms","tos"],
"Terms of Service (ToS) v4\n\nKey provisions:\n- Eligibility: 16+ (GDPR Art. 8), 13+ with parental consent (COPPA)\n- Acceptable use: no spam, no illegal content, no harassment, no automated scraping\n- Account termination: 30-day notice, data export available\n- Service availability: 99.9% SLA for enterprise, best-effort for free\n- Dispute resolution: binding arbitration (AAA), class action waiver\n- Governing law: Delaware, USA (for US users); Ireland (for EU users)\n- Changes to ToS: 30-day notice via email and in-app\n\nLiability cap: $100 or 12 months of fees paid, whichever is greater.")

d42 = doc("workspace_legal", "en", ["legal","privacy","policy"],
"Privacy Policy v4\n\nData collected:\n- Account: email, display name, profile photo\n- Messages: end-to-end encrypted, not accessible to KChat\n- Metadata: timestamps, contact lists (encrypted)\n- Device: model, OS version, app version (for compatibility)\n- Telemetry: anonymized, opt-out available\n\nData retention:\n- Active accounts: indefinite\n- Deleted accounts: 30 days, then permanent deletion\n- Backups: 35 days\n- Legal holds: per litigation hold notice\n\nThird parties: AWS (hosting), Cloudflare (CDN), Twilio (SMS). Full list at kchat.com/privacy/third-parties.")

d43 = doc("workspace_legal", "en", ["legal","ccpa","privacy"],
"CCPA/CPRA Compliance\n\nCalifornia Consumer Rights:\n- Know: categories and specific pieces of personal info collected\n- Delete: request deletion of personal info\n- Correct: request correction of inaccurate personal info\n- Opt-out: sale or sharing of personal info (we don't sell)\n- Limit: use of sensitive personal info\n- Non-discrimination: equal service regardless of rights exercised\n\nResponse timeline: 45 days (extendable by 45 days with notice). Verification: matching email + phone. Authorized agents: with written permission. DSAR portal: privacy.kchat.com/ca.")

d44 = doc("workspace_legal", "vi", ["legal","luật","bảo mật"],
"Chính sách Bảo mật theo Luật Việt Nam\n\nTheo Nghị định 13/2023/NĐ-CP về bảo vệ dữ liệu cá nhân:\n- Cơ sở xử lý: sự đồng ý (điều 7), nghĩa vụ hợp đồng (điều 8)\n- Quyền của chủ dữ liệu: truy cập, sao chép, sửa đổi, xóa, ngừng xử lý\n- Thông báo vi phạm: 72 giờ cho cơ quan quản lý\n- Đại diện bảo vệ dữ liệu: legal@kchat.com\n- Chuyển dữ liệu ra nước ngoài: đảm bảo mức bảo vệ tương đương\n\nLưu trữ: tài khoản hoạt động không thời hạn, tài khoản xóa 30 ngày. Khiếu nại: gửi đến legal@kchat.com hoặc Cục An toàn thông tin.")

d45 = doc("workspace_legal", "zh", ["legal","数据保护","隐私"],
"数据保护政策 — 中国\n\n根据《个人信息保护法》(PIPL):\n- 处理依据: 同意(第13条), 履行合同(第13条)\n- 个人权利: 知情、决定、查阅、复制、更正、删除\n- 跨境传输: 需通过安全评估或标准合同\n- 数据本地化: 用户数据存储在中国境内服务器\n- 个人信息保护负责人: legal@kchat.com\n\n数据保留: 活跃账户无限期, 注销账户30天后永久删除. 投诉: legal@kchat.com 或国家网信办.")

d46 = doc("workspace_legal", "ja", ["legal","個人情報保護","プライバシー"],
"個人情報保護方針 — 日本\n\n個人情報保護法 (APPI) に基づく:\n- 利用目的: サービス提供、セキュリティ、改善分析\n- 第三者提供: AWS、Cloudflare、Twilio（開示: kchat.com/privacy/third-parties）\n- 開示・訂正・利用停止: privacy@kchat.com、30日以内に対応\n- データ保管: アクティブ無期限、削除後30日\n- クロスボーダー移転: 標準契約条項 (SCC) + 移転影響評価\n\n個人情報保護管理者: legal@kchat.com. 苦情処理: 30日以内に対応.")

d47 = doc("workspace_legal", "ko", ["legal","개인정보보호","한국"],
"개인정보처리방침 — 한국\n\n개인정보보호법에 따라:\n- 처리 근거: 동의(제15조), 계약 이행(제15조)\n- 정보주체 권리: 열람, 정정, 삭제, 처리정지\n- 제3자 제공: AWS, Cloudflare, Twilio (kchat.com/privacy/third-parties)\n- 보관: 활성 계정 무기한, 탈퇴 후 30일\n- 파기: 30일 내 영구 삭제\n\n개인정보보호책임자: legal@kchat.com. 권리 행사: privacy@kchat.com, 10일 이내 응답.")

d48 = doc("workspace_legal", "hi", ["legal","dpdp","india"],
"डिजिटल पर्सनल डेटा प्रोटेक्शन (DPDP) अधिनियम\n\nDPDP Act 2023 के अनुसार:\n- संसाधन का आधार: स्पष्ट सहमति या वैध उपयोग\n- डेटा प्रिंसिपल अधिकार: पहुंच, सुधार, मिटाना, शिकायत\n- डेटा फिड्यूशरी: KChat Inc., legal@kchat.com\n- अंतर्राष्ट्रीय स्थानांतरण: अनुमत (नियम के अधीन)\n- उल्लंघन सूचना: 72 घंटे के भीतर डेटा प्रोटेक्शन बोर्ड को\n\nसंरक्षण: सक्रिय खाता अनिश्चित काल, हटाए गए खाते 30 दिन। शिकायत: legal@kchat.com.")

d49 = doc("workspace_legal", "en", ["legal","ip","trademark"],
"Intellectual Property and Trademark Policy\n\nTrademarks: 'KChat', 'KChat AI', 'On-Device AI' are registered in US, EU, JP, KR, CN.\n\nOpen source: kchat-core is MIT licensed. kchat-safety is Apache 2.0. kchat-generation is proprietary. Contributing: CLA required for all contributions.\n\nPatent policy: defensive only. We will not assert patents against open-source implementations. Patent troll protection: LOT Network member.\n\nCopyright: all content © KChat Inc. unless otherwise noted. DMCA: notices to copyright@kchat.com, 48-hour response.")

d50 = doc("workspace_legal", "en", ["legal","sla","enterprise"],
"Enterprise SLA Agreement\n\nService Level: 99.9% uptime per quarter (excluding scheduled maintenance).\n\nCredits:\n- 99.0-99.9%: 10% credit\n- 95.0-99.0%: 25% credit\n- <95.0%: 50% credit\n\nExclusions: force majeure, customer-caused issues, internet outages, planned maintenance (48h notice).\n\nSupport tiers:\n- Standard: 8x5, email, 4-hour response\n- Premium: 24x7, email + phone, 1-hour response\n- Enterprise: 24x7, dedicated CSM, 15-min response, quarterly reviews\n\nUptime tracking: status.kchat.com, third-party monitored by Pingdom.")

# ============================================================
# PERSONAL SCOPE (10 docs)
# ============================================================
d51 = doc("workspace_personal", "en", ["personal","travel","preferences"],
"Travel Preferences\n\nPreferred airlines: Singapore Airlines, ANA, Lufthansa (Star Alliance). Seat: aisle, exit row when available. Meal: vegetarian. Hotel preferences: Marriott Bonvoy Platinum, prefer properties with gym and executive lounge. Rental car: compact or mid-size, automatic transmission. Passport number: P1234567A (expires 2028). Known Traveler Number: 1234567 (Global Entry).")

d52 = doc("workspace_personal", "en", ["personal","health","fitness"],
"Health and Fitness Goals 2026\n\nCurrent: 75kg, 178cm, BMI 23.7. Goal: 72kg by December.\n\nRoutine:\n- Mon/Wed/Fri: 5km run (target < 25 min)\n- Tue/Thu: strength training (push/pull/legs split)\n- Sat: yoga or hiking\n- Sun: rest\n\nDiet: 2000 cal/day, 150g protein. Supplements: vitamin D, omega-3, creatine. Sleep: 7.5 hours target, lights out by 11pm. Last checkup: January 2026, all normal.")

d53 = doc("workspace_personal", "en", ["personal","reading","books"],
"Reading List 2026\n\nCompleted:\n1. 'Thinking, Fast and Slow' — Kahneman (cognitive biases)\n2. 'Sapiens' — Harari (human history)\n3. 'The Pragmatic Programmer' — Hunt & Thomas (software)\n\nCurrently reading:\n4. 'Designing Data-Intensive Applications' — Kleppmann\n\nUp next:\n5. 'The Lean Startup' — Ries\n6. 'Clean Architecture' — Martin\n7. 'Atomic Habits' — Clear\n\nGoal: 24 books in 2026. Current pace: 18 books through August.")

d54 = doc("workspace_personal", "vi", ["personal","gia đình","liên hệ"],
"Thông tin Gia đình và Liên hệ Khẩn cấp\n\nLiên hệ khẩn cấp:\n- Vợ: Nguyễn Thị Lan, SĐT: 0912-345-678\n- Mẹ: Phạm Thị Mai, SĐT: 0987-654-321\n- Bác sĩ: BS. Trần Văn Hùng, 0901-234-567\n\nNgười phụ thuộc: 2 con (7 tuổi, 4 tuổi). Bảo hiểm: Bảo Việt, số BH: BV2026-7890. Nhóm máu: O+. Dị ứng: không. Thuốc thường dùng: không.")

d55 = doc("workspace_personal", "ja", ["個人","趣味","音楽"],
"趣味と音楽\n\n好きなジャンル: ジャズ、ローファイ、クラシック\nお気に入りアーティスト:\n- 坂本龍一\n- Miles Davis\n- Bill Evans\n- Nujabes\n\n楽器: ピアノ（中級）、ギター（初級）。最近の発見: Mac DeMarco、Snarky Puppy。Spotifyプレイリスト: 'Jazz for Coding'（フォロワー1.2K）。コンサート年間予算: ¥50,000。")

d56 = doc("workspace_personal", "zh", ["个人","投资","理财"],
"投资理财组合\n\n当前持仓:\n- 股票: 60% (VTI 40%, VOO 20%)\n- 债券: 20% (BND)\n- 加密货币: 10% (BTC 7%, ETH 3%)\n- 现金: 10%\n\n投资策略: 长期持有, 定投每月$2000, 再平衡每年一次. 目标: 55岁退休, 4%提取率. 当前年龄: 35. 预期年化: 7%. 退休目标: $2M. 当前进度: $450K (22.5%).")

d57 = doc("workspace_personal", "en", ["personal","education","courses"],
"Professional Development Plan 2026\n\nCourses:\n- Q1: 'Advanced Rust' (O'Reilly, 20 hours) — completed\n- Q2: 'Distributed Systems' (MIT 6.824, 40 hours) — in progress\n- Q3: 'Machine Learning Engineering' (Coursera, 30 hours)\n- Q4: 'Leadership for Engineers' (LinkedIn Learning, 15 hours)\n\nConferences: RustConf (September, Seattle), KubeCon (November, Chicago). Budget: $5000 (company reimburses). Goal: 100 learning hours in 2026.")

d58 = doc("workspace_personal", "en", ["personal","recipes","cooking"],
"Favorite Recipes\n\n1. Thai Green Curry (vegetarian):\n   - 2 tbsp green curry paste, 1 can coconut milk, tofu, bell pepper, basil\n   - 20 min, serves 4\n\n2. Vietnamese Pho:\n   - Beef bones (8h simmer), rice noodles, star anise, cinnamon, lime\n   - 8h + 30 min, serves 6\n\n3. Japanese Curry Rice:\n   - Curry roux (Golden), onion, potato, carrot, chicken\n   - 45 min, serves 4\n\n4. Korean Bibimbap:\n   - Rice, spinach, carrot, bean sprouts, egg, gochujang\n   - 30 min, serves 2")

d59 = doc("workspace_personal", "es", ["personal","familia","contacto"],
"Información Familiar y Contacto de Emergencia\n\nContactos de emergencia:\n- Esposa: María García, +34 612-345-678\n- Madre: Carmen Ruiz, +34 678-901-234\n- Médico: Dr. López, +34 623-456-789\n\nDependientes: 2 hijos (8 y 5 años). Seguro: Sanitas, póliza SA2026-4567. Grupo sanguíneo: A+. Alergias: penicilina. Medicación: ninguno.")

d60 = doc("workspace_personal", "ko", ["개인","여가","여행"],
"여가 및 여행 계획\n\n2026년 여행 계획:\n- 봄: 제주도 3박4일 (가족)\n- 여름: 일본 오사카 5박6일 (가족)\n- 가을: 베트남 다낭 4박5일 (부부)\n- 겨울: 스위스 체르마트 7박8일 (가족)\n\n여행 예산: ₩15,000,000/년. 항공사: 대한항공, 아시아나 (스카이패스 골드). 호텔: 신라, 메리어트 (골드 회원).")

# ============================================================
# FINANCE SCOPE (10 docs)
# ============================================================
d61 = doc("workspace_finance", "en", ["finance","budget","q3"],
"Q3 2026 Financial Budget\n\nTotal budget: $1,200,000\n- Engineering: $600,000 (50%)\n- Marketing: $250,000 (21%)\n- Operations: $150,000 (12.5%)\n- Legal & Compliance: $100,000 (8.3%)\n- Finance & Admin: $60,000 (5%)\n- Contingency: $40,000 (3.3%)\n\nRevenue target: $800,000. Burn rate: $400K/month. Runway: 18 months. Key metrics: ARR $3.2M, growth rate 15% MoM, gross margin 78%.")

d62 = doc("workspace_finance", "en", ["finance","revenue","forecast"],
"Revenue Forecast 2026 H2\n\nQ3: $800K (SaaS $600K, Enterprise $200K)\nQ4: $1.1M (SaaS $750K, Enterprise $350K)\n\nAssumptions:\n- New customers: 50/month (SaaS), 2/month (Enterprise)\n- Churn: 3% monthly (SaaS), 1% monthly (Enterprise)\n- ARPU: $50/month (SaaS), $5000/month (Enterprise)\n- Upsell: 10% of base upgrade to higher tier\n\nRisk factors: economic downturn (30% probability, -20% impact), competitor pricing (20%, -10%), regulatory change (10%, -15%).")

d63 = doc("workspace_finance", "en", ["finance","investors","cap-table"],
"Cap Table and Investor Relations\n\nCurrent cap table:\n- Founders: 55%\n- Series A (2024): 25% ($5M, lead: Accel)\n- Series B (2026): 15% ($20M, lead: Sequoia)\n- ESOP: 5%\n\nInvestors:\n- Accel: board seat, monthly updates\n- Sequoia: board seat, monthly updates\n- Angel investors (5): quarterly updates\n\nNext round: Series C planned Q2 2027, target $50M, valuation $250M. Use of funds: international expansion, ML research, hiring (50 FTE).")

d64 = doc("workspace_finance", "en", ["finance","expenses","policy"],
"Expense Policy\n\nReimbursable expenses:\n- Travel: economy class (business class for flights > 6 hours), max $500/night hotel\n- Meals: $50/day domestic, $75/day international\n- Ground transport: Uber/Lyft, public transit preferred\n- Equipment: laptops every 3 years, monitors every 5 years\n- Professional development: $2000/year, pre-approved\n- Team events: $50/person, max $500/event\n\nNon-reimbursable: personal expenses, alcohol (unless client dinner), parking tickets. Submission: Expensify within 30 days. Approval: manager < $500, VP > $500.")

d65 = doc("workspace_finance", "en", ["finance","payroll","benefits"],
"Payroll and Benefits Summary\n\nPayroll cycle: semi-monthly (15th and last day). Currency: USD (US), EUR (EU), SGD (APAC).\n\nBenefits:\n- Health: Aetna PPO (US), Allianz (EU), full premium covered for employee, 50% for dependents\n- Dental & Vision: included\n- 401(k): 4% match (US), pension contribution (EU)\n- PTO: 25 days + 10 holidays\n- Parental leave: 16 weeks (primary), 8 weeks (secondary)\n- Equipment: $2000 setup budget, $500/year maintenance\n- Learning: $2000/year\n- Wellness: $500/year (gym, meditation apps)")

d66 = doc("workspace_finance", "en", ["finance","tax","compliance"],
"Tax Compliance\n\nJurisdictions:\n- US: Federal (Delaware C-Corp), California, New York, Texas\n- EU: Ireland (EU HQ), Germany, France, Netherlands\n- APAC: Singapore, Japan, Australia\n- Other: UK, Brazil\n\nSales tax/VAT: registered in all applicable jurisdictions. Stripe Tax for automated collection. Filing: monthly (large states), quarterly (small). Transfer pricing: Arm's length, documented in Master File + Local Files. BEPS Pillar 2: compliant (revenue < €750M threshold).")

d67 = doc("workspace_finance", "en", ["finance","audit","internal"],
"Internal Audit Plan 2026\n\nAudits scheduled:\n- Q1: SOC 2 Type II (external, Prescient Assurance)\n- Q2: GDPR compliance review (external, DPO Lane Clark)\n- Q3: Financial audit (external, PwC)\n- Q4: Security penetration test (external, Bishop Fox)\n\nInternal audits (quarterly):\n- Access controls review (IT)\n- Expense audit (Finance, 10% sample)\n- Vendor risk assessment (Procurement)\n- Data retention compliance (Legal)\n\nFindings tracking: Jira, 30-day remediation SLA. Board reporting: quarterly audit committee.")

d68 = doc("workspace_finance", "en", ["finance","procurement","vendors"],
"Procurement and Vendor Management\n\nVendor tiers:\n- Tier 1 (>$100K/year): AWS, Cloudflare, Twilio, Datadog. Annual review, SOC 2 required.\n- Tier 2 ($10K-$100K): GitHub, Slack, Notion, Figma. Biennial review.\n- Tier 3 (<$10K): ad-hoc, manager approval.\n\nProcess:\n1. Request via procurement portal\n2. Security review (Tier 1/2)\n3. Legal review (DPA, NDA)\n4. Price negotiation (3 quotes for >$50K)\n5. PO issuance\n6. Annual review\n\nDiversity: 20% of spend with diverse suppliers by 2027. Sustainability: carbon footprint tracking for Tier 1 vendors.")

d69 = doc("workspace_finance", "en", ["finance","forecast","cashflow"],
"Cash Flow Forecast\n\nStarting cash (Aug 2026): $7.2M\n\nMonthly:\n- Revenue: $267K avg\n- Expenses: $400K avg (payroll $280K, infra $50K, other $70K)\n- Net burn: -$133K/month\n\nProjected:\n- Sep: $7.07M\n- Dec: $6.67M\n- Mar 2027: $6.27M\n- Jun 2027: $5.87M (Series C close, +$50M)\n\nSensitivity: +20% revenue growth → runway 30 months. -20% revenue → runway 12 months. Cost cut scenario: reduce burn to $80K/month (layoffs, reduce infra).")

d70 = doc("workspace_finance", "en", ["finance","esg","sustainability"],
"ESG and Sustainability Report 2026\n\nEnvironmental:\n- Carbon: 450 tons CO2e (Scope 1: 0, Scope 2: 280, Scope 3: 170)\n- Offset: 100% via Climeworks DAC + reforestation (Madagascar)\n- Energy: 100% renewable (AWS sustainable regions)\n- Goal: Net zero by 2028, 50% reduction by 2027\n\nSocial:\n- Diversity: 42% women, 18% URM in workforce. 35% women in leadership.\n- Pay equity: audited annually, <1% gap adjusted\n- Volunteer: 2 days/year paid, 85% participation\n\nGovernance:\n- Board: 7 members, 3 independent, 2 women\n- Ethics hotline: anonymous, 24/7\n- Anti-corruption: training annually, 100% completion")

# ============================================================
# HR SCOPE (10 docs)
# ============================================================
d71 = doc("workspace_hr", "en", ["hr","hiring","process"],
"Hiring Process and Pipeline\n\nStages:\n1. Application review (recruiter, 2 days)\n2. Phone screen (recruiter, 30 min)\n3. Technical screen (engineer, 45 min)\n4. Onsite (4 interviews, 4 hours): coding, system design, behavioral, values\n5. Hiring committee (2 days)\n6. Offer (3 days)\n\nTime-to-hire target: 21 days. Current avg: 28 days. Open roles: 15 (8 engineering, 4 marketing, 3 operations). Pipeline: 240 candidates, 60 in process. Source mix: 40% referral, 30% LinkedIn, 20% job boards, 10% outbound.")

d72 = doc("workspace_hr", "en", ["hr","onboarding","checklist"],
"Onboarding Checklist (New Hire)\n\nWeek 0 (pre-start):\n- Equipment shipped (laptop, monitor, peripherals)\n- Accounts created (Google, Slack, GitHub, Notion)\n- Welcome email with first-day agenda\n\nWeek 1:\n- Day 1: orientation, HR setup, team lunch\n- Day 2-3: product overview, architecture deep-dive\n- Day 4-5: pair programming, first PR\n\nWeek 2-4:\n- Shadow on-call rotation\n- Complete security training\n- Meet with mentor weekly\n- First independent feature\n\n30-day check-in: manager + HR. 60-day: peer feedback. 90-day: performance review.")

d73 = doc("workspace_hr", "en", ["hr","performance","review"],
"Performance Review Process\n\nCycle: semi-annual (January, July).\n\nProcess:\n1. Self-assessment (employee, 1 week)\n2. Peer feedback (3-5 peers, 1 week)\n3. Manager assessment (1 week)\n4. Calibration (leadership, 1 week)\n5. Review meeting (30 min)\n6. Goal setting for next cycle\n\nRating scale: 1 (below) / 2 (meets) / 3 (exceeds) / 4 (outstanding). Distribution: 10% 4s, 25% 3s, 55% 2s, 10% 1s. Compensation: merit increase (2-8%), equity refresh (top 25%). Promotion: requires 2 consecutive 3+ ratings.")

d74 = doc("workspace_hr", "en", ["hr","culture","values"],
"Company Values and Culture\n\nCore values:\n1. Privacy First: user privacy is non-negotiable in every decision\n2. Ship with Quality: we ship fast but never compromise on safety\n3. Radical Transparency: default to open, share context generously\n4. Customer Obsession: we build for users, not for ourselves\n5. Continuous Learning: everyone is a teacher and a student\n\nCulture rituals:\n- Friday demos: 30 min, anyone can present\n- Monthly all-hands: company metrics + Q&A\n- Quarterly hackathon: 2 days, cross-team\n- Anonymous feedback: monthly pulse survey, results public\n- No-meeting Wednesdays: deep work focus")

d75 = doc("workspace_hr", "en", ["hr","compensation","bands"],
"Compensation Bands 2026\n\nEngineering:\n- L1 (Junior): $90K-$120K base + 0.05% equity\n- L2 (Mid): $130K-$170K base + 0.1% equity\n- L3 (Senior): $180K-$230K base + 0.2% equity\n- L4 (Staff): $240K-$300K base + 0.35% equity\n- L5 (Principal): $310K-$400K base + 0.5% equity\n\nMarketing:\n- M1: $80K-$110K, M2: $120K-$160K, M3: $170K-$220K\n\nSales:\n- Base + commission (OTE: $120K-$300K)\n\nLocation adjustment: SF/NYC 100%, Seattle/Austin 95%, Remote US 85%, International 60-80%. Review: annual, market data from Radford, Pave.")

d76 = doc("workspace_hr", "en", ["hr","diversity","dei"],
"DEI (Diversity, Equity, Inclusion) Program\n\nGoals (2026):\n- Women in engineering: 30% (current: 24%)\n- URM in engineering: 15% (current: 10%)\n- Women in leadership: 40% (current: 35%)\n- Pay equity gap: <1% (current: 0.5%)\n\nInitiatives:\n- Partnerships: Women Who Code, AfroTech, Lesbians Who Tech\n- Interview panel diversity: at least 1 URM interviewer\n- Bias training: required for all hiring managers\n- Mentorship: BIPOC mentorship program (20 pairs)\n- Employee Resource Groups: Women in Tech, Pride, BIPOC, Veterans (4 ERGs)\n\nReporting: quarterly to board, annual public DEI report.")

d77 = doc("workspace_hr", "en", ["hr","offboarding","exit"],
"Offboarding Process\n\nResignation:\n1. Employee submits notice (30 days minimum)\n2. Manager acknowledges, notifies HR\n3. Knowledge transfer plan (1 week)\n4. Exit interview (HR, 45 min)\n5. Equipment return (laptop, badge, keys)\n6. Account deactivation (last day, 5pm)\n7. Final paycheck + PTO payout (within 7 days)\n\nInvoluntary:\n1. Performance plan (30/60/90 days)\n2. If no improvement: termination with 60 days severance\n3. Outplacement services (3 months)\n4. Reference: confirm dates and title only\n\nAlumni: LinkedIn group, quarterly newsletter, annual reunion. Rehire policy: welcome back within 2 years.")

d78 = doc("workspace_hr", "en", ["hr","training","development"],
"Training and Development Programs\n\nTechnical:\n- Rust bootcamp (5 days, quarterly)\n- System design workshop (2 days, monthly)\n- Security awareness (1 hour, annual, required)\n- AI/ML fundamentals (10 hours, self-paced)\n\nLeadership:\n- New manager bootcamp (3 days)\n- Leadership circle (monthly, 1 hour)\n- Executive coaching (for L4+, 6 sessions)\n\nSoft skills:\n- Effective communication (2 hours, quarterly)\n- Conflict resolution (2 hours, quarterly)\n- Time management (1 hour, self-paced)\n\nBudget: $2000/employee/year + $50K company-wide for external speakers. Platform: LinkedIn Learning, O'Reilly, Coursera.")

d79 = doc("workspace_hr", "en", ["hr","remote","policy"],
"Remote Work Policy\n\nKChat is remote-first with optional hubs (SF, Singapore, Berlin).\n\nGuidelines:\n- Core hours: 10am-3pm in your timezone (overlap window)\n- Async communication: default to written (Slack, Notion), minimize meetings\n- Meeting-free: Wednesdays (no internal meetings)\n- Video on: optional, camera-free culture\n- Equipment: $2000 setup budget + $500/year maintenance + $50/month internet stipend\n- Coworking: $200/month reimbursement\n- Travel: 2 company offsites/year (fully paid), team meetup quarterly\n\nTime zones: teams organized to minimize span (max 5 hours difference). On-call: follows sun model.")

d80 = doc("workspace_hr", "en", ["hr","benefits","perks"],
"Benefits and Perks Summary\n\nHealth & Wellness:\n- Medical, dental, vision: 100% covered (employee), 50% (dependents)\n- Mental health: 10 free therapy sessions/year (Spring Health)\n- Wellness stipend: $500/year (gym, apps, equipment)\n- Ergonomic assessment: free for all new hires\n\nTime Off:\n- PTO: 25 days + 10 holidays + unlimited sick leave\n- Parental leave: 16 weeks primary, 8 weeks secondary\n- Sabbatical: 4 weeks after 5 years\n\nFinancial:\n- 401(k): 4% match (US), pension (EU)\n- Equity: all employees, 4-year vest, 1-year cliff\n- ESPP: 15% discount, $25K max\n\nOther:\n- $2000 learning budget\n- Free books (any technical book reimbursed)\n- Conference budget: $3000/year\n- Home office: $2000 setup + $500/year")

# ============================================================
# QUERIES (40+)
# ============================================================

# --- Simple keyword queries (existing + new) ---
q("authentication token expiry", d1, "workspace_engineering", desc="Simple keyword search in engineering scope")
q("database schema messaging", d2, "workspace_engineering", desc="Database schema lookup")
q("marketing budget Southeast Asia", d21, "workspace_marketing", desc="Marketing budget query")
q("passport number travel preferences", d51, "workspace_personal", desc="Personal travel info")
q("encryption AES TLS", d6, "workspace_engineering", desc="Vietnamese encryption doc", lang="vi")
q("rate limiting API", d4, "workspace_engineering", desc="API rate limiting policy")
q("GDPR compliance data processing", d36, "workspace_legal", desc="GDPR compliance framework")
q("AI runtime architecture on-device", d5, "workspace_engineering", desc="AI runtime architecture")
q("パフォーマンス モバイル 目標", d7, "workspace_engineering", desc="Japanese performance targets", lang="ja")
q("non-compete confidentiality employment", d37, "workspace_legal", desc="Employment contract terms")
q("brand color typography guidelines", d22, "workspace_marketing", desc="Brand design guidelines")
q("微服务 gRPC 架构", d8, "workspace_engineering", desc="Chinese microservices architecture", lang="zh")
q("اتفاقية عدم الإفصاح سرية", d38, "workspace_legal", desc="Arabic NDA document", lang="ar")
q("Marketing Strategie DACH Budget", d23, "workspace_marketing", desc="German marketing strategy", lang="de")
q("ऐप स्टोर ऑप्टिमाइज़ेशन ASO", d9, "workspace_engineering", desc="Hindi ASO strategy", lang="hi")
q("contrat licence logiciel garantie", d39, "workspace_legal", desc="French software license", lang="fr")

# --- ACL tests (denied scopes) ---
q("authentication security", [], "workspace_personal", denied=["workspace_engineering"], desc="ACL test: query engineering topic but only personal scope allowed")
q("marketing budget", [], "workspace_engineering", denied=["workspace_marketing"], desc="ACL test: query marketing but only engineering scope allowed")
q("GDPR compliance", [], "workspace_personal", denied=["workspace_legal"], desc="ACL test: query legal but only personal scope allowed")
q("salary compensation bands", [], "workspace_marketing", denied=["workspace_hr"], desc="ACL test: query HR but only marketing scope allowed")

# --- Multi-hop queries (expect 2+ docs) ---
q("authentication and rate limiting", [d1, d4], "workspace_engineering", desc="Multi-hop: auth + rate limiting")
q("deployment and monitoring", [d3, d10], "workspace_engineering", desc="Multi-hop: CI/CD + observability")
q("GDPR and CCPA compliance", [d36, d43], "workspace_legal", desc="Multi-hop: GDPR + CCPA")
q("marketing budget and brand guidelines", [d21, d22], "workspace_marketing", desc="Multi-hop: budget + brand")
q("hiring and onboarding process", [d71, d72], "workspace_hr", desc="Multi-hop: hiring + onboarding")
q("revenue forecast and cash flow", [d62, d69], "workspace_finance", desc="Multi-hop: revenue + cashflow")
q("caching and WebSocket realtime", [d15, d19], "workspace_engineering", desc="Multi-hop: caching + websocket")

# --- Cross-language queries ---
q("security encryption infrastructure", d6, "workspace_engineering", desc="Cross-lang: English query, Vietnamese doc")
q("performance mobile targets", d7, "workspace_engineering", desc="Cross-lang: English query, Japanese doc")
q("microservices gRPC architecture", d8, "workspace_engineering", desc="Cross-lang: English query, Chinese doc")
q("app store optimization ASO", d9, "workspace_engineering", desc="Cross-lang: English query, Hindi doc")
q("NDA confidentiality agreement", d38, "workspace_legal", desc="Cross-lang: English query, Arabic doc")
q("software license warranty", d39, "workspace_legal", desc="Cross-lang: English query, French doc")
q("marketing strategy budget", d23, "workspace_marketing", desc="Cross-lang: English query, German doc")
q("data protection privacy law", d44, "workspace_legal", desc="Cross-lang: English query, Vietnamese legal doc")
q("personal data protection PIPL", d45, "workspace_legal", desc="Cross-lang: English query, Chinese legal doc")
q("personal information protection APPI", d46, "workspace_legal", desc="Cross-lang: English query, Japanese legal doc")
q("personal data privacy Korea", d47, "workspace_legal", desc="Cross-lang: English query, Korean legal doc")
q("digital personal data protection India", d48, "workspace_legal", desc="Cross-lang: English query, Hindi legal doc")

# --- Semantic queries (paraphrased, not keyword match) ---
q("how do users log in securely", d1, "workspace_engineering", desc="Semantic: paraphrased auth query")
q("what are the brand colors", d22, "workspace_marketing", desc="Semantic: paraphrased brand query")
q("how is data protected in Europe", d36, "workspace_legal", desc="Semantic: paraphrased GDPR query")
q("what happens when someone quits", d77, "workspace_hr", desc="Semantic: paraphrased offboarding query")
q("how much money do we have", d69, "workspace_finance", desc="Semantic: paraphrased cash flow query")
q("how do we deploy code", d3, "workspace_engineering", desc="Semantic: paraphrased CI/CD query")
q("what tools do we use for monitoring", d10, "workspace_engineering", desc="Semantic: paraphrased observability query")
q("how do we handle incidents", d12, "workspace_engineering", desc="Semantic: paraphrased incident response query")

# --- New scope queries ---
q("Q3 financial budget", d61, "workspace_finance", desc="Finance budget query")
q("company values culture", d74, "workspace_hr", desc="HR culture values query")
q("compensation bands salary", d75, "workspace_hr", desc="HR compensation query")
q("remote work policy guidelines", d79, "workspace_hr", desc="HR remote work query")
q("expense reimbursement policy", d64, "workspace_finance", desc="Finance expense policy query")
q("investor relations cap table", d63, "workspace_finance", desc="Finance investor query")
q("ESG sustainability carbon", d70, "workspace_finance", desc="Finance ESG query")
q("diversity equity inclusion program", d76, "workspace_hr", desc="HR DEI query")
q("training development programs", d78, "workspace_hr", desc="HR training query")
q("accessibility WCAG standards", d20, "workspace_engineering", desc="Engineering a11y query")
q("internationalization i18n localization", d16, "workspace_engineering", desc="Engineering i18n query")
q("GraphQL federation setup", d17, "workspace_engineering", desc="Engineering GraphQL query")
q("vulnerability management CVE", d14, "workspace_engineering", desc="Engineering security query")
q("test coverage automation", d11, "workspace_engineering", desc="Engineering testing query")
q("data migration MongoDB PostgreSQL", d13, "workspace_engineering", desc="Engineering migration query")
q("email lifecycle automation", d30, "workspace_marketing", desc="Marketing email query")
q("social media community management", d26, "workspace_marketing", desc="Marketing social query")
q("PR media relations press", d31, "workspace_marketing", desc="Marketing PR query")
q("한국 마케팅 캠페인", d32, "workspace_marketing", desc="Korean marketing campaign", lang="ko")
q("日本マーケティング予算", d33, "workspace_marketing", desc="Japanese marketing budget", lang="ja")
q("แคมเปญการตลาดไทย", d34, "workspace_marketing", desc="Thai marketing campaign", lang="th")
q("kampanye marketing Indonesia", d35, "workspace_marketing", desc="Indonesian marketing campaign", lang="id")
q("campanha marketing Brasil", d29, "workspace_marketing", desc="Portuguese marketing campaign", lang="pt")
q("campaña marketing LATAM", d24, "workspace_marketing", desc="Spanish marketing campaign", lang="es")
q("campagne marketing France", d28, "workspace_marketing", desc="French marketing campaign", lang="fr")
q("chính sách bảo mật luật Việt Nam", d44, "workspace_legal", desc="Vietnamese legal compliance", lang="vi")
q("数据保护政策 中国", d45, "workspace_legal", desc="Chinese data protection", lang="zh")
q("個人情報保護方針 日本", d46, "workspace_legal", desc="Japanese privacy policy", lang="ja")
q("개인정보처리방침 한국", d47, "workspace_legal", desc="Korean privacy policy", lang="ko")
q("डिजिटल पर्सनल डेटा प्रोटेक्शन", d48, "workspace_legal", desc="Hindi DPDP act", lang="hi")
q("thông tin gia đình liên hệ khẩn cấp", d54, "workspace_personal", desc="Vietnamese family emergency", lang="vi")
q("趣味音楽", d55, "workspace_personal", desc="Japanese hobbies music", lang="ja")
q("投资理财组合", d56, "workspace_personal", desc="Chinese investment portfolio", lang="zh")
q("información familiar contacto emergencia", d59, "workspace_personal", desc="Spanish family emergency", lang="es")
q("여가 여행 계획", d60, "workspace_personal", desc="Korean travel plans", lang="ko")

# --- Multi-hop cross-scope ---
q("GDPR compliance and data processing agreement", [d36, d40], "workspace_legal", desc="Multi-hop: GDPR + DPA")
q("hiring process and compensation bands", [d71, d75], "workspace_hr", desc="Multi-hop: hiring + compensation")
q("marketing budget and revenue forecast", [d21, d62], ["workspace_marketing","workspace_finance"], desc="Multi-hop cross-scope: marketing + finance")
q("deployment strategy and incident response", [d18, d12], "workspace_engineering", desc="Multi-hop: deployment + incident")
q("accessibility and internationalization", [d20, d16], "workspace_engineering", desc="Multi-hop: a11y + i18n")
q("caching strategy and performance monitoring", [d15, d10], "workspace_engineering", desc="Multi-hop: caching + observability")

# Write dataset
dataset = {
    "name": "kchat-context-eval-dataset",
    "version": "2.0.0",
    "description": "Comprehensive context retrieval evaluation dataset with 80 documents across 6 scopes and 14 languages, 60+ queries including multi-hop, cross-language, semantic, and ACL tests.",
    "documents": docs,
    "queries": queries,
}

with open("context_dataset_v2.json", "w", encoding="utf-8") as f:
    json.dump(dataset, f, ensure_ascii=False, indent=2)

print(f"Documents: {len(docs)}")
print(f"Queries: {len(queries)}")
print(f"File size: {len(json.dumps(dataset))} bytes")

# Stats
from collections import Counter
scope_counts = Counter(d["scope"] for d in docs)
lang_counts = Counter(d["language"] for d in docs)
print(f"\nDocuments by scope: {dict(scope_counts)}")
print(f"Documents by language: {dict(lang_counts)}")

q_types = {"simple": 0, "acl": 0, "multihop": 0, "crosslang": 0, "semantic": 0}
for query in queries:
    desc = query["description"]
    if "ACL" in desc: q_types["acl"] += 1
    elif "Multi-hop" in desc: q_types["multihop"] += 1
    elif "Cross-lang" in desc: q_types["crosslang"] += 1
    elif "Semantic" in desc: q_types["semantic"] += 1
    else: q_types["simple"] += 1
print(f"\nQuery types: {q_types}")
