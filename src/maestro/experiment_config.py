"""
MAESTRO — Experiment configuration
Central registry of inputs, model pricing, and available strategies.
Single source of truth for the experiment matrix.

To add a new input:   append to INPUTS
To add a new model:   append to MODELS
To enable a strategy: add to STRATEGIES (once implemented)
"""

from pathlib import Path

from maestro.schemas import InputFile, ModelPricing, Strategy, Tier

# ---------------------------------------------------------------------------
# Base path for all data files (relative to project root)
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"


# ---------------------------------------------------------------------------
# Input registry — each entry is one benchmark case + ground truth
# ---------------------------------------------------------------------------

INPUTS: list[InputFile] = [
    # ── Tier 1 — BPMN (IDs 01–05, source: MIWG Category A / C) ─────────────
    InputFile(
        example_id="bpmn_1_01",
        tier=Tier.SIMPLE,
        entity_count=5,
        file_path=DATA_DIR / "01_bpmn_1.JSON",
        ground_truth_path=DATA_DIR / "01_bpmn_1_ground_truth.MMD",
        description="Simple sequential process: Start → Task 1 → Task 2 → Task 3 → End (MIWG A.1.0)",
    ),
    InputFile(
        example_id="bpmn_1_02",
        tier=Tier.SIMPLE,
        entity_count=8,
        file_path=DATA_DIR / "02_bpmn_1.JSON",
        ground_truth_path=DATA_DIR / "02_bpmn_1_ground_truth.MMD",
        description="Single process with exclusive gateway split and merge — 3 parallel paths (MIWG A.2.0)",
    ),
    InputFile(
        example_id="bpmn_1_03",
        tier=Tier.SIMPLE,
        entity_count=8,
        file_path=DATA_DIR / "03_bpmn_1.JSON",
        ground_truth_path=DATA_DIR / "03_bpmn_1_ground_truth.MMD",
        description="Single process with exclusive gateway, default flows, and convergence (MIWG A.2.1)",
    ),
    InputFile(
        example_id="bpmn_1_04",
        tier=Tier.SIMPLE,
        entity_count=10,
        file_path=DATA_DIR / "04_bpmn_1.JSON",
        ground_truth_path=DATA_DIR / "04_bpmn_1_ground_truth.MMD",
        description="Process with collapsed sub-process and two boundary events (MIWG A.3.0)",
    ),
    InputFile(
        example_id="bpmn_1_05",
        tier=Tier.SIMPLE,
        entity_count=9,
        file_path=DATA_DIR / "05_bpmn_1.JSON",
        ground_truth_path=DATA_DIR / "05_bpmn_1_ground_truth.MMD",
        description="Process with intermediate events and branching flows (MIWG C.8.0)",
    ),
    # ── Tier 2 — BPMN (IDs 11–15, source: MIWG A.4.0 / C) ──────────────────
    InputFile(
        example_id="bpmn_2_11",
        tier=Tier.COMPLEX,
        entity_count=17,
        file_path=DATA_DIR / "11_bpmn_2.JSON",
        ground_truth_path=DATA_DIR / "11_bpmn_2_ground_truth.MMD",
        description="Two-pool BPMN collaboration with message flows, lanes, and expanded sub-processes (MIWG A.4.0)",
    ),
    InputFile(
        example_id="bpmn_2_12",
        tier=Tier.COMPLEX,
        entity_count=16,
        file_path=DATA_DIR / "12_bpmn_2.JSON",
        ground_truth_path=DATA_DIR / "12_bpmn_2_ground_truth.MMD",
        description="Multi-pool collaboration with 4 lanes and message flows (MIWG C.1.0)",
    ),
    InputFile(
        example_id="bpmn_2_13",
        tier=Tier.COMPLEX,
        entity_count=18,
        file_path=DATA_DIR / "13_bpmn_2.JSON",
        ground_truth_path=DATA_DIR / "13_bpmn_2_ground_truth.MMD",
        description="Four-pool collaboration with complex cross-pool message flows (MIWG C.4.0)",
    ),
    InputFile(
        example_id="bpmn_2_14",
        tier=Tier.COMPLEX,
        entity_count=20,
        file_path=DATA_DIR / "14_bpmn_2.JSON",
        ground_truth_path=DATA_DIR / "14_bpmn_2_ground_truth.MMD",
        description="Single-pool process with 3 lanes, event-based gateways and timers (MIWG C.5.0)",
    ),
    InputFile(
        example_id="bpmn_2_15",
        tier=Tier.COMPLEX,
        entity_count=16,
        file_path=DATA_DIR / "15_bpmn_2.JSON",
        ground_truth_path=DATA_DIR / "15_bpmn_2_ground_truth.MMD",
        description="Single-pool process with parallel gateways and multiple end events (MIWG C.9.0)",
    ),
    # ── Tier 1 — IT Architecture (IDs 06–10) ────────────────────────────────
    InputFile(
        example_id="it_1_06",
        tier=Tier.SIMPLE,
        entity_count=5,
        file_path=DATA_DIR / "06_it_1.JSON",
        ground_truth_path=DATA_DIR / "06_it_1_ground_truth.MMD",
        description="SomeApp: web app on Infomaniak Public Cloud behind Cloudflare CDN with S3-compatible storage",
    ),
    InputFile(
        example_id="it_1_07",
        tier=Tier.SIMPLE,
        entity_count=7,
        file_path=DATA_DIR / "07_it_1.JSON",
        ground_truth_path=DATA_DIR / "07_it_1_ground_truth.MMD",
        description="SomeApp on Infomaniak: web app + PostgreSQL + object storage, with developer SSH access",
    ),
    InputFile(
        example_id="it_1_08",
        tier=Tier.SIMPLE,
        entity_count=7,
        file_path=DATA_DIR / "08_it_1.JSON",
        ground_truth_path=DATA_DIR / "08_it_1_ground_truth.MMD",
        description="Google Apps Script web app (OU-restricted, executes as deployer) with Code.gs + Index.html reading/writing data.json on Google Drive",
    ),
    InputFile(
        example_id="it_1_09",
        tier=Tier.SIMPLE,
        entity_count=8,
        file_path=DATA_DIR / "09_it_1.JSON",
        ground_truth_path=DATA_DIR / "09_it_1_ground_truth.MMD",
        description="GCP data analysis stack: IAP → Web App (MCP host) → Gemini + MCP Server → orders view on PostgreSQL",
    ),
    InputFile(
        example_id="it_1_10",
        tier=Tier.SIMPLE,
        entity_count=9,
        file_path=DATA_DIR / "10_it_1.JSON",
        ground_truth_path=DATA_DIR / "10_it_1_ground_truth.MMD",
        description="Small office network: router → firewall → switch → NAS, printer, VoIP phones (wired) + AP → laptops (WiFi)",
    ),
    # ── Tier 2 — IT Architecture (IDs 16–20) ────────────────────────────────
    InputFile(
        example_id="it_2_16",
        tier=Tier.COMPLEX,
        entity_count=11,
        file_path=DATA_DIR / "16_it_2.JSON",
        ground_truth_path=DATA_DIR / "16_it_2_ground_truth.MMD",
        description="SomeApp full delivery stack: GitLab CI + Terraform + test suite + runtime on Infomaniak behind Cloudflare",
    ),
    InputFile(
        example_id="it_2_17",
        tier=Tier.COMPLEX,
        entity_count=14,
        file_path=DATA_DIR / "17_it_2.JSON",
        ground_truth_path=DATA_DIR / "17_it_2_ground_truth.MMD",
        description="Expanded office network: adds ISP (explicit WAN edge), IP cameras + NVR, badge/access control, POS terminal to it_1_10",
    ),
    InputFile(
        example_id="it_2_18",
        tier=Tier.COMPLEX,
        entity_count=13,
        file_path=DATA_DIR / "18_it_2.JSON",
        ground_truth_path=DATA_DIR / "18_it_2_ground_truth.MMD",
        description="Extended GCP stack: adds Cloud Tasks + Background Worker, Cloud Storage, Secret Manager, Cloud Monitoring to it_1_09",
    ),
    InputFile(
        example_id="it_2_19",
        tier=Tier.COMPLEX,
        entity_count=19,
        file_path=DATA_DIR / "19_it_2.JSON",
        ground_truth_path=DATA_DIR / "19_it_2_ground_truth.MMD",
        description="Dual data center: active/standby load balancing with failover; each DC has DMZ (firewall + LB) and internal LAN (web app, auth/IAM, PostgreSQL); DB and IAM replicate across DCs via encrypted WAN",
    ),
    InputFile(
        example_id="it_2_20",
        tier=Tier.COMPLEX,
        entity_count=14,
        file_path=DATA_DIR / "20_it_2.JSON",
        ground_truth_path=DATA_DIR / "20_it_2_ground_truth.MMD",
        description="Hybrid cloud / on-premises: external users via cloud CDN/LB + VPN to on-prem app server (PostgreSQL, NFS, Active Directory); cloud layer provides VPN gateway, object storage, and monitoring",
    ),
    # ── Tier 3 — BPMN (IDs 21–25, source: MIWG B / C) ──────────────────────
    InputFile(
        example_id="bpmn_3_21",
        tier=Tier.CROSS_LAYER,
        entity_count=29,
        file_path=DATA_DIR / "21_bpmn_3.JSON",
        ground_truth_path=DATA_DIR / "21_bpmn_3_ground_truth.MMD",
        description="Two-pool collaboration with lanes, mixed task types, collapsed/expanded sub-processes, 3 call activities, message start/end events, timer start, terminate end (MIWG B.1.0)",
    ),
    InputFile(
        example_id="bpmn_3_22",
        tier=Tier.CROSS_LAYER,
        entity_count=29,
        file_path=DATA_DIR / "22_bpmn_3.JSON",
        ground_truth_path=DATA_DIR / "22_bpmn_3_ground_truth.MMD",
        description="Four-pool e-commerce collaboration: Customer / Amazon (Picker + Packager lanes) / Carrier / Credit Card Company; 5 message flows, error boundary on Checkout sub-process (MIWG C.2.0)",
    ),
    InputFile(
        example_id="bpmn_3_23",
        tier=Tier.CROSS_LAYER,
        entity_count=40,
        file_path=DATA_DIR / "23_bpmn_3.JSON",
        ground_truth_path=DATA_DIR / "23_bpmn_3_ground_truth.MMD",
        description="Travel Booking process: event-based gateway, parallel gateways, compensation patterns (boundary + throw events), Make Booking sub-process, Handle Compensation sub-process, 6 send + 6 service tasks (MIWG C.6.0)",
    ),
    InputFile(
        example_id="bpmn_3_24",
        tier=Tier.CROSS_LAYER,
        entity_count=23,
        file_path=DATA_DIR / "24_bpmn_3.JSON",
        ground_truth_path=DATA_DIR / "24_bpmn_3_ground_truth.MMD",
        description="Manual Check process (expanded C.9.2): parallel fraud + risk check split, escalation gateway to Senior Reviewer, intermediate message catch for additional documents",
    ),
    InputFile(
        example_id="bpmn_3_25",
        tier=Tier.CROSS_LAYER,
        entity_count=24,
        file_path=DATA_DIR / "25_bpmn_3.JSON",
        ground_truth_path=DATA_DIR / "25_bpmn_3_ground_truth.MMD",
        description="Vacation Request process (expanded C.8.1): balance check + gateway before business-rule engine, HR Committee Review branch with intermediate message catch, new Insufficient Balance terminal",
    ),
    # ── Tier 3 — IT Architecture (IDs 26–30) ────────────────────────────────
    InputFile(
        example_id="it_3_26",
        tier=Tier.CROSS_LAYER,
        entity_count=34,
        file_path=DATA_DIR / "26_it_3.JSON",
        ground_truth_path=DATA_DIR / "26_it_3_ground_truth.MMD",
        description="Dual-DC active/standby LB with full IAM stack (IAM server + LDAP + token cache), external geofencing (zone + IP + telecom MFA), and Logging-as-a-Service for DB audit trail; extends it_2_19",
    ),
    InputFile(
        example_id="it_3_27",
        tier=Tier.CROSS_LAYER,
        entity_count=32,
        file_path=DATA_DIR / "27_it_3.JSON",
        ground_truth_path=DATA_DIR / "27_it_3_ground_truth.MMD",
        description="Two-office network (HQ + branch) with AWS hub: each office has router/fw/switch/VPN GW/NAS/NVR/cameras/POS/AP/clients; HQ adds access control + VoIP; AWS provides S3 NAS backup + S3 video archive via site-to-site VPN; remote users access via client VPN",
    ),
    InputFile(
        example_id="it_3_28",
        tier=Tier.CROSS_LAYER,
        entity_count=25,
        file_path=DATA_DIR / "28_it_3.JSON",
        ground_truth_path=DATA_DIR / "28_it_3_ground_truth.MMD",
        description="Extended GCP stack: Claude API replaces Vertex AI; external users enter via CDN+WAF+API Gateway with OAuth2; separate internal DB for external client data; vector store for RAG; event pipeline via Cloud Tasks+Pub/Sub+notification service; Cloud Scheduler for periodic jobs",
    ),
    InputFile(
        example_id="it_3_29",
        tier=Tier.CROSS_LAYER,
        entity_count=25,
        file_path=DATA_DIR / "29_it_3.JSON",
        ground_truth_path=DATA_DIR / "29_it_3_ground_truth.MMD",
        description="Full SomeApp stack: CI/CD env (GitLab runner + Terraform + tests + Vault + artifact registry), Staging env, Production env (web app + Redis + PostgreSQL + object storage + backup + Prometheus + log server); Cloudflare CDN/WAF; PostHog analytics",
    ),
    InputFile(
        example_id="it_3_30",
        tier=Tier.CROSS_LAYER,
        entity_count=25,
        file_path=DATA_DIR / "30_it_3.JSON",
        ground_truth_path=DATA_DIR / "30_it_3_ground_truth.MMD",
        description="IoT edge + stream processing platform: sensors → edge gateway + MQTT broker → Kafka → data validator + stream processor → time-series DB + Redis cache; rule engine + ML anomaly detector → alert manager → notification gateway; Grafana dashboard; audit log; cloud archival",
    ),
]


# ---------------------------------------------------------------------------
# Model registry — pricing per model for cost calculation
# ---------------------------------------------------------------------------

# Synthetic "model" used only for control-strategy rows. Controls bypass the
# LLM entirely so no real model is involved; this entry exists so the
# ``RunConfig.model`` column has an honest value ("control") rather than
# borrowing the name of a real model and lying about what produced the row.
# Zero pricing means control rows never affect cost rollups.
CONTROL_MODEL = ModelPricing(
    model="control",
    input_price_per_1m=0.0,
    output_price_per_1m=0.0,
)

MODELS: list[ModelPricing] = [
    ModelPricing(
        model="claude-haiku-4-5-20251001",
        input_price_per_1m=0.80,
        output_price_per_1m=4.00,
    ),
    ModelPricing(
        # Pinned to snapshot for reproducibility
        model="gpt-4o-mini-2024-07-18",
        input_price_per_1m=0.15,
        output_price_per_1m=0.60,
    ),
    ModelPricing(
        model="mistral-small-2603",
        input_price_per_1m=0.15,
        output_price_per_1m=0.60,
    ),
    ModelPricing(
        model="gemini-2.5-flash-lite",
        input_price_per_1m=0.10,
        output_price_per_1m=0.40,
    ),
    # DeepSeek — the cross-provider replication dimension's emerging-Chinese
    # entry (proposal §3.2). Consumed via the OpenAI-compatible endpoint
    # (see providers/deepseek.py). Efficiency-tier model only for now; the
    # edge model (deepseek-v4-pro) can be added later.
    #
    # Model id: "deepseek-v4-flash" is the CURRENT model as confirmed against
    # the DeepSeek API docs (https://api-docs.deepseek.com/) on 2026-06-01.
    # The older "deepseek-chat"/"deepseek-reasoner" aliases are the *legacy*
    # names slated for deprecation on 2026-07-24 — deliberately avoided here so
    # the pinned id outlives the frozen main run.
    #
    # Pricing is the *cache-miss* (standard) rate as of 2026-06-01
    # (https://api-docs.deepseek.com/quick_start/pricing): DeepSeek also offers
    # a ~10x-cheaper cache-hit input price, but ModelPricing has a single input
    # rate, so we use cache-miss — every call is costed as a miss, making the
    # tracked cost an upper bound on actual spend (never an under-count).
    # Verify both id and price against the live console before the frozen run.
    ModelPricing(
        model="deepseek-v4-flash",
        input_price_per_1m=0.14,
        output_price_per_1m=0.28,
    ),
    # --- Add new models below ---
    # ModelPricing(
    #     model="claude-sonnet-4-20250514",
    #     input_price_per_1m=3.00,
    #     output_price_per_1m=15.00,
    # ),
]


# ---------------------------------------------------------------------------
# Strategy registry — only strategies with working implementations
# ---------------------------------------------------------------------------

STRATEGIES: list[Strategy] = [
    Strategy.SINGLE_AGENT,
    Strategy.SOP_BASED,
    Strategy.CREW_AI,
    Strategy.LANG_GRAPH,
    # Control conditions (no LLM, deterministic). Included in the default
    # matrix so every full run produces a metric-pipeline sanity record;
    # ``build_matrix`` in run.py collapses their model/repeat fan-out to
    # one row per (input, control) cell since neither dimension varies.
    Strategy.NULL_CONTROL,
    Strategy.COPY_CONTROL,
    Strategy.GROUND_TRUTH_CONTROL,
]


# Set used by ``build_matrix`` and analysis code to special-case controls:
# - matrix builder uses CONTROL_MODEL and run_number=1 for these strategies
# - analysis can exclude them from ANOVA / cost rollups with
#   ``WHERE strategy NOT IN (SELECT value FROM control_strategies)`` or the
#   in-Python equivalent ``s not in CONTROL_STRATEGIES``.
CONTROL_STRATEGIES: set[Strategy] = {
    Strategy.NULL_CONTROL,
    Strategy.COPY_CONTROL,
    Strategy.GROUND_TRUTH_CONTROL,
}


# ---------------------------------------------------------------------------
# Experiment defaults
# ---------------------------------------------------------------------------

# Number of repeated runs per (input, strategy, model) cell
DEFAULT_REPEATS = 5

# SQLite database path (project root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DB_PATH = PROJECT_ROOT / "maestro.db"
