"""
MAESTRO — Experiment configuration
Central registry of inputs, model pricing, and available strategies.
Single source of truth for the experiment matrix.

To add a new input:   append to INPUTS
To add a new model:   append to MODELS
To enable a strategy: add to STRATEGIES (once implemented)
"""

import os
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

# Two models per provider: a frontier ("best") model and an efficiency model,
# so the experiment can compare quality against cost within and across vendors.
# Prices are USD per 1M tokens, verified against each provider's pricing page in
# April 2026 for the frozen main run. IDs are pinned to dated snapshots where
# the provider offers one, so the run stays reproducible.
#
# Note: provider dispatch (run.py) is by substring (claude / gpt / mistral /
# gemini / deepseek), so any new model id must contain its provider's needle.
# tests/providers/test_provider_dispatch.py enforces this for every entry here.
MODELS: list[ModelPricing] = [
    # Anthropic
    ModelPricing(
        model="claude-opus-4-8",  # frontier
        input_price_per_1m=5.00,
        output_price_per_1m=25.00,
        # Opus 4.7+ removed sampling params; sending temperature returns 400.
        supports_temperature=False,
    ),
    ModelPricing(
        model="claude-haiku-4-5-20251001",  # efficiency
        input_price_per_1m=1.00,
        output_price_per_1m=5.00,
    ),
    # OpenAI (GPT-5 family: max_completion_tokens, no custom temperature)
    ModelPricing(
        model="gpt-5.5-2026-04-23",  # frontier
        input_price_per_1m=5.00,
        output_price_per_1m=30.00,
        supports_temperature=False,
    ),
    ModelPricing(
        model="gpt-5.4-mini-2026-03-17",  # efficiency
        input_price_per_1m=0.75,
        output_price_per_1m=4.50,
        supports_temperature=False,
    ),
    # Mistral
    ModelPricing(
        model="mistral-medium-3-5",  # frontier
        input_price_per_1m=1.50,
        output_price_per_1m=7.50,
    ),
    ModelPricing(
        model="mistral-small-2603",  # efficiency
        input_price_per_1m=0.15,
        output_price_per_1m=0.60,
    ),
    # Gemini
    ModelPricing(
        model="gemini-3.5-flash",  # frontier
        input_price_per_1m=1.50,
        output_price_per_1m=9.00,
    ),
    ModelPricing(
        model="gemini-3.1-flash-lite",  # efficiency
        input_price_per_1m=0.25,
        output_price_per_1m=1.50,
    ),
    # DeepSeek: the cross-provider replication dimension's emerging-Chinese
    # entry (proposal section 3.2), consumed via the OpenAI-compatible endpoint
    # (see providers/deepseek.py). Pricing is the cache-MISS (standard) rate;
    # DeepSeek also offers a cheaper cache-hit input price, but ModelPricing has
    # a single input rate, so cache-miss makes the tracked cost an upper bound
    # on actual spend (never an under-count).
    ModelPricing(
        model="deepseek-v4-pro",  # frontier
        input_price_per_1m=0.435,
        output_price_per_1m=0.87,
    ),
    ModelPricing(
        model="deepseek-v4-flash",  # efficiency
        input_price_per_1m=0.14,
        output_price_per_1m=0.28,
    ),
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

# SQLite database path. One canonical location, out/maestro.db, used by every
# consumer: the local runner, the Docker runner, and the visualizer all resolve
# here by default, so results land in one place regardless of how the run was
# launched. out/ is the host-accessible experiment-output directory (the
# project root holds the installed package). MAESTRO_DB_PATH overrides it; the
# Docker compose still sets it explicitly so the path matches the mount.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DB_PATH = PROJECT_ROOT / "out" / "maestro.db"
DB_PATH = Path(os.environ.get("MAESTRO_DB_PATH") or DEFAULT_DB_PATH)
