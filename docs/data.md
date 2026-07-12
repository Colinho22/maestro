# Data corpus

Reference for the benchmark inputs in `data/`. Written for anyone reading
the existing corpus, adding a new input, or authoring a ground-truth
diagram. Everything the runner does with the corpus is driven by the
`INPUTS` registry in `src/maestro/experiment_config.py`; this document
describes the file conventions that registry expects.

---

## 1. What lives in `data/`

Two files per input, matched by prefix:

- `NN_<slug>.JSON` : structured input the strategies read.
- `NN_<slug>_ground_truth.MMD` : the Mermaid diagram the output is
  scored against.

`NN` is a two-digit ordinal (01..30 in the current corpus), sortable and
stable across releases. The `<slug>` encodes diagram type and tier
(`bpmn_1`, `it_2`, `bpmn_3`, `it_3`), matching the `example_id` in the
registry.

The current corpus has 30 inputs across two diagram families (BPMN
processes, IT architecture) and three complexity tiers:

| Tier | Value | Rough size | Inputs |
|---|---|---|---|
| `SIMPLE` | 1 | fewer than 10 entities | 10 (bpmn_1_01..05, it_1_06..10) |
| `COMPLEX` | 2 | 10 to 25 entities | 10 (bpmn_2_11..15, it_2_16..20) |
| `CROSS_LAYER` | 3 | 25+ entities, multi-pool, or cross-layer flows | 10 (bpmn_3_21..25, it_3_26..30) |

---

## 2. Input JSON format

Every input is a JSON document with a `metadata` block and an
`elements`/`nodes` list. Example (`data/01_bpmn_1.JSON`):

```json
{
  "metadata": {
    "id": "bpmn_1_01",
    "source": "A.1.0.bpmn",
    "diagram_type": "bpmn_process",
    "tier": 1,
    "entity_count": 5,
    "container_count": 0,
    "attachment_count": 0,
    "description": "Simple sequential process: Start -> Task 1 -> ... -> End"
  },
  "nodes": [
    { "id": "task_1", "name": "Task 1", "type": "task", "lane": null, "attached_to": null },
    ...
  ],
  "edges": [
    { "source": "start_event", "target": "task_1", "type": "sequence_flow" },
    ...
  ]
}
```

### 2.1 `metadata`

| Field | Purpose |
|---|---|
| `id` | Must equal the registry's `example_id`. Also mirrors the filename slug. |
| `source` | Origin of the case (a MIWG file id, an internal author id, or similar). Documentation only. |
| `diagram_type` | One of `bpmn_process`, `bpmn_collaboration`, `c4_container`, `network_topology`. Drives strategy behaviour (label conventions, container semantics). |
| `tier` | Integer matching the registry's `Tier`. Redundant with the registry but useful when a file is inspected in isolation. |
| `entity_count`, `container_count`, `attachment_count` | Author-recorded totals. The metric layer counts independently; these are for at-a-glance sanity checks. |
| `description` | Short prose. Documentation only. |

### 2.2 `nodes` / `elements`

Every node has an `id` (matches the ground truth's Mermaid node id) and a
`name` (the human-readable label). `type` is a small controlled
vocabulary per diagram family (`task`, `gateway`, `event`, `container`,
`device`, `system`, ...). Optional fields:

- `lane`: the pool or lane the element belongs to. Consumed by
  collaboration-diagram strategies.
- `attached_to`: for BPMN boundary events, the id of the host activity.
  Scored as an attachment (`o--o` edge).
- `technology`: free-text for C4 / network diagrams (e.g. `PostgreSQL`).

An empty `name` (`""` or absent) is meaningful: the input is stating that
this element has no author-provided label. See section 3.

### 2.3 `edges`

Every edge has `source` and `target` (matching node ids) and `type`
(e.g. `sequence_flow`, `message_flow`, `association`). Some edges carry a
`label`; a missing label means the ground-truth diagram draws the edge
unlabelled.

---

## 3. Empty labels

An input node with an empty `name` (BPMN gateway, unnamed event, an
unlabelled network element) becomes a ground-truth Mermaid node with the
literal label `"a"`:

```mermaid
gw_result{"a"}
```

Not `""` and not an empty bracket like `gw_result{}`. Mermaid's parser
rejects the empty forms with a syntax error, so `"a"` is used as the
placeholder throughout. The metric layer knows about this convention:
`extract_input_unnamed_ids` reads the input JSON, finds the ids with an
empty `name`, and scores those nodes by id only (not by label), so the
`"a"` placeholder does not distort the entity-name F1.

Never edit an existing `.MMD` fixture to change this convention. The
placeholder is part of the scoring contract.

---

## 4. Ground-truth Mermaid

Each ground truth is a valid Mermaid `flowchart LR` diagram. Every input
convention has a parallel Mermaid convention:

- **Diagram header**: always `flowchart LR`. C4, sequence, class, and
  other Mermaid dialects are out of contract.
- **Node ids**: verbatim from the input's `id`. This is how the entity-id
  metric matches predictions to truth.
- **Node labels**: quoted (`node_id["Label"]`) so labels with spaces,
  parentheses, slashes, or line breaks stay parseable.
- **Multi-line labels**: for architecture and infrastructure diagrams,
  the label is `name\n[Type]\ntechnology` with `\n` as a literal
  two-character separator inside the quoted label. For process diagrams
  (BPMN process, BPMN collaboration), the label is the entity name only.
- **Empty labels**: `"a"` (see section 3).
- **Subgraphs**: any pool, lane, boundary, or deployment environment
  becomes a `subgraph id["Label"]` block. The subgraph label is quoted
  and non-empty.
- **Edges**: quoted labels between pipes, e.g. `a -->|"Approved"| b`.
  Unlabelled edges use a plain arrow `a --> b`. Never an empty label
  like `-->||` or `-->| |`.
- **Attachments** (BPMN boundary events, network `associated with` links):
  `host o--o event`. Direction ignored.

The Mermaid output contract that every strategy must satisfy is enumerated
in `src/maestro/prompts.py`. Ground truths follow the same contract, so a
correct model can achieve a perfect score by adhering to it.

---

## 5. Adding a new input

Two files, one registry entry.

### 5.1 Author the files

1. Pick the next free ordinal `NN` (looking at `data/`, or 31+ for the
   current corpus).
2. Write `NN_<slug>.JSON` following section 2.
3. Write `NN_<slug>_ground_truth.MMD` following section 4.

### 5.2 Register the input

Append an entry to `INPUTS` in `src/maestro/experiment_config.py`:

```python
InputFile(
    example_id="bpmn_1_31",
    tier=Tier.SIMPLE,
    entity_count=6,
    file_path=DATA_DIR / "31_bpmn_1.JSON",
    ground_truth_path=DATA_DIR / "31_bpmn_1_ground_truth.MMD",
    description="Short prose description for the registry.",
)
```

### 5.3 Smoke test

Verify the input actually runs end to end before committing to a full
matrix:

```bash
python -m maestro.run \
  --example bpmn_1_31 \
  --strategy single_agent \
  --model claude-haiku-4-5-20251001 \
  --repeats 1
```

Expected: one row in `run_results` with `error IS NULL`, one row in
`metric_results`, `entities_in_output` and `entities_in_truth` roughly
matching the author-recorded `entity_count`.

If the diagram fails to parse, `mmdc` prints the error; fix the ground
truth and re-run.

### 5.4 Watch the tier

The `Tier` enum has a fixed set of values (`SIMPLE`, `COMPLEX`,
`CROSS_LAYER`) with rough entity-count buckets (documented on the enum in
`schemas.py`). A new input should belong to an existing bucket. Adding a
tier is a schema change: it means new stratification levels in analysis
and must be a version bump, not a mid-run edit.

---

## 6. Where the corpus came from

- **BPMN**: the tier-1 and tier-3 BPMN cases are derived from the OMG
  BPMN 2.0 MIWG conformance corpus, with the source file recorded in
  `metadata.source` (e.g. `A.1.0.bpmn`). Tier-2 BPMN cases are internal.
- **IT architecture**: tier-1 IT cases are compact reference topologies;
  tier-2 and tier-3 IT cases extend them with realistic operational
  elements (CI/CD, monitoring, multi-DC, cloud). Sources are documented
  in the input description.

The published `v1.0.1` dataset was produced against the corpus exactly as
it exists in the tagged release. Adding or editing inputs after a scored
run means the analysis for that run is no longer comparable and requires
a fresh matrix.