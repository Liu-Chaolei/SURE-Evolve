# Main-Flow Strict-Core Protocol Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent the main-flow agent and its TTS/VC execution surfaces from generating or accepting any inference protocol other than `strict_core`.

**Architecture:** Establish one main-flow protocol-selection contract, inject it as a non-bypassable agent constraint, materialize `strict_core` into every structured handoff, and reject all other runtime values in each self-contained shell template. A focused regression test audits documentation, examples, JSON/YAML templates, and actual shell rejection behavior.

**Tech Stack:** Markdown contracts, JSON/YAML templates, Bash execution templates, pytest.

---

### Task 1: Add the failing protocol-policy regression test

**Files:**
- Create: `tests/test_main_flow_protocol_policy.py`

- [ ] **Step 1: Write the static contract and template tests**

Create tests that assert:

```python
MAIN_FLOW_ROOT = REPO_ROOT / "docs" / "agents" / "main_flow_agent"
RUN_TEMPLATES = (
    "run_single_model.sh",
    "run_single_model_single_dataset.sh",
    "run_audio_evaluation_only.sh",
)

def test_main_flow_declares_strict_core_system_constraint():
    text = (MAIN_FLOW_ROOT / "AGENTS.md").read_text(encoding="utf-8")
    assert "[SYSTEM_CONSTRAINT: MAIN_FLOW_PROTOCOL_SELECTION]" in text
    assert "strict_core" in text
    assert "standard_system" in text

def test_every_input_example_materializes_strict_core():
    for path in sorted((MAIN_FLOW_ROOT / "examples").glob("input_*.md")):
        assert re.search(r"(?m)^\s+protocol_id:\s+strict_core\s*$", path.read_text(encoding="utf-8")), path

def test_structured_templates_never_leave_protocol_id_open():
    for path in sorted((MAIN_FLOW_ROOT / "templates").glob("*.json")):
        values = collect_key_values(json.loads(path.read_text(encoding="utf-8")), "protocol_id")
        assert all(value == "strict_core" for value in values), (path, values)
```

- [ ] **Step 2: Write runtime rejection tests**

For each shell template, run it with an existing temporary `MODEL_DIR`, a fake
AiSpeech source root, and `PROTOCOL_ID` set first to `custom_tts_protocol` and
then to `standard_system`. Assert exit code `2` and an error containing
`main-flow protocol must be strict_core`. This proves rejection happens before
dataset preparation, inference, or evaluation.

- [ ] **Step 3: Run the focused test and verify RED**

Run:

```bash
pytest -q tests/test_main_flow_protocol_policy.py
```

Expected: failures for the missing system constraint, missing protocol contract,
missing example fields, and missing shell rejection guards.

- [ ] **Step 4: Commit the failing test**

```bash
git add tests/test_main_flow_protocol_policy.py
git commit -m "test: enforce strict core main flow protocol"
```

### Task 2: Define the authoritative main-flow protocol contract

**Files:**
- Create: `docs/agents/main_flow_agent/contracts/inference_protocol_selection.md`
- Modify: `docs/agents/main_flow_agent/AGENTS.md`
- Modify: `docs/agents/main_flow_agent/contracts/main_flow_architecture.md`

- [ ] **Step 1: Add the protocol-selection contract**

Define the global enumeration (`strict_core`, `standard_system`), pin every
main-flow inference run to `strict_core`, prohibit derived/custom identifiers,
require TTS/VC evaluation-only inheritance, define rejection rather than silent
normalization, and list required evidence fields.

- [ ] **Step 2: Add the non-bypassable system constraint**

Add this constraint to the system-prompt block in `AGENTS.md`:

```text
[SYSTEM_CONSTRAINT: MAIN_FLOW_PROTOCOL_SELECTION]
1. GLOBAL_ENUM: the only defined IDs are strict_core and standard_system.
2. MAIN_FLOW_SELECTION: every main-flow inference run MUST select strict_core.
3. NO_DERIVATION: protocol IDs MUST NOT be derived from task, model, metric,
   pipeline, segment, run, or output names.
4. NO_OVERRIDE: standard_system and custom values are invalid on the main-flow
   execution surface and MUST stop before work begins.
5. INHERITANCE: TTS/VC evaluation-only and segmented retries MUST inherit
   strict_core and MUST NOT create a new protocol ID.
```

- [ ] **Step 3: Connect the architecture contract**

Record protocol selection as a main-agent decision with a single legal
main-flow result (`strict_core`) and deterministic-script enforcement.

- [ ] **Step 4: Run the focused test**

Expected: system-constraint assertions pass; remaining template/example/runtime
assertions still fail.

- [ ] **Step 5: Commit the contract layer**

```bash
git add docs/agents/main_flow_agent/AGENTS.md \
  docs/agents/main_flow_agent/contracts/inference_protocol_selection.md \
  docs/agents/main_flow_agent/contracts/main_flow_architecture.md
git commit -m "docs: constrain main flow protocol selection"
```

### Task 3: Materialize strict_core in structured templates and examples

**Files:**
- Modify: `docs/agents/main_flow_agent/templates/main_agent_plan.json`
- Modify: `docs/agents/main_flow_agent/templates/main_agent_execution_surface.json`
- Modify: `docs/agents/main_flow_agent/templates/main_agent_script_routing.json`
- Modify: `docs/agents/main_flow_agent/templates/main_agent_run_report.json`
- Modify: `docs/agents/main_flow_agent/templates/model_eval_manifest.json`
- Modify: `docs/agents/main_flow_agent/templates/protocol.yaml`
- Modify: `docs/agents/main_flow_agent/examples/input_template.md`
- Modify: all `docs/agents/main_flow_agent/examples/input_*.md`
- Modify: `docs/agents/main_flow_agent/examples/README.md`

- [ ] **Step 1: Add protocol evidence to structured JSON templates**

Set top-level `protocol_id` to `strict_core` in plan, routing, run-report, and
manifest templates. Add this object under execution-surface `resolved_inputs`:

```json
"protocol": {
  "allowed_protocol_ids": ["strict_core", "standard_system"],
  "selected_protocol_id": "strict_core",
  "selection_policy": "main_flow_strict_minimum_inference",
  "allow_override": false
}
```

- [ ] **Step 2: Pin the protocol manifest template**

Replace the three open `{protocol_id}` placeholders in `templates/protocol.yaml`
with `strict_core`. Keep protocol definition/model parameter placeholders intact.

- [ ] **Step 3: Update every input example**

Add exactly this field under each `harness:` mapping:

```yaml
protocol_id: strict_core  # Fixed by main-flow policy; not a custom namespace.
```

Explain in `examples/README.md` that examples may not use `standard_system` or
invent protocol IDs.

- [ ] **Step 4: Run JSON parsing and the focused test**

Run:

```bash
python -m json.tool docs/agents/main_flow_agent/templates/main_agent_execution_surface.json >/dev/null
pytest -q tests/test_main_flow_protocol_policy.py
```

Expected: structured/example assertions pass; shell runtime assertions still fail.

- [ ] **Step 5: Commit structured materialization**

```bash
git add docs/agents/main_flow_agent/templates docs/agents/main_flow_agent/examples
git commit -m "docs: materialize strict core in main flow inputs"
```

### Task 4: Reject non-strict protocols in every execution template

**Files:**
- Modify: `docs/agents/main_flow_agent/templates/run_single_model.sh`
- Modify: `docs/agents/main_flow_agent/templates/run_single_model_single_dataset.sh`
- Modify: `docs/agents/main_flow_agent/templates/run_audio_evaluation_only.sh`

- [ ] **Step 1: Add the same early guard to each self-contained template**

Before resolving results paths or performing work, use:

```bash
REQUESTED_PROTOCOL_ID="${PROTOCOL_ID:-strict_core}"
if [[ "$REQUESTED_PROTOCOL_ID" != "strict_core" ]]; then
  echo "ERROR: main-flow protocol must be strict_core; got: $REQUESTED_PROTOCOL_ID" >&2
  echo "ERROR: standard_system and custom protocol IDs are outside the main-flow strict-minimum inference policy." >&2
  exit 2
fi
readonly PROTOCOL_ID="strict_core"
```

- [ ] **Step 2: Verify runtime RED-to-GREEN behavior**

Run:

```bash
pytest -q tests/test_main_flow_protocol_policy.py
bash -n docs/agents/main_flow_agent/templates/run_single_model.sh
bash -n docs/agents/main_flow_agent/templates/run_single_model_single_dataset.sh
bash -n docs/agents/main_flow_agent/templates/run_audio_evaluation_only.sh
```

Expected: all focused tests pass and all shell syntax checks exit `0`.

- [ ] **Step 3: Commit runtime enforcement**

```bash
git add docs/agents/main_flow_agent/templates/run_*.sh
git commit -m "fix: reject custom main flow protocols"
```

### Task 5: Align all explanatory contracts and compliance guidance

**Files:**
- Modify: `docs/agents/main_flow_agent/README.md`
- Modify: `docs/agents/main_flow_agent/contracts/eval_run_layout.md`
- Modify: `docs/agents/main_flow_agent/contracts/main_agent_execution_surface_unit.md`
- Modify: `docs/agents/main_flow_agent/contracts/main_agent_script_routing_unit.md`
- Modify: `docs/agents/main_flow_agent/contracts/single_model_single_dataset_shell.md`
- Modify: `docs/agents/main_flow_agent/contracts/tts_vc_audio_evaluation_surface.md`
- Modify: `docs/agents/main_flow_agent/templates/main_agent_script_routing_compliance_check_prompt.md`
- Modify: `docs/agents/main_flow_agent/templates/script_routing_compliance_check.json`

- [ ] **Step 1: Replace open protocol path notation for main-flow outputs**

Where the documents describe main-flow publication paths, state the concrete
suffix `<model_name>/strict_core/`. When explaining the generic global protocol
catalog, retain `<protocol_id>` only with an explicit two-value enumeration.

- [ ] **Step 2: Add unit-level requirements and must-not rules**

Require execution-surface, script-routing, shell, and TTS/VC units to preserve
`strict_core`; prohibit task/metric/segment-derived IDs and evaluation-only
renaming.

- [ ] **Step 3: Extend compliance evidence**

Add protocol checks to the compliance prompt and JSON result template:

```json
"protocol_policy": {
  "passed": false,
  "selected_protocol_id": "",
  "allowed_protocol_ids": ["strict_core", "standard_system"],
  "main_flow_required_protocol_id": "strict_core",
  "custom_protocol_detected": false
}
```

- [ ] **Step 4: Run focused and related tests**

Run:

```bash
pytest -q tests/test_main_flow_protocol_policy.py tests/test_evaluation_scripts_contracts.py
```

Expected: all tests pass.

- [ ] **Step 5: Commit explanatory alignment**

```bash
git add docs/agents/main_flow_agent
git commit -m "docs: align main flow protocol policy"
```

### Task 6: Complete the protocol-policy audit

**Files:**
- Verify: `docs/agents/main_flow_agent/**`
- Verify: `tests/test_main_flow_protocol_policy.py`

- [ ] **Step 1: Run the full focused verification**

```bash
pytest -q tests/test_main_flow_protocol_policy.py
for path in docs/agents/main_flow_agent/templates/run_*.sh; do bash -n "$path"; done
```

- [ ] **Step 2: Audit all protocol assignments**

```bash
rg -n -i "protocol_id|PROTOCOL_ID|strict_core|standard_system" docs/agents/main_flow_agent
```

Review every match and confirm there is no third literal protocol ID, no open
main-flow override, and no TTS/VC segment-specific protocol derivation.

- [ ] **Step 3: Verify the actual rejection boundary**

Run each shell template with `PROTOCOL_ID=standard_system` and
`PROTOCOL_ID=custom_tts_protocol`; confirm each exits `2` before work begins.

- [ ] **Step 4: Inspect the final diff**

```bash
git diff --check
git status --short
```

Confirm unrelated pre-existing worktree changes remain untouched.
