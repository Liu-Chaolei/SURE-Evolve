# Local speech corpus pipeline

The operator entrypoint is `scripts/corpus_cluster.py implement --run-dir RUN
--old-job OLD_JOB`. Run the operator from the login node with
`SPEECH_PIPELINE_KEY` inherited in its environment. It submits compute through
Slurm on n03; do not run corpus inference on the login node. Never write the
key into command examples, scripts, logs or manifests.

The corpus directory is RUN's parent and must contain `catalog.csv` and source
metadata. This mode explicitly accepts a bulk archive, not a fabricated
successful `paper_collect` manifest. Model weights are read-only from
`/shared/models/Qwen3.8-27B-W8A8/`. The pinned serving and parsing image digests
are in `corpus_cluster.py`.

## Rollout and resource transitions

1. Keep the existing parser job running on six cards while the temporary TP2
   Qwen service occupies the remaining two. The pilot client needs only CPUs.
2. Benchmark 24 real papers at concurrency 4, 8 and 16. Separate output
   directories and prefix labels prevent benchmark checkpoint/cache reuse.
   Select the fastest profile with at least 95% validated papers.
3. Extract 120 stratified basic papers, at least ten deep papers per domain
   (overlap deduplicated), build the graph, and write two reviewed survey
   sections for each domain. Any deep/survey failure blocks migration.
4. Submit permanent main (2 NPU, 64 CPU, 256 GiB), flow (CPU only, 16 CPU,
   64 GiB), and parser (6 NPU, 96 CPU, 320 GiB) jobs. Persist `jobs.json` and
   cancel only the specified old job. The model and coordinator have separate
   Slurm allocations and scratch directories. The parser job has four lanes
   per card, with up to six PDFs per leased batch.
5. After downloads and parsing become terminal, parser workers release their
   cards. Fill any missing TP2 replica slots up to four model instances. Once basic and core extraction
   finish, cancel these extra replicas and keep TP2 for survey generation.
6. Produce three English surveys of 8,000–12,000 body words, with two revision
   rounds per section. All model calls use the authenticated loopback service.

The model defaults to 32,768 context, 16 sequences, MTP3 and 8,192 batched
tokens. A service startup fallback uses 16,384 context and 8 sequences without
MTP. A successful `service-profile.json` freezes the profile across replicas.
Compiler/operator caches, queues and writable SQLite builds stay node-local;
validated results and published database snapshots live on shared storage.

`resource-plan.json` can select 2, 4 or 6 parsing cards. A submitted parser job
freezes that count in its batch command so its worker count matches its Slurm
allocation even if a later plan changes. Each parsing card receives 16 CPU,
48 GiB for its step and four parser lanes; the job has an additional 32 GiB
reserve. Reallocation drains current batches before replacing the parser job.
New jobs can depend on the old allocation's completion with `afterany`.

The current speech run uses four parsing cards and two TP2 extraction
instances (four cards). After parsing finishes, two further TP2 instances
can be added. Replica identity is tracked in `jobs.json`, so existing
instances are neither duplicated nor prevented from expanding further.

## Evidence and recovery

PDF identity is SHA-256. A valid parse requires matching PDF, Markdown and
archive checksums plus the content-list and middle-layout JSON records.
Existing validated parse bundles are imported. A shared parse record is
published before the worker acknowledges its lease, so an interrupted ACK
does not discard completed work. Local queue leases expire after ten minutes
and active workers renew them every thirty seconds.

Every parsed paper receives basic extraction. ASR means speech recognition,
TTS means speech synthesis and SD means speaker diarization. General speech
papers can be RELATED. Each domain selects 500 core papers using topic/year
strata; cross-domain papers receive deep processing only once. Deep extraction
adds graph facts and calibration to the basic pass. Survey requests may
promote at most 100 additional papers per domain.

Quotes must resolve to literal source spans (whitespace differences allowed).
Extraction requests present numbered, unmodified source excerpts. The model
returns excerpt IDs; the validator resolves them to original text and checks
entity names and numeric values against that text. Survey claim IDs similarly
resolve to validated paper facts, so neither stage needs to retype equations.
Reference-list mentions alone do not support method-use claims. Invalid
numbers, names and quotes are excluded; excessive invalid facts fail the
request. Failed retries preserve diagnostics alongside pass checkpoints.
Source/model/prompt signatures govern reuse; incompatible completed artifacts
are moved to `artifacts/stale` and never silently reused.

Restart with the same run directory and key. Queues reconstruct from shared
artifacts, so node-local SQLite is not the source of truth. A different key
requires restarting all pipeline services and workers consistently. Do not
reuse an old job ID without checking its owner and role. Slurm timeouts,
preemption and node failures are resubmitted by the operator; application/OOM
failures write `deployment-blocker.json` for diagnosis. Monitoring uses
`scontrol` and durable job exit records because Slurm accounting may be
disabled. The operator reloads the job ledger so a repaired allocation can be
adopted without losing the remaining jobs. Explicit recovery launchers in the
run's operations directory are preserved on resubmission.

## Inspecting a run

- `status.json`, `corpus.json`: phase, catalog and queue counts.
- `benchmark-progress.json`, `benchmark.json`, `pilot.json`: validation gates
  and measured throughput. An unfinished pilot is not a production rollout.
- `backend-*.json`, `service.json`: service metadata; the coordinator also
  checks live `/models` responses before using a backend.
- `artifacts/parses`, `artifacts/basic`, `artifacts/deep`: durable paper results.
- `artifacts/passes`, `artifacts/failures`, parser/model logs: diagnostics.
- `core-selection.json`, `supplements.json`: deep extraction accounting.
- `artifacts/graphs`: total/domain JSON, JSONL, SQLite/FTS and integrity report.
- `artifacts/surveys/{asr,tts,sd}`: Markdown, structured sections, citations,
  claim/source traces, report and manifest. Pilot outputs have `-pilot` suffix.

Only `phase: complete` and successful final survey manifests mean the full
corpus run finished. Download-unavailable, parse-failed and extraction-failed
papers remain explicitly counted. Successful gates require at least 95%
parse/basic/deep coverage and 500 validated core candidates per domain;
they do not claim that every online paper was found or parsed.
