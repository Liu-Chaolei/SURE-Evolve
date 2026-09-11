IDEA_RESULT_ALIGNMENT_PROMPT = """
You are the final public-facing alignment editor for a research idea before it is written to idea_result.json.

== Topic ==
{topic}

== Mature idea anchor (optional; if empty, ignore) ==
{mature_idea}

== Refinement scope (optional; if empty, ignore) ==
{refinement_scope}

== Candidate idea payload (JSON) ==
{idea}

== Grounding papers ==
{papers}

Your job is to rewrite the candidate into a final public-facing idea description and the public experiment contract that will be written to idea_result.json and research_idea.json.

Rules:
- If a mature idea anchor is provided, present the candidate as a direct refinement of that anchor. The public narrative should make the relationship to the mature idea explicit.
- Do NOT present internal MCTS intermediate artifacts, temporary aliases, or parent-node names as if they were user-facing baselines or prior methods.
- If the candidate currently describes itself mainly as a repair of an intermediate node, rewrite it so it reads as a refinement of the mature idea instead.
- Keep temporary internal names only if they remain essential to the final public method. Otherwise, fold them into a final name that is anchored in the mature idea and the actual mechanism.
- In `abstract`, `method`, `research_question`, and `hypothesis`, explicitly state what limitation of the mature idea is being repaired and how.
- If contrasting with other methods or baselines, only contrast against the provided papers. Do NOT compare against internal search artifacts, parent nodes, or MCTS intermediate candidates.
- Stay faithful to the actual mechanism in the candidate. Do not invent a different method.
- Keep the method concrete and implementation-ready, but remove internal-search framing.
- `experiment_plan`, `data_requirements`, `baselines`, and `metrics` must be concrete lists generated from the candidate and grounding papers. Do not return placeholders such as "primary_task_metric", "baseline", "dataset", "TBD", or generic survey boilerplate.

Return STRICT JSON only:
{{
  "title": "public-facing paper title",
  "abstract": "public-facing abstract aligned to the mature idea when present",
  "core_contribution": "public-facing main mechanism claim",
  "research_question": "specific research question answered by the final idea",
  "hypothesis": "specific falsifiable hypothesis tied to the final mechanism",
  "method": "public-facing method description aligned to the mature idea when present",
  "experiment_plan": ["provider-generated experiment step grounded in the final idea and papers"],
  "data_requirements": ["provider-generated dataset/resource requirement grounded in the task and papers"],
  "baselines": ["provider-generated paper-grounded or method-grounded comparison baseline"],
  "metrics": ["provider-generated evaluation metric tied to the hypothesis"],
  "risks": ["public-facing risk"]
}}
"""
