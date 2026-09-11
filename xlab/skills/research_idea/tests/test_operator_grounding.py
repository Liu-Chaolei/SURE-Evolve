from __future__ import annotations

import sys
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PACKAGE_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from research_idea_lib.algorithm.contracts import IdeaComponent, IdeaState  # noqa: E402
from research_idea_lib.algorithm.operator_grounding import (  # noqa: E402
    MECHANISM_COMMIT_OPERATOR,
    THEORY_TRANSFER_OPERATOR,
    OperatorComponentHit,
    OperatorComponentRetrieval,
    OperatorGroundingEvaluator,
    OperatorQuery,
)


def idea() -> IdeaState:
    return IdeaState(
        "Candidate", "A candidate abstract.", "A mechanism.", "Test it.",
        "Distribution shift.", (IdeaComponent("core", "Current mechanism."),),
        root_domains=("Computer Science",),
    )


class Queries:
    def theory_transfer_query(self, state, *, prompt_mode):
        return OperatorQuery("theory_transfer", "transfer query", "transfer", "stabilize", '{"op":"theory"}')

    def mechanism_commit_query(self, state, *, prompt_mode):
        return OperatorQuery("mechanism_commit", "mechanism query", "gate", "route", '{"op":"mechanism"}')


class Retriever:
    def __init__(self, hits):
        self.hits = hits
        self.limits = []

    def retrieve_operator_components(self, query, *, limit):
        self.limits.append(limit)
        return OperatorComponentRetrieval(tuple(self.hits), '{"adapter":"fake"}')


def hit(node, component, score, domain, *, paper_title="", full_name="", label=""):
    return OperatorComponentHit(
        node,
        component,
        component,
        f"{component} description",
        score,
        domain,
        (f"paper:{component}",),
        "components:v1",
        f"trace:{component}",
        paper_title,
        full_name,
        label,
    )


def test_theory_groups_before_filtering_and_bounds_core_nodes_and_components():
    hits = [
        hit("n1", "a", .9, "Biology"), hit("n1", "b", .4, "Biology"),
        hit("n1", "c", .7, "Biology"), hit("n2", "d", .6, "computer science"),
        hit("n3", "e", .5, None), hit("n4", "f", .6, "Physics"),
        hit("n5", "g", .95, "Chemistry"), hit("n6", "h", .85, "Medicine"),
    ]
    retriever = Retriever(hits)
    result = OperatorGroundingEvaluator(Queries(), retriever, top_k=2).evaluate(THEORY_TRANSFER_OPERATOR, idea())
    assert retriever.limits == [12]
    assert [(item.core_node_id, item.component_id) for item in result.grounding.evidence] == [
        ("n5", "g"), ("n1", "a"), ("n1", "c")
    ]
    assert not result.skip_operator
    assert "paper:a" in result.references_provenance_json


def test_theory_filters_full_ranked_pool_before_top_k_selection():
    result = OperatorGroundingEvaluator(Queries(), Retriever([
        hit("n1", "a", .99, "Computer Science"),
        hit("n2", "b", .98, "computer science"),
        hit("n3", "c", .97, None),
        hit("n4", "d", .96, "Biology"),
    ]), top_k=3).evaluate(THEORY_TRANSFER_OPERATOR, idea())
    assert not result.skip_operator
    assert [item.core_node_id for item in result.grounding.evidence] == ["n4"]



def test_component_and_node_tie_order_matches_upstream_contract():
    result = OperatorGroundingEvaluator(
        Queries(),
        Retriever(
            [
                hit("node-z", "zeta", .8, None, paper_title="Beta"),
                hit("node-z", "Alpha", .8, None, paper_title="Beta"),
                hit("node-z", "middle", .8, None, paper_title="Beta"),
                hit("node-a", "other", .8, None, paper_title="Alpha"),
            ]
        ),
        top_k=1,
    ).evaluate(MECHANISM_COMMIT_OPERATOR, idea())

    assert [item.component_id for item in result.grounding.evidence] == ["other"]

    same_node = OperatorGroundingEvaluator(
        Queries(),
        Retriever(
            [
                hit("node", "zeta", .8, None),
                hit("node", "Alpha", .8, None),
                hit("node", "middle", .8, None),
            ]
        ),
    ).evaluate(MECHANISM_COMMIT_OPERATOR, idea())
    assert [item.component_id for item in same_node.grounding.evidence] == ["Alpha", "middle"]


def test_equal_node_keys_preserve_component_rank_encounter_order():
    result = OperatorGroundingEvaluator(
        Queries(),
        Retriever(
            [
                hit("node-z", "Alpha", .8, None, paper_title="Shared"),
                hit("node-a", "zeta", .8, None, paper_title="Shared"),
            ]
        ),
        top_k=1,
    ).evaluate(MECHANISM_COMMIT_OPERATOR, idea())
    assert [item.core_node_id for item in result.grounding.evidence] == ["node-z"]


def test_complete_component_ties_preserve_encounter_order():
    first = hit("node", "same", .8, None)
    second = OperatorComponentHit(
        "node", "different-id", "same", "second description", .8, None,
        ("paper:second",), "components:v1", "trace:second"
    )
    result = OperatorGroundingEvaluator(
        Queries(), Retriever([first, second]), top_k=1
    ).evaluate(MECHANISM_COMMIT_OPERATOR, idea())
    assert [item.component_id for item in result.grounding.evidence] == ["same", "different-id"]


def test_operator_specific_empty_semantics_and_thresholds():
    theory = OperatorGroundingEvaluator(Queries(), Retriever([hit("n", "a", .59, "Biology")])).evaluate(THEORY_TRANSFER_OPERATOR, idea())
    assert theory.skip_operator and not theory.explicit_empty
    accepted_theory = OperatorGroundingEvaluator(Queries(), Retriever([hit("n", "a", .6, "Biology")])).evaluate(THEORY_TRANSFER_OPERATOR, idea())
    assert not accepted_theory.skip_operator
    mechanism = OperatorGroundingEvaluator(Queries(), Retriever([hit("n", "a", .59, None)])).evaluate(MECHANISM_COMMIT_OPERATOR, idea())
    assert mechanism.explicit_empty and not mechanism.skip_operator
    accepted_mechanism = OperatorGroundingEvaluator(Queries(), Retriever([hit("n", "a", .6, None)])).evaluate(MECHANISM_COMMIT_OPERATOR, idea())
    assert not accepted_mechanism.explicit_empty


def test_mechanism_allows_same_and_missing_domains():
    result = OperatorGroundingEvaluator(Queries(), Retriever([
        hit("n1", "a", .6, "computer science"), hit("n2", "b", .7, None)
    ])).evaluate(MECHANISM_COMMIT_OPERATOR, idea())
    assert {item.component_id for item in result.grounding.evidence} == {"a", "b"}
