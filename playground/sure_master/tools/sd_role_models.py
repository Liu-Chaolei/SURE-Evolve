"""Explicit API roles for SD native research; unknown operations fail closed."""

GLM = 'sure-glm-router'
OPENAI = 'sure-openai-router'
GLM_OPERATIONS = frozenset({
    'xlab.sure.parent_projection.v1',
    'xlab.sure.candidate_review.v1',
    'xlab.research_idea.background.generate.v1',
    'xlab.research_idea.retrieval.query.v1',
    'xlab.research_idea.reference.rank.v1',
    'xlab.research_idea.keynote.score.v1',
    'xlab.research_idea.keynote.compress.v1',
    'xlab.research_idea.keynote.rollup.v1',
    'xlab.research_idea.operator.theory-transfer-query.v1',
    'xlab.research_idea.operator.mechanism-commit-query.v1',
    'xlab.research_idea.idea.diagnostic.v1',
    'xlab.research_idea.idea.evaluate.v1',
    'xlab.research_idea.component_novelty.evaluate.v1',
})
OPENAI_OPERATIONS = frozenset({
    'xlab.research_idea.analysis.generate.v1',
    'xlab.research_idea.analysis.replan.v1',
    'xlab.research_idea.idea.generate.v1',
    'xlab.research_idea.idea.materialize.v1',
    'xlab.research_idea.fusion.generate.v1',
    'xlab.research_idea.fusion.referee.v1',
    'xlab.research_idea.fusion.repair.v1',
})


def operation_model(operation: str) -> str:
    if operation in GLM_OPERATIONS:
        return GLM
    if operation in OPENAI_OPERATIONS:
        return OPENAI
    raise ValueError(f'No approved model role for operation: {operation}')
