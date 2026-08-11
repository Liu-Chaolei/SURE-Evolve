from __future__ import annotations

from collections.abc import Iterable

from paper_library.schemas.research_design_card import ResearchDesignCard
from paper_library.utils.ids import stable_id


def card_id_for(card: ResearchDesignCard) -> str:
    identity = (
        card.tested_relationship.value
        or card.intervention_delta.value
        or card.candidate_system.value
        or card.title
    )
    return stable_id("card", card.paper_id, card.card_type.value, identity.casefold())


def ensure_stable_card_ids(cards: Iterable[ResearchDesignCard]) -> None:
    seen: set[str] = set()
    for card in cards:
        expected = card_id_for(card)
        if card.card_id != expected:
            raise ValueError(f"Card ID {card.card_id!r} does not match stable identity: {expected}")
        if card.card_id in seen:
            raise ValueError(f"Duplicate Card ID: {card.card_id}")
        seen.add(card.card_id)
