"""Keep unavailable infrastructure distinct from invalid scientific proposals."""
from .contracts import ProviderError, ProviderContractError, ProviderRecoveryBlockedError


def raise_infrastructure_failure(error: BaseException) -> None:
    seen = set()
    current = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, ProviderRecoveryBlockedError) or (isinstance(current, ProviderError) and not isinstance(current, ProviderContractError)):
            raise current
        current = current.__cause__ or current.__context__
