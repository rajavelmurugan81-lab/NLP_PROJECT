"""Input validation utilities for scientific claim verification requests."""

from typing import Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)

MIN_CLAIM_LENGTH = 5
MAX_CLAIM_LENGTH = 1000


def validate_claim_input(claim: Any) -> Tuple[bool, str, Optional[str]]:
    """Validate and normalize user-submitted scientific claim input.

    Returns:
        Tuple of (is_valid: bool, normalized_claim_text: str, error_message: Optional[str])
    """
    if claim is None:
        return False, "", "Claim input cannot be None."

    if not isinstance(claim, str):
        return False, str(claim), f"Claim input must be a string, got {type(claim).__name__}."

    normalized = claim.strip()

    if not normalized:
        return False, "", "Claim input cannot be empty or whitespace-only."

    if len(normalized) < MIN_CLAIM_LENGTH:
        return False, normalized, f"Claim input is too short ({len(normalized)} chars). Minimum length is {MIN_CLAIM_LENGTH} characters."

    if len(normalized) > MAX_CLAIM_LENGTH:
        return False, normalized[:MAX_CLAIM_LENGTH], f"Claim input is too long ({len(normalized)} chars). Maximum length is {MAX_CLAIM_LENGTH} characters."

    return True, normalized, None
