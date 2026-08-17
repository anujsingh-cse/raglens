"""Exception hierarchy for RagLens."""


class RagLensError(Exception):
    """Base class for all RagLens errors."""


class ProviderError(RagLensError):
    """Raised when the underlying LLM provider fails or is misconfigured."""


class DatasetError(RagLensError):
    """Raised when a dataset cannot be loaded or is malformed."""


class AttributionError(RagLensError):
    """Raised when attribution analysis cannot be completed for a sample."""
