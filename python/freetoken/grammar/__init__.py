"""Grammar-constrained decoding for Structured Outputs and strict tool calling."""

from .backend import (
    GrammarBackend,
    GrammarCompiler,
    GrammarMatcher,
    GrammarUnavailableError,
    LLGuidanceBackend,
    XGrammarBackend,
    is_grammar_available,
    is_llguidance_installed,
    is_xgrammar_installed,
)
from .bitmask import (
    allocate_host_bitmask,
    apply_token_bitmask_inplace,
    bitmask_allows_token,
    bitmask_word_count,
)
from .schema import (
    SchemaValidationError,
    build_structured_output_spec,
    canonical_schema_json,
    resolve_force_reasoning,
    resolve_reasoning_grammar_mode,
    validate_json_schema,
)

__all__ = [
    "GrammarBackend",
    "GrammarCompiler",
    "GrammarMatcher",
    "GrammarUnavailableError",
    "LLGuidanceBackend",
    "SchemaValidationError",
    "XGrammarBackend",
    "allocate_host_bitmask",
    "apply_token_bitmask_inplace",
    "bitmask_allows_token",
    "bitmask_word_count",
    "build_structured_output_spec",
    "canonical_schema_json",
    "is_grammar_available",
    "is_llguidance_installed",
    "is_xgrammar_installed",
    "resolve_force_reasoning",
    "resolve_reasoning_grammar_mode",
    "validate_json_schema",
]
