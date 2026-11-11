"""The backward form of a ``fast`` ``log_probs`` call without ``num_valid_rows``, by the vendored package's own rules
(``cake_lm_head_nondet_train`` README, "Short-tail fold" and "Vocabulary-major log-probability backward"). The GPU
tests compare ``fast``'s ``dW`` bits with ``det``'s, or with the call given the count, only where the package
documents them equal: no folded short tail and no vocabulary-major backward (and no zero-gradient row)."""

from importlib import import_module

from miles_ext_kernels.cake._impl.lm_head import ops


def _backend():
    return import_module(".cake_backend", ops._package("fast").__name__)


def fast_folds(num_rows: int, rows: int, chunk: int) -> bool:
    """Does the plan of a call of ``num_rows`` rows (``T``) whose chunk loop runs ``rows`` rows (the valid rows, or
    ``T`` when every row is valid) fold its short tail (``loop_chunks`` against ``plan_chunks``)?"""
    backend = _backend()
    return len(backend.loop_chunks(num_rows, rows, chunk)) < len(backend.plan_chunks(rows, chunk))


def fast_vocab_major(rows: int, chunk: int) -> bool:
    """Does the backward of a call over ``rows`` loop rows that trains ``W`` run vocabulary-major (the auto rule's
    row counts, at most ``VM_MAX_ROWS`` rows)?"""
    backend = _backend()
    return 1 <= rows <= backend.VM_MAX_ROWS and backend._logprob_vm_auto_rows(rows, chunk)


def fast_keeps_det_dw(num_rows: int, rows: int, chunk: int) -> bool:
    """Is ``fast``'s ``dW`` documented as ``det``'s bits for this call (absent zero-gradient rows): the call neither
    folds its short tail nor runs the vocabulary-major backward, so its chunk loop runs the chunks of ``chunk``
    rows."""
    return not (fast_folds(num_rows, rows, chunk) or fast_vocab_major(rows, chunk))
