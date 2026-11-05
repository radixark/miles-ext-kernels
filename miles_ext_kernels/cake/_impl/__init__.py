"""The implementations of the Cake ops: ``_impl/<op>/`` belongs to the op module ``miles_ext_kernels.cake.<op>``.

An op's package calls ``loader.install_op("<op>")`` at import and defines its entry points (the op's ``OP.exports``),
``prebuild(*, device=None, max_workers=None, **model_params)`` and ``release()``. It reads its variant and options
when a call runs (``config.variant`` / ``config.option``), takes temporary buffers from ``workspace.take`` and never
reads the environment or imports flashinfer.
"""
