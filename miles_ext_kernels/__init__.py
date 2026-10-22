"""Training kernels for miles, enabled by miles' ``--use-miles-ext-kernels``.

miles owns backend selection (``--linear-attention-backend cake``, ...) and imports the kernel package at its
dispatch seam. Each kernel package here owns its own arguments and validates them.
"""

from argparse import ArgumentParser, Namespace

# Each entry is a module exposing add_arguments(group) and validate_args(args).
_KERNEL_PACKAGES = ()


def add_arguments(parser: ArgumentParser) -> None:
    group = parser.add_argument_group("miles-ext-kernels")
    for package in _KERNEL_PACKAGES:
        package.add_arguments(group)


def validate_args(args: Namespace) -> None:
    for package in _KERNEL_PACKAGES:
        package.validate_args(args)
