import argparse

import miles_ext_kernels


def test_every_kernel_package_registers_and_validates_its_arguments():
    parser = argparse.ArgumentParser()
    miles_ext_kernels.add_arguments(parser)

    miles_ext_kernels.validate_args(parser.parse_args([]))
