# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


__doc__ = """This module contains constants used by both C++ and gateware.

The shared contants are mostly op code numbers and register IDs.
"""

import argparse
import sys


class Constants:
    """Constants shared by C++ and Gateware."""

    ###########################################################################
    # Register IDs
    # For convenience, readable and writable register IDs are allocated from a
    # shared pool of values 0-127.

    # A write of any value to REG_RESET causes the accelerator gateware to be
    # reset and lose all state
    REG_RESET = 0

    # Number of 32 bit filter words
    REG_FILTER_NUM_WORDS = 1

    # Number of 32 bit input words
    REG_INPUT_NUM_WORDS = 2

    # Input offset for multiply-accumulate unit
    REG_INPUT_OFFSET = 3

    # Set next filter word
    REG_SET_FILTER = 4

    # Set next input word
    REG_SET_INPUT = 5

    # These registers contain values 0-3 of the current filter word
    REG_FILTER_0 = 0x10
    REG_FILTER_1 = 0x11
    REG_FILTER_2 = 0x12
    REG_FILTER_3 = 0x13

    # These registers contain values 0-3 of the current input word
    REG_INPUT_0 = 0x18
    REG_INPUT_1 = 0x19
    REG_INPUT_2 = 0x1a
    REG_INPUT_3 = 0x1b

    # Set input values to multiply-accumulate unit
    REG_MACC_INPUT_0 = 0x20
    REG_MACC_INPUT_1 = 0x21
    REG_MACC_INPUT_2 = 0x22
    REG_MACC_INPUT_3 = 0x23

    # Set input values to multiply-accumulate unit
    REG_MACC_FILTER_0 = 0x28
    REG_MACC_FILTER_1 = 0x29
    REG_MACC_FILTER_2 = 0x2a
    REG_MACC_FILTER_3 = 0x2b

    # Retrieve result from multiply-accumulate unit
    REG_MACC_OUT = 0x30

    ###########################################################################
    # Funct3 codes - used to route CFU to instructions
    INS_SET = 0
    INS_GET = 1
    INS_PING = 7


def main():
    parser = argparse.ArgumentParser(
        description='Write C header file with constants')
    'outfile',
    parser.add_argument('output', metavar='FILE', nargs='?',
                        type=argparse.FileType('w'), default=sys.stdout,
                        help='Where to send output')
    args = parser.parse_args()

    with args.output as f:
        print("// Generated file", file=f)
        print("// Shared constants generated from gateware/constants.py",
              file=f)
        for name, value in vars(Constants).items():
            if not name.startswith('_'):
                print(f"#define {name} {value}", file=f)


if __name__ == "__main__":
    main()