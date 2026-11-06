"""The shapes the prebuild contract test runs (``test_prebuild_gpu.py``), torch-free so that a CPU test checks them
against the packages' program selection (``test_conv1d_ops.py``).

The packages pick a call's program from the architecture, the route, the channel count and, for some programs, the
token count. These points reach every program of every route on every architecture: the fallback programs (136
channels) and each channel threshold of the selection rules (256 to 2560 channels) on 4097 tokens in four documents
(one empty, one of 3 tokens), 65536 tokens at 264 channels (the token floor of the det row-group backward programs
from 264 channels: 65536 on SM100a, 24576 on SM103a, 32768 on SM107a), and 262401 tokens at 256 channels (past the
262144-token floor of an SM107a backward program).
"""

# (channels, document lengths) of the single-rank calls
SINGLE_RANK_POINTS = (
    *((C, (1000, 0, 3, 3094)) for C in (136, 256, 264, 512, 768, 1024, 1032, 2048, 2560)),
    (264, (30000, 0, 3, 35533)),
    (256, (100000, 162401)),
)
# channels of the context-parallel calls (one rank holding every fragment of a cut pack)
KCP_CHANNELS = (256, 512, 1024, 2560)
