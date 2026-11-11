"""The shapes the prebuild contract test runs (``test_prebuild_gpu.py``), torch-free so that a CPU test checks them
against the packages' program selection (``test_conv1d_ops.py``).

The packages pick a call's program from the architecture, the route, the channel count and, for some programs, the
token count. These points reach every program of every route on every architecture: the fallback programs (136
channels) and each channel threshold of the selection rules (256 to 2560 channels) on 4097 tokens in four documents
(one empty, one of 3 tokens), 65536 tokens at 264 channels (the token floor of the det row-group backward programs
from 264 channels: 65536 on SM100a, 24576 on SM103a, 32768 on SM107a; and of the SM107a two-stage forward with FP32
taps from 256 channels) and at 512 channels (the same floor of the SM107a two-stage forward with BF16 taps from 512
channels), and 262401 tokens at 256 channels (past the 262144-token floor of an SM107a backward program). The
context-parallel calls reach every program of the context-parallel routes and of the single-rank forward route their
forward runs as its main program.
"""

# (channels, document lengths) of the single-rank calls
SINGLE_RANK_POINTS = (
    *((C, (1000, 0, 3, 3094)) for C in (136, 256, 264, 512, 768, 1024, 1032, 2048, 2560)),
    (264, (30000, 0, 3, 35533)),
    (512, (30000, 0, 3, 35533)),
    (256, (100000, 162401)),
)
# channels of the context-parallel calls (one rank holding every fragment of a cut pack)
KCP_CHANNELS = (256, 512, 1024, 2560)
# the cut pack of those calls: history rows from two source fragments, head rows with consumers
KCP_DOCUMENTS, KCP_CUTS = (5000, 1003), (1, 2, 4997, 1003)
# (channels, document lengths) of one context-parallel call on a pack of 65536 tokens, one fragment per document: the
# context-parallel forward runs the single-rank forward program of FP32 taps as its main program, and this call
# reaches the SM107a two-stage one (from 256 channels and 65536 tokens)
KCP_LONG_POINT = (256, (30000, 35536))
