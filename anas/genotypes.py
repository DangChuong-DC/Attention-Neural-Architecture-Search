from collections import namedtuple

AttGenotype = namedtuple('AttGenotype', 'att')

ATT_PRIMITIVES = [
    'none',
    'skip_connect',
    'relu_conv_bn_1x1',
    'sep_conv_3x3',
    'dil_conv_3x3',
    'avg_pool_3x3',
    'max_pool_3x3',
    'channel_score',
    'spatial_score'
]


ATTENTION = AttGenotype(
    att=[('dil_conv_3x3', 1), ('dil_conv_3x3', 0), ('sep_conv_3x3', 1), ('channel_score', 2), ('spatial_score', 2), ('channel_score', 3)]
)
