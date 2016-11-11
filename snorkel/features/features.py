from snorkel.features.content_features import *
from snorkel.features.table_features import *
from snorkel.features.visual_features import *


def get_all_feats(candidate):
    for f, v in get_content_feats(candidate):
        yield f, v
    for f, v in get_table_feats(candidate):
        yield f, v
    for f, v in get_visual_feats(candidate):
        yield f, v
