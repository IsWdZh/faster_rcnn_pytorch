from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from PIL import Image
from data.pascal_voc import pascal_voc
def data(imdb_names):
    def filter_roidb(roidb):
        # filter the image without bounding box.
        print('before filtering, there are %d images...' % (len(roidb)))
        i = 0
        while i < len(roidb):
            if len(roidb[i]['boxes']) == 0:
                del roidb[i]
                i -= 1
            i += 1

        print('after filtering, there are %d images...' % (len(roidb)))
        return roidb

    dbtype, year = imdb_names.split("_")[2], imdb_names.split("_")[1]  # trainval, 2007
    imdb = pascal_voc(dbtype, year)
    imdb.set_proposal_method('gt')

    roidb = imdb.roidb



    roidb = filter_roidb(roidb)


