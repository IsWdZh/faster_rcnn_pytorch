import numpy as np
from PIL import Image
from data.pascal_voc import pascal_voc
from logger import get_logger


logger = get_logger()
class PrepareData():
    def __init__(self, imdb_names):
        self.dbtype = imdb_names.split("_")[2]    # trainval
        self.year = imdb_names.split("_")[1]     # 2007

    def combine(self):
        roidb = self.getroidb()
        imdb = pascal_voc(self.dbtype, self.year)
        roidb = self.filter_roidb(roidb)
        ratio_list, ratio_index = self.rank_roidb_ratio(roidb)

        return imdb, roidb, ratio_list, ratio_index


    def getroidb(self):
        self.imdb = pascal_voc(self.dbtype, self.year)

        # logger.info("Appending horizontally-flipped training examples")
        logger.info('Before horizontally-flipping, there are {} images'.format(len(self.imdb.image_index)))
        self.imdb.append_flipped_images()   # 水平翻转图像
        logger.info('After horizontally-flipping, there are {} images'.format(len(self.imdb.image_index)))

        self.prepare_roidb(self.imdb)
        self.roidb = self.imdb.roidb

        return self.roidb


    def prepare_roidb(self, imdb):
        """Enrich the imdb's roidb by adding some derived quantities that
        are useful for training. This function precomputes the maximum
        overlap, taken over ground-truth boxes, between each ROI and
        each ground-truth box. The class with maximum overlap is also
        recorded.
        """
        roidb = imdb.roidb
        sizes = [Image.open(imdb.image_path_at(i)).size
                 for i in range(imdb.num_images)]

        for i in range(len(imdb.image_index)):
            roidb[i]['img_id'] = i
            roidb[i]['image'] = imdb.image_path_at(i)  # pic's absolutly path

            roidb[i]['width'] = sizes[i][0]
            roidb[i]['height'] = sizes[i][1]

            # need gt_overlaps as a dense array for argmax
            gt_overlaps = roidb[i]['gt_overlaps'].toarray()
            # max overlap with gt over classes (columns)
            max_overlaps = gt_overlaps.max(axis=1)
            # gt class that had the max overlap
            max_classes = gt_overlaps.argmax(axis=1)
            roidb[i]['max_classes'] = max_classes
            roidb[i]['max_overlaps'] = max_overlaps
            # sanity checks
            # max overlap of 0 => class should be zero (background)
            zero_inds = np.where(max_overlaps == 0)[0]
            assert all(max_classes[zero_inds] == 0)
            # max overlap > 0 => class should not be zero (must be a fg class)
            nonzero_inds = np.where(max_overlaps > 0)[0]
            assert all(max_classes[nonzero_inds] != 0)

    def rank_roidb_ratio(self, roidb):
        # rank roidb based on the ratio between width and height.
        ratio_large = 2  # largest ratio to preserve.
        ratio_small = 0.5  # smallest ratio to preserve.

        ratio_list = []
        for i in range(len(roidb)):
            width = roidb[i]['width']
            height = roidb[i]['height']
            ratio = width / float(height)

            if ratio > ratio_large:
                roidb[i]['need_crop'] = 1
                ratio = ratio_large
            elif ratio < ratio_small:
                roidb[i]['need_crop'] = 1
                ratio = ratio_small
            else:
                roidb[i]['need_crop'] = 0

            ratio_list.append(ratio)

        ratio_list = np.array(ratio_list)
        ratio_index = np.argsort(ratio_list)
        return ratio_list[ratio_index], ratio_index

    def filter_roidb(self, roidb):
        # filter the image without bounding box.
        logger.info('before filtering, there are %d images...' % (len(roidb)))
        i = 0
        while i < len(roidb):
            if len(roidb[i]['boxes']) == 0:
                del roidb[i]
                i -= 1
            i += 1

        logger.info('after filtering, there are %d images...' % (len(roidb)))
        return roidb



