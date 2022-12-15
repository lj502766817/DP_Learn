# Copyright (c) OpenMMLab. All rights reserved.
from .builder import DATASETS
from .custom import CustomDataset
from mmcv.utils import print_log
from mmseg.utils import get_root_logger
import mmcv


@DATASETS.register_module()
class CarDataset(CustomDataset):
    CLASSES = (
        'background', 'car', 'wheel', 'lights', 'window'
    )

    PALETTE = [[112, 128, 144], [0, 255, 255], [128, 0, 0], [255, 255, 0], [128, 128, 0]]

    def __init__(self, **kwargs):
        super(CarDataset, self).__init__(
            img_suffix='.png', seg_map_suffix='.png', **kwargs)

    def load_annotations(self, img_dir, img_suffix, ann_dir, seg_map_suffix, split):
        img_infos = []
        if split is not None:
            lines = mmcv.list_from_file(
                split, file_client_args=self.file_client_args)
            for line in lines:
                img_name = line.strip()
                img_info = dict(filename=img_name + img_suffix)
                if ann_dir is not None:
                    seg_map = img_name + seg_map_suffix
                    img_info['ann'] = dict(seg_map=seg_map)
                img_infos.append(img_info)
        else:
            for img in self.file_client.list_dir_or_file(
                    dir_path=img_dir,
                    recursive=True):
                img_info = dict(filename=img)
                if ann_dir is not None:
                    seg_map = img.replace(img_suffix, seg_map_suffix)
                    img_info['ann'] = dict(seg_map=seg_map)
                img_infos.append(img_info)
            img_infos = sorted(img_infos, key=lambda x: x['filename'])

        print_log(f'Loaded {len(img_infos)} images', logger=get_root_logger())
        return img_infos
