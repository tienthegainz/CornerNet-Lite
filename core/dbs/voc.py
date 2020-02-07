import os
import json
import numpy as np

from .detection import DETECTION
from ..paths import get_file_path


class VOC(DETECTION):
    def __init__(self, db_config, split=None, sys_config=None):
        assert split is None or sys_config is not None
        super(VOC, self).__init__(db_config)

        self._mean = np.array([0.40789654, 0.44719302, 0.47026115], dtype=np.float32)
        self._std = np.array([0.28863828, 0.27408164, 0.27809835], dtype=np.float32)
        self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571], dtype=np.float32)
        self._eig_vec = np.array([
            [-0.58752847, -0.69563484, 0.41340352],
            [-0.5832747, 0.00994535, -0.81221408],
            [-0.56089297, 0.71832671, 0.41158938]
        ], dtype=np.float32)

        

        self._voc_cls_names = [
            "balo", "balo_diutre", "bantreem", "binh_sua", "cautruot", "coc_sua", "ghe_an",
            "ghe_bap_benh", "ghe_ngoi_oto", "ghedualung_treem", "ke", "noi", "person", "phao",
            "quay_cui", "tham", "thanh_chan_cau_thang", "thanh_chan_giuong", "xe_babanh", "xe_choichan",
            "xe_day", "xe_tapdi", "xichdu", "yem",
        ]  # 24 class

        self._voc_cls_ids = list(range(1, len(self._voc_cls_names)+1))
        self._cls2voc = {ind + 1: voc_id for ind, voc_id in enumerate(self._voc_cls_ids)}
        self._voc2cls = {voc_id: cls_id for cls_id, voc_id in self._cls2voc.items()}

        self._voc2name = {cls_id: cls_name for cls_id, cls_name in zip(self._voc_cls_ids, self._voc_cls_names)}
        self._name2voc = {cls_name: cls_id for cls_name, cls_id in self._voc2name.items()}

        

    def _load_voc_annos(self):
        

    def image_path(self, ind):
        

    def detections(self, ind):
        

    def cls2name(self, cls):
        voc = self._cls2voc[cls]
        return self._voc2name[voc]

    def _to_float(self, x):
        return float('{:.2f}'.format(x))

    def convert_to_coco(self, all_bboxes):
        detections = []
        for image_id in all_bboxes:
            coco_id = self._eval_ids[image_id]
            for cls_ind in all_bboxes[image_id]:
                category_id = self._cls2voc[cls_ind]
                for bbox in all_bboxes[image_id][cls_ind]:
                    bbox[2] -= bbox[0]
                    bbox[3] -= bbox[1]

                    score = bbox[4]
                    bbox = list(map(self._to_float, bbox[0:4]))

                    detection = {
                        'image_id': int(coco_id),
                        'category_id': int(category_id),
                        'bbox': bbox,
                        'score': float('{:.2f}'.format(score))
                    }

                    detections.append(detection)
        return detections

    def evaluate(self, result_json, cls_ids, image_ids):
        return None
        