import time
import mmcv
import numpy as np
import torch

from mmcv.transforms import Compose
from mmdet.utils import get_test_pipeline_cfg
from mmengine.config import ConfigDict
from mmyolo.utils import register_all_modules

from backendwrapper import TRTWrapper

register_all_modules()


def preprocess(config):
    data_preprocess = config.get('model', {}).get('data_preprocessor', {})
    mean = data_preprocess.get('mean', [0., 0., 0.])
    std = data_preprocess.get('std', [1., 1., 1.])
    mean = torch.tensor(mean, dtype=torch.float32).reshape(1, 3, 1, 1)
    std = torch.tensor(std, dtype=torch.float32).reshape(1, 3, 1, 1)

    class PreProcess(torch.nn.Module):

        def __init__(self):
            super().__init__()

        def forward(self, x):
            x = x[None].float()
            x -= mean.to(x.device)
            x /= std.to(x.device)
            return x

    return PreProcess().eval()


class DetectorRTMDet(object):
    def __init__(self, engine_path, cfg):
        
        self.device = 'cuda:0'
        self.model = TRTWrapper(engine_path, self.device)
        self.model.to(self.device)
        
        class_names = cfg.get('class_name')

        self.test_pipeline = get_test_pipeline_cfg(cfg)
        self.test_pipeline[0] = ConfigDict({'type': 'mmdet.LoadImageFromNDArray'})
        self.test_pipeline = Compose(self.test_pipeline)

        self.pre_pipeline = preprocess(cfg)
        
    def detect(self, img, conf_th=0.3, id=1, is_profiling=False):
        timeline = []
        ### preprocess ###
        tic = time.time()
        rgb = mmcv.imconvert(img, 'bgr', 'rgb')
        data, samples = self.test_pipeline(dict(img=rgb, img_id=id)).values()
        pad_param = samples.get('pad_param',
                                np.array([0, 0, 0, 0], dtype=np.float32))
        h, w = samples.get('ori_shape', rgb.shape[:2])
        pad_param = torch.asarray(
            [pad_param[2], pad_param[0], pad_param[2], pad_param[0]],
            device=self.device)
        scale_factor = samples.get('scale_factor', [1., 1])
        scale_factor = torch.asarray(scale_factor * 2, device=self.device)
        data = self.pre_pipeline(data).to(self.device)
        toc = time.time()
        timeline.append(toc-tic)
        
        ### inference ###
        tic = time.time()
        result = self.model(data)
        toc = time.time()
        timeline.append(toc-tic)
        
        ### postprocess ###
        # Get candidate predict info by num_dets
        tic = time.time()
        num_dets, bboxes, scores, labels = result
        scores = scores[0, :num_dets]
        bboxes = bboxes[0, :num_dets]
        labels = labels[0, :num_dets]
        bboxes -= pad_param
        bboxes /= scale_factor

        bboxes[:, 0::2].clamp_(0, w)
        bboxes[:, 1::2].clamp_(0, h)
        bboxes = bboxes.round().int()
        toc = time.time()
        timeline.append(toc-tic)
        
        if is_profiling:
            sum_timeline = sum(timeline)
            print("Time breaking:\n PRE-", str(float(timeline[0])), " ", str(float(timeline[0]) / sum_timeline), "\n", \
                " INF-", str(float(timeline[1])), " ", str(float(timeline[1]) / sum_timeline), "\n", \
                " POST-", str(float(timeline[2])), " ", str(float(timeline[2]) / sum_timeline))
        
        
        return bboxes.tolist(), scores.tolist(), labels.tolist()