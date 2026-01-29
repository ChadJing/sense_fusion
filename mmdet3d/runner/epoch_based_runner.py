from mmcv.runner import EpochBasedRunner
from mmcv.runner.builder import RUNNERS

@RUNNERS.register_module()
class CustomEpochBasedRunner(EpochBasedRunner):
    def set_dataset(self, dataset):
        self._dataset = dataset


    def train(self, data_loader, **kwargs):
        # update the schedule for data augmentation
        for dataset in self._dataset:
            dataset.set_epoch(self.epoch)

        import torch.distributed as dist
        import os
        if not dist.is_initialized():
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = '12345'
            dist.init_process_group(backend='nccl',rank=0, world_size = 1)

        super().train(data_loader, **kwargs)
