"""
简单直接的参数冻结Hook
"""

import torch
import torch.nn as nn
from mmcv.runner import HOOKS, Hook
import logging

@HOOKS.register_module()
class SimpleFreezeHook(Hook):
    """简单参数冻结Hook
    
    功能：
    1. 如果当前epoch小于freeze_epochs，就冻结指定的模块
    2. 如果当前epoch大于等于freeze_epochs，就跳过（不解冻也不做其他操作）
    
    Args:
        freeze_epochs (int): 冻结的epoch数
        freeze_prefixes (list): 要冻结的参数名前缀列表
    """
    
    def __init__(self, freeze_epochs=5, freeze_prefixes=None):
        self.freeze_epochs = freeze_epochs
        self.freeze_prefixes = freeze_prefixes or []
        
        # 保存初始权重用于检查
        self.initial_weights = {}
    
    def before_train_epoch(self, runner):
        """每个epoch训练前执行"""
        if runner.epoch < self.freeze_epochs:
            # 冻结指定模块
            for name, param in runner.model.named_parameters():
                for prefix in self.freeze_prefixes:
                    if name.startswith(prefix):
                        param.requires_grad = False
                        break
        elif runner.epoch >= self.freeze_epochs:
            for name, param in runner.model.named_parameters():
                for prefix in self.freeze_prefixes:
                    if name.startswith(prefix):
                        param.requires_grad = True
                        break
            # epoch达到设定值，跳过此hook（不执行任何操作）
        else:
            pass
    
    def before_run(self, runner):
        """训练开始前保存初始权重"""
        # 保存初始权重
        for name, param in runner.model.named_parameters():
            if param.requires_grad:
                for prefix in self.freeze_prefixes:
                    if name.startswith(prefix):
                        self.initial_weights[name] = param.data.clone()
                        break


@HOOKS.register_module()
class CheckFreezeHook(Hook):
    """检查冻结Hook是否正常运行
    
    功能：检查被冻结的模块权重是否有变化
    
    Args:
        freeze_epochs (int): 冻结的epoch数（需与SimpleFreezeHook一致）
        freeze_prefixes (list): 要检查的参数名前缀列表
        check_interval (int): 检查间隔（多少个epoch检查一次）
    """
    
    def __init__(self, freeze_epochs=5, freeze_prefixes=None, check_interval=1):
        self.freeze_epochs = freeze_epochs
        self.freeze_prefixes = freeze_prefixes or []
        self.check_interval = check_interval
        
        # 保存初始权重
        self.initial_weights = {}
        
        # 保存检查结果
        self.check_results = {}
    
    def before_run(self, runner):
        """训练开始前保存初始权重"""
        for name, param in runner.model.named_parameters():
            for prefix in self.freeze_prefixes:
                if name.startswith(prefix):
                    self.initial_weights[name] = param.data.clone()
                    break
    
    def after_train_epoch(self, runner):
        """每个epoch训练后检查权重变化"""
        if runner.epoch >= self.freeze_epochs:
            # 超过冻结epoch，停止检查
            return
            
        if runner.epoch % self.check_interval != 0:
            # 不是检查间隔，跳过
            return
        
        changed_params = []
        unchanged_params = []
        
        for name, param in runner.model.named_parameters():
            for prefix in self.freeze_prefixes:
                if name.startswith(prefix):
                    if name in self.initial_weights:
                        # 检查权重是否有变化
                        is_changed = not torch.allclose(
                            param.data, 
                            self.initial_weights[name],
                            rtol=1e-5, 
                            atol=1e-8
                        )
                        
                        if is_changed:
                            changed_params.append(name)
                        else:
                            unchanged_params.append(name)
                    break
        
        # 记录检查结果
        epoch_key = f'epoch_{runner.epoch}'
        self.check_results[epoch_key] = {
            'changed': changed_params,
            'unchanged': unchanged_params
        }
        
        # 打印检查结果
        if runner.logger:
            if changed_params:
                runner.logger.info(
                    f'CheckFreezeHook: Epoch {runner.epoch} - '
                    f'发现{len(changed_params)}个冻结参数有变化！'
                )
                for i, param_name in enumerate(changed_params[:3]):  # 只显示前3个
                    runner.logger.info(f'  变化参数: {param_name}')
                if len(changed_params) > 3:
                    runner.logger.info(f'  ... 还有{len(changed_params)-3}个参数有变化')
            else:
                runner.logger.info(
                    f'CheckFreezeHook: Epoch {runner.epoch} - '
                    f'所有冻结参数均无变化，Hook工作正常'
                )
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger('simple_runner')

# 测试代码
if __name__ == '__main__':
    print("测试SimpleFreezeHook")
    print("=" * 50)
    
    # 创建简单的测试模型
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Linear(10, 20)
            self.vtransform = nn.Linear(20, 30)
            self.fuser = nn.Linear(30, 40)
            self.head = nn.Linear(40, 10)
            
        def forward(self, x):
            return self.head(self.fuser(self.vtransform(self.backbone(x))))
    
    model = TestModel()
    
    # 模拟runner
    class MockRunner:
        def __init__(self, model):
            self.model = model
            self.epoch = 0
            self.logger = logger
        
        def train_epoch(self):
            """模拟训练一个epoch"""
            print(f"\nEpoch {self.epoch}:")
            
            # 打印参数状态
            for name, param in self.model.named_parameters():
                print(f"  {name:25s} requires_grad={param.requires_grad}")
            
            # 模拟参数更新
            if self.epoch >= 3:  # 假设从第3个epoch开始权重会有变化
                with torch.no_grad():
                    for param in self.model.parameters():
                        if param.requires_grad:
                            param.data += 0.01 * torch.randn_like(param.data)
    
    # 创建runner
    runner = MockRunner(model)
    
    # 创建Hook
    freeze_hook = SimpleFreezeHook(
        freeze_epochs=3,
        freeze_prefixes=['backbone', 'head']  # 冻结backbone和head
    )
    
    # 创建检查Hook
    check_hook = CheckFreezeHook(
        freeze_epochs=3,
        freeze_prefixes=['backbone', 'head'],
        check_interval=1
    )
    
    # 初始化Hook
    freeze_hook.before_run(runner)
    check_hook.before_run(runner)
    
    # 模拟训练过程
    for epoch in range(8):
        runner.epoch = epoch
        
        # 执行冻结Hook
        freeze_hook.before_train_epoch(runner)
        
        # 训练一个epoch
        runner.train_epoch()
        
        # 执行检查Hook
        check_hook.after_train_epoch(runner)
    
    print("\n" + "=" * 50)
    print("测试完成！")
    
    # 打印检查结果
    print("\n检查结果汇总:")
    for epoch_key, result in check_hook.check_results.items():
        print(f"\n{epoch_key}:")
        print(f"  有变化的参数: {len(result['changed'])}个")
        print(f"  无变化的参数: {len(result['unchanged'])}个")