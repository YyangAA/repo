import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

class nnUNetTrainer_FreezeEncoder(nnUNetTrainer):
    # 严格匹配你提供的基类参数：plans, configuration, fold, dataset_json, device
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        # 调用父类，不多传也不少传
        super().__init__(plans, configuration, fold, dataset_json, device)
        
        # 微调配置
        self.initial_lr = 1e-3 
        self.num_epochs = 50 

    # def initialize(self):
    #     # 1. 正常的初始化，创建网络
    #     super().initialize()
        
    #     # 2. 锁定 Encoder 参数
    #     if self.network is not None:
    #         print("\n" + "="*20)
    #         print("=== 检测到微调任务：正在锁定 Encoder 参数 ===")
            
    #         # 检查 network 是否有 encoder
    #         if hasattr(self.network, 'encoder'):
    #             for param in self.network.encoder.parameters():
    #                 param.requires_grad = False
                
    #             # 统计结果确认
    #             encoder_trainable_params = sum(p.numel() for p in self.network.encoder.parameters() if p.requires_grad)
    #             print(f"Encoder 可训练参数数量: {encoder_trainable_params} (应为 0)")
    #             print("=== Encoder 锁定成功，仅更新 Decoder 和 Segmentation Heads ===")
    #         else:
    #             print("=== 警告：在当前网络结构中未找到 .encoder 属性 ===")
    #         print("="*20 + "\n")
    #     else:
    #         print("=== 错误：网络尚未初始化 ===")
    def initialize(self):
        # 1. 正常的初始化，创建网络
        super().initialize()
        
        # 2. 锁定 Encoder 参数
        if self.network is not None:
            print("\n" + "="*20)
            print("=== 检测到微调任务：正在部分锁定 Encoder 参数 ===")
            
            if hasattr(self.network, 'encoder'):
                # 获取 encoder 的 stages 数量
                num_stages = len(self.network.encoder.stages)
                
                # 遍历所有 stage
                for i in range(num_stages):
                    if i < num_stages - 3:
                        # 锁定前面的层
                        for param in self.network.encoder.stages[i].parameters():
                            param.requires_grad = False
                        print(f"Stage {i}: 已锁定")
                    else:
                        # 放开最后一层 (Bottleneck 层)
                        for param in self.network.encoder.stages[i].parameters():
                            param.requires_grad = True
                        print(f"Stage {i}: 已激活 (最后一层)")
                
                # 统计结果确认：此时不应为 0
                encoder_trainable_params = sum(p.numel() for p in self.network.encoder.parameters() if p.requires_grad)
                print(f"Encoder 剩余可训练参数数量: {encoder_trainable_params}")
                print("=== Encoder 部分锁定成功，现在最后一层参与微调 ===")
            else:
                print("=== 警告：在当前网络结构中未找到 .encoder 属性 ===")
            print("="*20 + "\n")
        else:
            print("=== 错误：网络尚未初始化 ===")

    def on_train_start(self):
        super().on_train_start()
        # 再次确认打印，方便在控制台日志中查看
        print(">>> 迁移学习启动：使用预训练权重微调 5.0T 数据 <<<")