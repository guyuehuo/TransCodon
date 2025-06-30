"""PyTorch Lightning module for standard training."""

import math
import argparse
import os
import torch
import numpy as np
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from data_module import CodonDataModule
from checkpointing import PeriodicCheckpoint
from transcodon.sequence import CodonSequence
from transcodon.alphabet import Alphabet
from transcodon.model import ProteinBertModel
from pytorch_lightning.strategies import DDPStrategy
import time

class MultiClassFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha  # 可传入类别权重（与CrossEntropy的weight类似）

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(
            inputs, targets, reduction='none', ignore_index=-100
        )
        pt = torch.exp(-ce_loss)                     # 计算预测概率
        focal_loss = (1 - pt) ** self.gamma * ce_loss  # Focal Loss核心公式
        # if self.alpha is not None:
        #     alpha_weights = self.alpha[targets]       # 根据目标标签选择权重
        #     focal_loss = alpha_weights * focal_loss
        
        return focal_loss.mean()
    
class CodonModel(pl.LightningModule):
    """PyTorch Lightning module for standard training."""
    def __init__(self, args, alphabet):
        super().__init__()
        self.args = args
        self.alphabet = alphabet
        self.model = ProteinBertModel(args, alphabet)

        def init_weights(module):
            if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
                torch.nn.init.normal_(module.weight, std=.02)

            if isinstance(module, (torch.nn.Linear)):
                module.bias.data.fill_(0)
        self.model.apply(init_weights)

        self.loss_fn = nn.CrossEntropyLoss(reduction='mean', ignore_index=-100)
        self.loss_fn_2d = nn.CrossEntropyLoss(reduction='mean', ignore_index=-100)
        #self.existence_loss =nn.functional.binary_cross_entropy_with_logits
        #self.struct_loss_fn = nn.BCEWithLogitsLoss()
        self.save_hyperparameters()

    def forward(self, x, y):
        return self.model(x,y)
    
    #*********************************transcodon setting*********************************
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay)

        if self.args.lr_scheduler == 'none':
            return optimizer
        elif self.args.lr_scheduler == 'warmup_sqrt':
            def schedule(global_step):
                if global_step < self.args.warmup_steps:
                    return (global_step+1) / self.args.warmup_steps
                else:
                    return np.sqrt(self.args.warmup_steps / global_step)
        elif self.args.lr_scheduler == 'warmup_cosine':
            def schedule(global_step):
                if global_step < self.args.warmup_steps:
                    return (global_step+1) / self.args.warmup_steps
                else:
                    progress = (global_step - self.args.warmup_steps) / self.args.num_steps
                    return max(0., .5 * (1. + math.cos(math.pi * progress)))
        else:
            raise ValueError('Unrecognised learning rate scheduler')

        scheduler = {
            'scheduler': torch.optim.lr_scheduler.LambdaLR(optimizer, schedule),
            'name': 'learning_rate',
            'interval': 'step',
            'frequency': 1
        }
        return [optimizer], [scheduler]

    def training_step(self, train_batch, batch_idx):
        data,organism, labels ,struct_label = \
            train_batch['input'].to(), \
            train_batch['organism'].to(dtype=torch.int64),\
            train_batch['labels'].to(dtype=torch.int64),\
            train_batch['struct_label'].to(dtype=torch.int64)
        

        output = self.model(data,organism)
        likelihoods = output['logits']
        struct_label_output=output['struct_label_output']

        #print("organism size",organism.size())

        # 计算序列重建损失（例如交叉熵损失）
        loss = self.loss_fn(
            likelihoods.view(-1, len(self.alphabet.all_toks)),
            labels.view(-1)
        )
        self.log('train_mask_loss', loss)
        
        if (struct_label != -100).any(): 
            target = struct_label.view(-1)
            num_classes = struct_label_output.size(-1)
            #print("struct_label",struct_label)

            # if target.max() >= num_classes:
            #     print("Invalid target label found:", target.unique())
            #     raise ValueError("Label out of range for CrossEntropyLoss")

            loss_2d = self.loss_fn_2d(
                struct_label_output.view(-1,struct_label_output.size(2)),
                struct_label.view(-1)
            )
            self.log('train_2d_mask_loss', loss_2d)

            loss += 0.4*loss_2d
       
            self.log('train_total_loss', loss)

        return loss

    def validation_step(self, val_batch, batch_idx):
        data,organism, labels ,struct_label = \
            val_batch['input'].to(), \
            val_batch['organism'].to(dtype=torch.int64),\
            val_batch['labels'].to(dtype=torch.int64),\
            val_batch['struct_label'].to(dtype=torch.int64)
            
        
        output = self.model(data,organism)
        likelihoods = output['logits']
        struct_label_output=output['struct_label_output']
        loss = self.loss_fn(
            likelihoods.view(-1, len(self.alphabet.all_toks)),
            labels.view(-1)
        )
        self.log('val_mask_loss', loss)
        #print("struct_label_output.size(2)",struct_label_output.size(2))
        if (struct_label != -100).any(): 
            loss_2d = self.loss_fn_2d(
                struct_label_output.view(-1,struct_label_output.size(2)),
                struct_label.view(-1)
            )
            self.log('val_2d_mask_loss', loss_2d)
        #self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
if __name__ == '__main__':

    # parsing
    parser = argparse.ArgumentParser()
    # parser.add_argument('--max_positions', type=int, default=1024)
    #parser.add_argument('--max_positions', type=int, default=2048)
    #parser.add_argument('--max_positions', type=int, default=2560)
    parser.add_argument('--max_positions', type=int, default=2048)

    #parser.add_argument('--rope_embedding', type=bool, default=False)
    parser.add_argument('--batch_size', type=int, default=3)
    #parser.add_argument('--batch_size', type=int, default=6)
    #parser.add_argument('--accumulate_gradients', type=int, default=40)
    parser.add_argument('--accumulate_gradients', type=int, default=6)
    parser.add_argument('--mask_proportion', type=float, default=.25)
    parser.add_argument('--leave_percent', type=float, default=.1)
    parser.add_argument('--mask_percent', type=float, default=.8)
    parser.add_argument('--warmup_steps', type=int, default=1000)
    #parser.add_argument('--warmup_steps', type=int, default=1)
    parser.add_argument('--weight_decay', type=float, default=0.1)
    parser.add_argument('--lr_scheduler', type=str, default='warmup_cosine')
    #parser.add_argument('--lr_scheduler', type=str, default='custom')  # Custom scheduler
    # parser.add_argument('--learning_rate', type=float, default=5e-7)  
    parser.add_argument('--learning_rate', type=float, default=1e-4) 
    #parser.add_argument('--learning_rate', type=float, default=5e-5) 
    parser.add_argument('--max_epochs', type=int, default=15)
    # parser.add_argument('--num_steps', type=int, default=121000)
    parser.add_argument('--num_steps', type=int, default=121000)
    parser.add_argument('--num_gpus', type=int, default=2)



    #parser.add_argument('--rope_embedding', type=bool, default=False)
    
    ProteinBertModel.add_args(parser)
    args = parser.parse_args()
   
    # 然后计算 total_steps
    print("args.num_steps:",args.num_steps)
    # args.num_steps=int((2298185 * args.max_epochs) / (args.batch_size * args.num_gpus * args.accumulate_gradients))
    # args.num_steps=int((4027021 * args.max_epochs) / (args.batch_size * args.num_gpus * args.accumulate_gradients))
    # args.num_steps=int((10942365 * args.max_epochs) / (args.batch_size * args.num_gpus * args.accumulate_gradients))
    # args.num_steps=int((18377 * args.max_epochs) / (args.batch_size * args.num_gpus * args.accumulate_gradients))
    args.num_steps=int((3556 * args.max_epochs) / (args.batch_size * args.num_gpus * args.accumulate_gradients))
    #args.num_steps=int((27562 * args.max_epochs) / (args.batch_size * args.num_gpus * args.accumulate_gradients))
    args.warmup_steps=int(args.num_steps*0.1)
    print("args.num_steps:",args.num_steps)
    print("args.warmup_steps:",args.warmup_steps)
    

    # data
    alphabet = Alphabet.from_architecture('CodonModel')
    print("len(alphabet):",len(alphabet))
    # datamodule = CodonDataModule(args, alphabet,
    #     'data/heldout/ecoli.fasta', args.batch_size)
    #print("output_part4")
    # datamodule = CodonDataModule(args, alphabet,
    #     'data/training_data/output_part4.fasta', args.batch_size)
    test=False
    if test:
        datamodule = CodonDataModule(args, alphabet,
            'test2.csv', args.batch_size)
        #exit()
    else:
        datamodule = CodonDataModule(args, alphabet,
           'top_dataset1_with_cai.csv', args.batch_size)
    #exit()

    # model
   

    # training

    name='transcodon-fintune-epoch15'

    time_begin=time.time()
    print('time begin:',time_begin)

    #name = 'Codon-AA'
    #!!!!!!!!!!!!!!!!!!!!!注意这里注释掉了wandb
    # wandb.init(mode="offline")
    # logger = WandbLogger(name=name, project='data1', version='version1-18')
    logger = TensorBoardLogger("board_logs", name=name) 

    #!!!!!!!!!!!!!!!!!!!!!注意这里注释掉了wandb
    # 加载 checkpoint，只加载权重
    model = CodonModel(args, alphabet)
    ckpt_path = './pretraining/epoch=2-step=225258.ckpt'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dict = torch.load(ckpt_path, map_location=device)["state_dict"]
    model.load_state_dict(state_dict, strict=False)  # 如果有不匹配项可以设置 strict=False


    trainer = pl.Trainer(num_nodes=1, precision='16-mixed',
    max_steps=args.num_steps,
    max_epochs=args.max_epochs,
    logger=logger,
    log_every_n_steps=1,
    #val_check_interval=1,
    val_check_interval=50*args.accumulate_gradients,          #!!!!!!tag ,should be 500
    accumulate_grad_batches=args.accumulate_gradients,
    limit_val_batches=1, accelerator='gpu',strategy=DDPStrategy(find_unused_parameters=True), #!!!!!!tag ,should be 0.25
    #limit_val_batches=1.0, accelerator='gpu',strategy='ddp', 
    callbacks=[PeriodicCheckpoint(1000, name),
        LearningRateMonitor(logging_interval='step')])

    trainer.fit(model, datamodule=datamodule)
    

    
    time_end=time.time()
    print('time cost:',(time_end-time_begin)/3600,'h')
    
