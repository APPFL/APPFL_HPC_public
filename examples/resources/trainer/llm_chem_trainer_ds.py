import copy
import time
import torch
import torch.nn as nn
import math
import wandb
import importlib
import numpy as np
from tqdm import tqdm
from torch.nn import Module
from omegaconf import DictConfig
from typing import Tuple, Dict, Optional, Any
from torch.utils.data import Dataset, DataLoader
from appfl.privacy import laplace_mechanism_output_perturb
from appfl.algorithm.trainer.base_trainer import BaseTrainer
from appfl.misc.utils import parse_device_str, apply_model_device


from appfl.algorithm.trainer.vanilla_trainer import VanillaTrainer

class llm_chem_trainer_ds(VanillaTrainer):
    def __init__(
        self,
        model: Optional[Module] = None,
        loss_fn: Optional[Module] = None,
        metric: Optional[Any] = None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        train_configs: DictConfig = DictConfig({}),
        logger: Optional[Any] = None,
        **kwargs,
    ):
        super().__init__(
            model=model,
            loss_fn=loss_fn,
            metric=metric,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            train_configs=train_configs,
            logger=logger,
            **kwargs,
        )
        if not hasattr(self.train_configs, "device"):
            self.train_configs.device = "cpu"

        self.val_dataloader = (
            DataLoader(
                self.val_dataset,
                batch_size=self.train_configs.get("val_batch_size", 32),
                shuffle=self.train_configs.get("val_data_shuffle", False),
                num_workers=self.train_configs.get("num_workers", 0),
            )
            if self.val_dataset is not None
            else None
        )
        if (
            hasattr(self.train_configs, "enable_wandb")
            and self.train_configs.enable_wandb
        ):
            self.enabled_wandb = True
            self.wandb_logging_id = self.train_configs.wandb_logging_id
        else:
            self.enabled_wandb = False
        self._sanity_check()

    def train(self, **kwargs):
        """
        Train the model for a certain number of local epochs or steps and store the mode state
        (probably with perturbation for differential privacy) in `self.model_state`.
        """
        # Distribute samples across GPUs
        samples_per_gpu = len(self.train_dataset) // self.world_size
        start_idx = self.global_rank * samples_per_gpu
        end_idx = start_idx + samples_per_gpu if self.global_rank < self.world_size - 1 else len(self.train_dataset)
        
        local_train_data = self.train_dataset[start_idx:end_idx]
        
        print(f"Rank {self.global_rank}: Has {len(local_train_data)} samples")

        def collate_fn(batch_items):
            max_length = max(len(item[0]) for item in batch_items)
            
            # Pad to multiple of 8
            max_length = ((max_length + 8 - 1) // 8) * 8
            
            batch = {'input_ids': [], 'attention_mask': [], 'labels': []}
            
            for input_ids, attention_mask, labels in batch_items:
                padding_length = max_length - len(input_ids)
                
                # Pad input_ids
                padded_input_ids = torch.cat([input_ids, torch.full((padding_length,), 0, dtype=input_ids.dtype)])
                batch['input_ids'].append(padded_input_ids)
                
                padded_attention_mask = torch.cat([attention_mask, torch.zeros(padding_length, dtype=attention_mask.dtype)])
                batch['attention_mask'].append(padded_attention_mask)
                
                padded_labels = torch.cat([labels, torch.full((padding_length,), -100, dtype=labels.dtype)])
                batch['labels'].append(padded_labels)
            
            return {
                'input_ids': torch.stack(batch['input_ids']),
                'attention_mask': torch.stack(batch['attention_mask']),
                'labels': torch.stack(batch['labels'])
            }
        
        # Create batches
        self.train_dataloader = []
        batch_size=self.train_configs.get("train_batch_size", 16)
        
        for i in range(0, len(local_train_data), batch_size):
            # if i + batch_size <= len(local_train_data):
            batch_data = local_train_data[i:i+batch_size]
            batch = collate_fn(batch_data)
            self.train_dataloader.append(batch)
    
        device = self.model_engine.device
        self.model_engine.train()
        self.total_pre_val_time = "n/a"
        self.total_val_time = "n/a"
        self.total_forward_time = 0.0
        self.total_backward_time = 0.0

        train_start_time = time.time()
        # Start training
        if self.train_configs.mode == "epoch":
            self.avg_epoch_loss_dict = {} 
            for epoch in range(self.train_configs.num_local_epochs):
                epoch_loss = 0
                for step, batch in enumerate(self.train_dataloader):
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)

                    forward_start_time = time.time()
                    outputs = self.model_engine(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=False
                    )
                    
                    logits = outputs.logits

                    # Shift for causal LM: predict next token - same as original
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    
                    # Flatten for loss computation - same as original
                    shift_logits = shift_logits.view(-1, shift_logits.size(-1))
                    shift_labels = shift_labels.view(-1)

                    loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                    loss = loss_fct(shift_logits, shift_labels)
                    self.total_forward_time += time.time() - forward_start_time

                    backward_start_time = time.time()
                    self.model_engine.backward(loss)
                    self.total_backward_time += time.time() - backward_start_time
                    
                    self.model_engine.step()

                    if self.global_rank == 0:
                        epoch_loss += loss.item()
                        print(f"step loss is {loss.item():.4f}")
                
                if self.global_rank == 0:
                    avg_epoch_loss = epoch_loss / len(self.train_dataloader)
                    print(f"Average epoch loss is {avg_epoch_loss}")

                    self.avg_epoch_loss_dict[epoch] = avg_epoch_loss

            # run evaluation or save the local model here
        elif self.train_configs.mode == "step":
            total_loss = 0

            if self.global_rank == 0:
                progress_bar = tqdm(range(self.train_configs.num_local_steps), desc="Training")

            data_iter = iter(self.train_dataloader)
            for _ in range(self.train_configs.num_local_steps):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(self.train_dataloader)
                    batch = next(data_iter)

                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                forward_start_time = time.time()
                outputs = self.model_engine(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False
                )
                
                logits = outputs.logits

                # Shift for causal LM
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                
                # Flatten for loss computation
                shift_logits = shift_logits.view(-1, shift_logits.size(-1))
                shift_labels = shift_labels.view(-1)

                loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                loss = loss_fct(shift_logits, shift_labels)
                self.total_forward_time += time.time() - forward_start_time

                backward_start_time = time.time()
                self.model_engine.backward(loss)
                self.total_backward_time += time.time() - backward_start_time
                
                self.model_engine.step()

                if self.global_rank == 0:
                    total_loss += loss.item()
                    progress_bar.update(1)
                    print(f"step loss is {loss.item():.4f}")
            
            if self.global_rank == 0:
                self.avg_step_loss = total_loss / self.train_configs.num_local_steps
                print(f"Average step loss: {self.avg_step_loss} with training time {time.time() - train_start_time:.2f} seconds.")