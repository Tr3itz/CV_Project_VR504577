# PyTorch imports
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Stats imports
from sklearn.metrics import accuracy_score, top_k_accuracy_score
from zeus.monitor import ZeusMonitor

# Utils imports
import gc
import os
import random
import yaml
import logging
import numpy as np
import pandas as pd
from ptflops import get_model_complexity_info # get FLOPs info
from tqdm.notebook import tqdm
from timeit import default_timer as timer
from datetime import datetime

# My imports
from alternatives import ConvBNRelu, DepthwiseSeparableConv
from LocalDNAS import SuperNet, SuperRegularizer


def baseline_SuperNet(exp_seed: int, dataset: Dataset, identity: bool=False, verbose: bool=True):
    # Seeding for genearation
    torch.manual_seed(exp_seed)

    # Seed network
    seed_net = torch.load(f'./main_models/MobileNetV3_init_seed{exp_seed}.pt', weights_only=False)

    # Alternatives
    alt_dict = torch.nn.ModuleDict()

    # Alternative instantiations
    for module in seed_net.named_modules():
        if module[1].__class__.__name__ == 'SqueezeExcitation':
            alt_1 = ConvBNRelu(in_channels=module[1].fc1.in_channels, out_channels=module[1].fc2.out_channels, groups=24)
            alt_2 = ConvBNRelu(in_channels=module[1].fc1.in_channels, out_channels=module[1].fc2.out_channels, groups=2, point=True)
            alt_3 = DepthwiseSeparableConv(in_channels=module[1].fc1.in_channels, out_channels=module[1].fc2.out_channels, point_groups=2, bnorm=True)
            key = '/'.join(module[0].split('.'))
            alt_dict[key] = torch.nn.ModuleList([nn.Identity(), alt_1, alt_2, alt_3]) if identity else torch.nn.ModuleList([alt_1, alt_2, alt_3])

            if verbose:
                print(f'{key}:')
                print(f'\t- #Parameters SqueezeExcitation: {sum(p.numel() for p in module[1].parameters() if p.requires_grad)}')
                print(f'\t- #Parameters Standard ConvBNRelu (24 filters): {sum(p.numel() for p in alt_1.parameters() if p.requires_grad)}')
                print(f'\t- #Parameters Point-wise ConvBNRelu (2 filters): {sum(p.numel() for p in alt_2.parameters() if p.requires_grad)}')
                print(f'\t- #Parameters Depth-wise Separable ConvBN (2 filters): {sum(p.numel() for p in alt_3.parameters() if p.requires_grad)}\n')

    # SuperNet instantiation
    random_sample = dataset[torch.randint(high=len(dataset), size=(1,))][0]
    SuperNet_model = SuperNet(seed=seed_net, branches=alt_dict, input_shape=random_sample.shape)

    return SuperNet_model


def train_model(
        model: nn.Module,
        device: str,
        data_loader: DataLoader,
        optimizer: torch.optim,
        loss_fn: nn.Module,
        regularizer: SuperRegularizer=None,
        epochs: int=100,
        lr_scheduler: torch.optim.lr_scheduler=None,
        val_dataloader: DataLoader=None,
        batch_update: int=1000,
        seed: int=42,
        verbose: bool=True
):
    # Seeding
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Move the model to the target device
    if device != 'cpu': model = model.to(device)
    
    # Training stats and history
    total_time = 0
    avg_loss, avg_acc = 0, 0
    loss_h, acc_h = [], []

    # Validation history
    val_loss_h, val_acc_h = [], []

    for epoch in range(epochs):
        curr_lr = lr_scheduler.get_last_lr()[0] if lr_scheduler is not None else optimizer.state_dict()['param_groups'][-1]['lr']
        print(f'*********************\nEPOCH {epoch} on {device} - Current learning rate: {curr_lr:.4f}\n')

        epoch_start = timer()
        running_loss, running_acc = 0, 0

        # TRAINING 
        model.train()
        for batch, (image, label) in tqdm(enumerate(data_loader), unit='batch', total=len(data_loader)):
            
            # Move data to target device
            if device != 'cpu': image, label = image.to(device), label.to(device)

            # Clear gradients
            optimizer.zero_grad()

            # 1. Forward pass
            out_logits = model(image)
            out_probs = torch.softmax(out_logits, dim=1)
            out_preds = torch.argmax(out_probs, dim=1)

            # 2. Loss computation
            loss = loss_fn(out_logits, label)
            reg_loss = regularizer(model=model) if regularizer is not None else torch.zeros(1, requires_grad=True, device=loss.device).squeeze()
            total_loss = loss + reg_loss

            # 3. Backpropagation and optimization
            total_loss.backward()
            optimizer.step() 

            # Epoch stats update
            running_loss += total_loss.detach().item() # Prevent RAM saturation
            if device != 'cpu': label, out_preds = label.cpu(), out_preds.cpu()
            running_acc += accuracy_score(label, out_preds)

            # Keep track of performance
            if verbose:
                if batch % batch_update == batch_update - 1:
                    print(f'Computed {batch:4}/{len(data_loader)} batches - Avg Loss: {(running_loss/batch):.5f} | Avg Accuracy: {((running_acc/batch)*100):.2f}%')

            # Compute average loss and accracy on the whole dataloader
            if batch == len(data_loader) - 1:
                avg_loss = running_loss/len(data_loader)
                avg_acc = running_acc/len(data_loader)

            # Free memory
            del image, label, loss, out_logits, out_probs, out_preds
            torch.cuda.empty_cache()

        # Update learning rate
        if lr_scheduler is not None: lr_scheduler.step()

        # Stop timer and update training time
        epoch_end = timer()
        time_elapsed = epoch_end - epoch_start
        total_time += time_elapsed

        # Update history
        loss_h.append(avg_loss)
        acc_h.append(avg_acc) 

        if verbose:
            print(f'\nEnd of EPOCH {epoch} - Avg Loss: {(avg_loss):.5f} | Avg Accuracy: {(avg_acc*100):.2f}%')
            print(f'Training time: {time_elapsed:.3f} seconds.')
            if isinstance(model, SuperNet):
                print(f'\nDNAS parameters:')
                for path, weights in model.get_supermodules_weights().items():
                    print(f'\t- {path}: [{weights}]')

        # VALIDATION
        if val_dataloader is not None:
            print(f'\nVALIDATION...')
            val_avg_acc = 0
            val_avg_loss = 0

            model.eval()
            with torch.no_grad():
                for _, (image, label) in tqdm(enumerate(val_dataloader), unit='batch', total=len(val_dataloader)):
                    if device != 'cpu': image, label = image.to(device), label.to(device)

                    val_out_logits = model(image)
                    val_out_probs = torch.softmax(val_out_logits, dim=1)
                    val_out_preds = torch.argmax(val_out_probs, dim=1)

                    val_avg_loss += loss_fn(val_out_logits, label).detach().item()

                    if device != 'cpu': val_out_preds, label = val_out_preds.cpu(), label.cpu()
                    val_avg_acc += accuracy_score(label, val_out_preds)

                    del image, label, val_out_logits, val_out_preds, val_out_probs
                    torch.cuda.empty_cache()

            val_avg_loss /= len(val_dataloader)
            val_loss_h.append(val_avg_loss)

            val_avg_acc /= len(val_dataloader)
            val_acc_h.append(val_avg_acc)
            
            print(f'VALIDATION COMPLETED - Avg Accuracy: {(val_avg_acc*100):.2f}%')

        # Free memory
        torch.cuda.empty_cache()
        gc.collect()

    # Move the model back to the cpu
    if device != 'cpu': model = model.cpu()
    torch.cuda.empty_cache()
    
    total_time_dict = seconds_to_hours(int(total_time))

    print(f'TRAINING COMPLETED - Avg Loss: {(avg_loss):.5f} | Avg Accuracy: {(avg_acc*100):.2f}%')
    print(f"Total trianing time: {total_time_dict['hours']} hours {total_time_dict['minutes']} minutes {total_time_dict['seconds']} seconds.")

    return loss_h, acc_h, val_loss_h, val_acc_h


def eval_model(
        model: nn.Module,
        device: str,
        data_loader: DataLoader,
        seed: int=42
):
    # Seeding
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Move the model to the target device
    model = model.to(device) if device != 'cpu' else model
    
    avg_acc = 0

    # TESTING
    model.eval()
    with torch.no_grad():
        for _, (image, label) in tqdm(enumerate(data_loader), unit='batch', total=len(data_loader)):
            image = image.to(device) if device != 'cpu' else image

            # Inference
            out_logits = model(image)
            out_probs = torch.softmax(out_logits, dim=1)
            out_preds = torch.argmax(out_probs, dim=1)

            # Performance metrics
            out_preds = out_preds.cpu() if device != 'cpu' else out_preds
            avg_acc += accuracy_score(label, out_preds)

            # Free memory
            del image, out_logits, out_preds, out_probs
            torch.cuda.empty_cache()

    # Update stats
    avg_acc /= len(data_loader)
    print(f'EVALUATION COMPLETED - Avg Accuracy: {(avg_acc*100):.2f}%') 

    # Move the model back to the cpu
    model = model.cpu() if device != 'cpu' else model
    
    # Free memory
    torch.cuda.empty_cache()
    gc.collect()

    return avg_acc


def eval_model_comparison(
        model: nn.Module,
        device: str,
        data_loader: DataLoader,
        warmup: int=6,
        seed: int=42
):
    # Seeding
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Store the current log level to disable Zeus logging
    original_log_level = logging.getLogger().getEffectiveLevel()
    logging.disable(logging.CRITICAL)

    # Move the model to the target device
    model = model.to(device) if device != 'cpu' else model
    
    avg_acc, avg_top5 = 0, 0
    inference_times, inference_engy = [], []

    # Warm-up for correct inference time measurements
    print('WARM UP...')
    model.eval()
    with torch.no_grad():
        for _ in tqdm(range(warmup), total=warmup):
            for _, (image, label) in enumerate(data_loader):
                image = image.to(device) if device != 'cpu' else image
                model(image)
                torch.cuda.empty_cache()

    # TESTING
    print('\nEVALUATION...')
    model.eval()
    with torch.no_grad():
        # Create the monitor for time and consumption
        monitor = ZeusMonitor(gpu_indices=[torch.cuda.current_device()], approx_instant_energy=True)

        for _, (image, label) in tqdm(enumerate(data_loader), unit='batch', total=len(data_loader)):
            image = image.to(device) if device != 'cpu' else image

            monitor.begin_window('inf', sync_execution=True)
            # Inference
            out_logits = model(image)
            out_probs = torch.softmax(out_logits, dim=1)
            out_preds = torch.argmax(out_probs, dim=1)

            # Compute inference time
            torch.cuda.synchronize()
            end_inf = monitor.end_window('inf', sync_execution=True)
            inference_times.append(end_inf.time)
            inference_engy.append(end_inf.total_energy)

            # Performance metrics
            out_probs = out_probs.cpu() if device != 'cpu' else out_probs
            out_preds = out_preds.cpu() if device != 'cpu' else out_preds
            avg_acc += accuracy_score(label, out_preds)
            avg_top5 += top_k_accuracy_score(label, out_probs, k=5, labels=np.arange(out_probs.shape[-1]))

            # Free memory
            del image, out_logits, out_preds, out_probs
            torch.cuda.empty_cache()

    results = {}

    # Update accuracy stats
    results['acc'] = avg_acc / len(data_loader)
    results['acc@5'] = avg_top5 / len(data_loader)
    
    # Update inference time stats
    results['max_t'] = max(inference_times)
    results['min_t'] = min(inference_times)
    results['avg_t'] = sum(inference_times) / len(inference_times)
    
    # Update energy consumption stats
    results['max_p'] = max(inference_engy)
    results['min_p'] = min(inference_engy)
    results['avg_p'] = sum(inference_engy) / len(inference_engy)
    
    print(f'EVALUATION COMPLETED:')
    print(f"\t- Avg Accuracy: {(results['acc']*100):.2f}% | Avg Acc@5: {(results['acc@5']*100):.2f}%") 
    print(f"\t- Inference times (ms) - Max: {(results['max_t']*1000):.2f} | Min: {(results['min_t']*1000):.2f} | Avg: {(results['avg_t']*1000):.2f}")
    print(f"\t- Energy consumption (J) - Max: {results['max_p']:.5f} | Min: {results['min_p']:.5f} | Avg: {results['avg_p']:.5f}")

    # Move the model back to the cpu
    model = model.cpu() if device != 'cpu' else model
    
    # Free memory
    torch.cuda.empty_cache()
    gc.collect()

    # Restore the original log level after the tests
    logging.disable(original_log_level)

    return results


def save_experiment(root: str, model: nn.Module, seed: int, settings: dict, metrics_df: pd.DataFrame=None, exported: bool=False):
    
    # Create experiment directory
    now = datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
    
    if not exported:
        exp_dir = f'{root}/Baseline_{now}'
    else:
        exp_dir = f'{root}/regularizers/SuperNet_{now}' if 'regularizer' in settings.keys() else f'{root}/soundness/SuperNet_{now}'
    
    os.mkdir(f'{exp_dir}')

    # Save configurations
    with open(f'{exp_dir}/config.yaml', 'w')as f:
        yaml.dump(settings, f)

    # Save history of metrics
    if metrics_df is not None: metrics_df.to_csv(f'{exp_dir}/metrics.csv')

    # Create models directory
    models_dir = f'{exp_dir}/models'
    os.mkdir(f'{models_dir}')

    # Save the model's file and state_dict
    model_name = model.__class__.__name__
    model_name = f'{model_name}_exported' if exported else model_name
    torch.save(model.state_dict(), f'{models_dir}/{model_name}_weights_seed{seed}.pt')
    torch.save(model, f'{models_dir}/{model_name}_seed{seed}.pt')

    # Save a torch.jit script for inference without instantiating the model's class 
    model_scripted = torch.jit.script(model)
    model_scripted.save(f'{models_dir}/{model_name}_script_seed{seed}.pt')


def check_gradients(model: SuperNet, primar_loss_fn: nn.Module, reg_loss_fn: SuperRegularizer, dataset: Dataset):
    img, label = dataset[torch.randint(high=len(dataset), size=(1,))]
    img = img.unsqueeze(dim=0)
    label = torch.tensor([label])

    # Feed the img forward and compute the loss
    out_logits = model(img)
    primar_loss = primar_loss_fn(out_logits, label)

    # Compute the regularization loss
    stable_loss, target_loss = reg_loss_fn.split_costs(model)

    # Primar loss gradient
    gradients_primar = torch.autograd.grad(primar_loss, model.parameters(), retain_graph=True)
    grad_norm_primar = torch.stack([torch.norm(grad)**2 for grad in gradients_primar if grad is not None])
    grad_norm_primar = torch.sqrt(torch.sum(grad_norm_primar))
    print(f'Primar Loss ({primar_loss_fn.__class__,__name__}):\n- Loss:\t\t{primar_loss}\n- Gradient:\t{grad_norm_primar}')

    # SWR loss gradient
    if stable_loss.item() > 0.0:
        gradients_swr = torch.autograd.grad(stable_loss, model.parameters(), allow_unused=True, retain_graph=True)
        grad_norm_swr = torch.stack([torch.norm(grad)**2 for grad in gradients_swr if grad is not None])
        grad_norm_swr = torch.sqrt(torch.sum(grad_norm_swr))
        print(f'SWR Loss (Metric: {reg_loss_fn.type_cost}):\n- Loss:\t\t{stable_loss}\n- Gradient:\t{grad_norm_swr}')

    # Soft-Constraint loss gradient
    if target_loss.item() > 0.0:
        gradients_cos = torch.autograd.grad(target_loss, model.parameters(), allow_unused=True, retain_graph=True)
        grad_norm_cos = torch.stack([torch.norm(grad)**2 for grad in gradients_cos if grad is not None])
        grad_norm_cos = torch.sqrt(torch.sum(grad_norm_cos))
        print(f'Soft-Constraint Loss (Metric: {reg_loss_fn.type_cost} | Target: {reg_loss_fn.target}):\n- Loss:\t\t{target_loss}\n- Gradient:\t{grad_norm_cos}')

    # Clear gradients
    model.zero_grad()


def model_size(model: nn.Module):
    # Estimate the size of model in MB
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    
    return size_all_mb

def count_parameters(model: nn.Module):
    # Count parameters of model
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_flops(model: nn.Module, dataset: Dataset):
    # Count FLOPs of model 
    x = dataset[torch.randint(high=len(dataset), size=(1,))][0]
    flops = get_model_complexity_info(model=model, input_res=tuple(x.squeeze().size()), print_per_layer_stat=False, as_strings=False, backend='pytorch')[0]

    # Remove FLOPs counting attrs
    delattr(model, 'start_flops_count')
    delattr(model, 'stop_flops_count')
    delattr(model, 'reset_flops_count')
    delattr(model, 'compute_average_flops_cost')

    return flops


def count_submodule_flops(model: nn.Module, sm_name: str, dataset: Dataset):
    for _, module in model.named_modules():
        if module.__class__.__name__ == sm_name: module.register_forward_hook(_hook_flops_fn)

    x = dataset[torch.randint(high=len(dataset), size=(1,))][0]
    model(x.unsqueeze(0))

    total_sm_flops = 0
    for path, module in model.named_modules():
        if module.__class__.__name__ == sm_name:
            y = torch.rand(module.input_shape)
            sm_flops=get_model_complexity_info(model=module, input_res=tuple(y.squeeze().size()), print_per_layer_stat=False, as_strings=False, backend='pytorch')[0]
            print(f'#FLOPs {sm_name} @ {path}: {sm_flops}') 
            total_sm_flops += sm_flops
            delattr(module, 'input_shape')

    return total_sm_flops


def seconds_to_hours(seconds: int) -> dict[str, int]:
    res = dict()

    res['hours'] = seconds // 3600
    seconds = seconds % 3600

    res['minutes'] = seconds // 60
    seconds = seconds % 60

    res['seconds'] = seconds

    return res


def register_hook_flops_fn(model: torch.nn.Module, block_name: str):
    for module in model.modules():
        if module.__class__.__name__ == block_name:
            module.register_forward_hook(_hook_flops_fn)


def print_module_shapes(model: torch.nn.Module, input_shape: torch.Size):
    rand_tensor = torch.unsqueeze(torch.rand(input_shape), dim=0)
    model(rand_tensor)

    
def _hook_flops_fn(module: nn.Module, input: tuple, output: torch.Tensor):
    setattr(module, 'input_shape', input[0].shape)

