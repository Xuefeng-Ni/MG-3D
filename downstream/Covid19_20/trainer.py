"""Jia-Xin ZHUANG
Modified on Feb 8, 2024.
"""

import os
import sys
import time
import math
import csv

import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data.distributed
from tensorboardX import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
import torch.distributed as dist
from monai.data import decollate_batch
import nibabel as nib
from monai.transforms import VoteEnsemble

from utils.misc import calculate_time, print_with_timestamp, MetricLogger, SmoothedValue, distributed_all_gather


def train_epoch(model, loader, optimizer, scaler, epoch, loss_func, args):
    '''Training
    '''
    model.train(True)
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = f'Epoch: [{epoch}/{args.max_epochs}]'

    for idx, batch_data in enumerate(metric_logger.log_every(loader, args.print_freq, header)):
        data, target = batch_data["image"].cuda(), batch_data["label"].cuda()
        with autocast(enabled=args.amp):
            logits = model(data)
            loss = loss_func(logits, target)

        if args.amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # accumate for some iterations and update
        loss /= args.accum_iter
        if (idx+1) % args.accum_iter == 0:
            if args.amp:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_value)
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            for param in model.parameters():
                param.grad = None

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print_with_timestamp(f"Loss is {loss_value}, stopping training, stopped at Epoch {epoch}/{idx}")
            sys.exit(-1)

        if args.distributed:
            loss_list = distributed_all_gather([loss], out_numpy=True)
            loss_value = np.mean(np.mean(np.stack(loss_list, axis=0), axis=0), axis=0)
        metric_logger.update(train_loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

    for param in model.parameters():
        param.grad = None

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print_with_timestamp("Averaged stats:", metric_logger)
    return metric_logger.meters['train_loss'].global_avg


def val_epoch(model, loader, acc_func,
              args, epoch, model_inferer=None, post_label=None, post_pred=None):
    '''Validation for segmentation.'''
    model.eval()

    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('acc', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = f'=>Val Epoch: [{epoch}]'

    accs = []
    acc_func.reset()
    with torch.no_grad():
        for idx, batch_data in enumerate(metric_logger.log_every(loader, args.print_freq, header)):
            data, target = batch_data["image"].cuda(), batch_data["label"].cuda()
            with autocast(enabled=args.amp):
                if model_inferer is not None:
                    logits = model_inferer(data)
                else:
                    logits = model(data)

            if post_label:
                val_labels_list = decollate_batch(target)
                val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list]
            else:
                val_labels_convert = target
            if post_pred:
                val_outputs_list = decollate_batch(logits)
                val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]
            else:
                val_output_convert = logits
            acc = acc_func(y_pred=val_output_convert, y=val_labels_convert)

            if args.distributed:
                acc = acc.cuda()
                acc_list = distributed_all_gather([acc], out_numpy=True)    #, is_valid=idx < loader.sampler.valid_length
                accs.extend(acc_list)
            else:
                acc_list = acc.detach().cpu().numpy()
                if not isinstance(acc_list, list):
                    acc_list = [acc_list]
                accs.extend(acc_list)

            avg_acc = np.mean([np.nanmean(l) for l in accs]) * 100
            metric_logger.update(acc=avg_acc)

    acc_func.reset()
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print_with_timestamp(f"{args.model_name} on fold {args.fold} of the {args.dataset_name}. Averaged stats: {metric_logger}.")
    return avg_acc


@calculate_time
def run_training(model, train_loader, val_loader, optimizer, loss_func, acc_func, args, model_inferer=None, scheduler=None,
                 start_epoch=0, post_label=None, post_pred=None):
    '''Run training.'''
    if args.logdir and args.rank == 0:
        writer = SummaryWriter(log_dir=args.logdir)
        print("Writing Tensorboard logs to ", args.logdir)
    else:
        writer = None
    scaler = GradScaler() if args.amp else None

    val_acc_max = 0.0 if args.best_acc is None else args.best_acc
    hist_epoch_secs = 0.0
    for epoch in range(start_epoch, args.max_epochs):
        epoch_time = time.time()
        train_loss = train_epoch(
            model, train_loader, optimizer, scaler=scaler, epoch=epoch,
            loss_func=loss_func, args=args
        )
        # Update the history epoch time.
        if epoch == 0 or epoch == 1:
            # The first epoch is usually slower than the rest, so reset it.
            hist_epoch_secs = time.time() - epoch_time
        else:
            hist_epoch_secs = (time.time() - epoch_time) * 0.9 + hist_epoch_secs * 0.1

        if args.rank == 0:
            train_log = f'=> Remaining: {hist_epoch_secs*(args.max_epochs-epoch-1)/3600:.2f} h to finish training.'
            print_with_timestamp(train_log)
            if writer:
                lr = optimizer.param_groups[0]["lr"]
                writer.add_scalar("lr", lr, epoch)
                writer.add_scalar("train_loss", train_loss, epoch)

        # Validation.
        if (epoch + 1) % args.val_every == 0 or epoch == 0:
            if args.distributed:
                dist.barrier()
            print_with_timestamp('Start validation.')
            epoch_time = time.time()
            val_avg_acc = val_epoch(
                model,
                val_loader,
                acc_func=acc_func,
                model_inferer=model_inferer,
                epoch=epoch,
                args=args,
                post_label=post_label,
                post_pred=post_pred,
            )
            if args.distributed:
                dist.barrier()
            if args.rank == 0:
                # use a barrier to make sure training is done on all ranks
                if writer:
                    writer.add_scalar("val_acc", val_avg_acc, epoch)
                if val_avg_acc > val_acc_max:
                    print_with_timestamp(f'new best ({val_acc_max:.6f} --> {val_avg_acc:.6f}).')
                    val_acc_max = val_avg_acc
                    # Save best model checkpoint.
                    if args.logdir:
                        save_checkpoint(model, epoch, args, best_acc=val_acc_max,
                                        optimizer=optimizer, scheduler=scheduler,
                                        loss_scaler=scaler)
                # Save final model checkpoint.
                print_with_timestamp(f'Current acc: {val_avg_acc:.6f}. Best acc:{val_acc_max:.6f}.')
                save_checkpoint(model, epoch, args, best_acc=val_acc_max,
                                optimizer=optimizer, scheduler=scheduler,
                                loss_scaler=scaler, filename="model_final.pt")
        if scheduler:
            scheduler.step()

    print_with_timestamp(f"Training Finished !, Best Accuracy: {val_acc_max}")
    return val_acc_max


def save_checkpoint(model, epoch, args, filename="model.pt", best_acc=0,
                    optimizer=None, scheduler=None, loss_scaler=None):
    '''Save checkpoint.'''
    state_dict = model.state_dict()
    save_dict = {"epoch": epoch, "best_acc": best_acc, "state_dict": state_dict}
    if optimizer is not None:
        save_dict["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        save_dict["scheduler"] = scheduler.state_dict()
    if loss_scaler is not None:
        save_dict['loss_scaler'] = loss_scaler.state_dict()
    if args is not None:
        save_dict["args"] = args

    filename = os.path.join(args.logdir, filename)
    torch.save(save_dict, filename)
    print_with_timestamp("Saving checkpoint", filename)


@torch.no_grad()
def infer(model, loader, val_files,
          dice_metric, nsd_metric, hd_metric,
          args=None, model_inferer=None,
          post_label=None, post_pred=None, post_nsd=None):
    '''Inference for segmentation.
    '''
    if isinstance(model, list):
        for ml in model:
            ml.eval()
    else:
        model.eval()

    accs_bg, accs, nsd_values, hd_values = [], [], [], []
    accs_cls = []
    time_list = []

    with torch.no_grad():
        with open(args.pred_csv, 'w', newline='\n', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            header = ['data_path', 'tagret_path', 'prediction', 'DSCbg', 'DSC', 'NSD']
            writer.writerow(header)

            start_time = time.time()
            for idx, (batch_data, val_file) in enumerate(zip(loader, val_files)):
                data, target = batch_data["image"], batch_data["label"]

                try:
                    data_name = os.path.basename(val_file['image'])
                    label_name = os.path.basename(val_file['label'])
                    label_name = label_name.split('.')[0] + '_label.nii.gz'
                    pred_name = f'pred_{label_name}'

                    data_path = os.path.join(args.pred_dir, data_name)
                    label_path = os.path.join(args.pred_dir, label_name)
                    pred_path = os.path.join(args.pred_dir, pred_name)

                    assert data.meta['filename_or_obj'][0] == val_file['image']
                    assert target.meta['filename_or_obj'][0] == val_file['label']
                except Exception as e:
                    print_with_timestamp(f'Error: {e}')
                    data_name = idx

                data = data.to(torch.device(args.device))
                with autocast(enabled=args.amp):
                    if isinstance(model_inferer, list):
                        logits = []
                        for mi in model_inferer:
                            logit = mi(data)
                            logits.append(logit)
                        logits = torch.cat(logits, dim=0)
                        logits = VoteEnsemble()(logits).unsqueeze(dim=0)
                    else:
                        print_with_timestamp(f'Infering {idx+1}/{len(loader)}:{data_name}, {data.device}')
                        logits = model_inferer(data)
                        print_with_timestamp(f'Infering {idx+1}/{len(loader)}:{data_name}, {data.device} done.')

                # Calculate the elapsed time
                end_time = time.time()
                elapsed_time = end_time - start_time
                time_list.append(elapsed_time)

                val_labels_list = decollate_batch(target)
                if post_label:
                    val_labels_convert = [post_label(val_label_tensor).cpu() for val_label_tensor in val_labels_list]
                else:
                    val_labels_convert = [val_label_tensor.cpu() for val_label_tensor in val_labels_list]
                val_outputs_list = decollate_batch(logits)
                val_output_convert = [post_pred(val_pred_tensor).cpu() for val_pred_tensor in val_outputs_list]

                val_output_nsd = [post_nsd(val_pred_tensor).cpu().numpy().astype(bool) for val_pred_tensor in val_outputs_list]
                val_label_nsd = [val_label_tensor.cpu().numpy().astype(bool) for val_label_tensor in val_labels_list]

                if args.save_visualization:
                    print_with_timestamp(f'Saving visualization results. to {args.pred_dir}')
                    # save prediction for visualization
                    affine = target[0].meta['affine'].numpy()
                    nib.save(nib.Nifti1Image(data.cpu().numpy().squeeze(), affine), data_path)
                    lbl_output = np.argmax(val_labels_convert[0].numpy(), axis=0).astype(np.uint8)
                    nib.save(nib.Nifti1Image(lbl_output, affine), label_path)
                    val_output = np.argmax(val_output_convert[0].numpy(), axis=0).astype(np.uint8)
                    nib.save(nib.Nifti1Image(val_output, affine), pred_path)

                dice_metric.reset()
                nsd_metric.reset()
                # hd_metric.reset()

                # Dice
                acc = dice_metric(y_pred=val_output_convert, y=val_labels_convert)
                acc_bg = acc.nanmean().detach().cpu().numpy().squeeze() * 100
                acc_item = acc[:, 1:].nanmean().detach().cpu().numpy().squeeze() * 100
                acc_cls = acc[:, :].detach().cpu().numpy().squeeze() * 100
                accs_bg.append(acc_bg)
                accs.append(acc_item)
                accs_cls.append(acc_cls)
                print_with_timestamp(f'Evaluating {idx+1}/{len(loader)}:{data_name}. Acc_bg {acc_bg:.2f}, Acc {acc_item:.2f}')
                # NSD
                nsd_metric(y_pred=val_output_nsd[0][0], y=val_label_nsd[0][0])
                # import cupy as cp
                # nsd_metric(y_pred=cp.asarray(val_output_nsd[0][0]), y=cp.asarray(val_label_nsd[0][0]))
                nsd = nsd_metric.aggregate() * 100
                nsd_values.append(nsd)
                print_with_timestamp(f'Evaluating {idx+1}/{len(loader)}:{data_name}. Acc_bg {acc_bg:.2f}, Acc {acc_item:.2f}, NSD {nsd:.4f}')
                # HD
                # hd_metric(y_pred=val_output_nsd[0][0], y=val_label_nsd[0][0])
                # hd = hd_metric.aggregate()
                hd = 0
                hd_values.append(hd)

                print_with_timestamp(f'Evaluating {idx+1}/{len(loader)}:{data_name}. Acc_bg {acc_bg:.2f}, Acc {acc_item:.2f}, NSD {nsd}, HD {hd}, using time {elapsed_time:.2f}s')

                output_row = [val_file['image'], val_file['label'], pred_path, f'{acc_bg:.2f}', f'{acc_item:.2f}', f'{nsd:.4f}', f'{hd:.4f}']
                writer.writerow(output_row)

                # the elapsed time to include data loading and model processing.
                start_time = time.time()

        avg_acc_bg = np.mean([np.nanmean(l) for l in accs_bg])
        avg_acc = np.mean([np.nanmean(l) for l in accs])
        avg_acc_cls = np.nanmean(np.array(accs_cls), axis=0)
        avg_nsd = np.mean([np.nanmean(l) for l in nsd_values])
        avg_hd = np.mean([np.nanmean(l) for l in hd_values])
        print_with_timestamp(f'=>Overall statistic: [{args.fold}] Acc bg {avg_acc_bg:.4f}, Acc {avg_acc:.4f}, NSD {avg_nsd:.4f}, HD {avg_hd:.4f}, Avg time {np.mean(time_list):.2f}s with std {np.std(time_list):.2f}s per image')
        print_with_timestamp(f'Acc_cls {avg_acc_cls}')
        return accs_bg, accs