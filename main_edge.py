import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from model.edge_model import EdgeXResNet1d
from itertools import cycle
import pandas as pd
from Dataset import EdgeLabeledDataset, EdgeUnlabeledDataset
from dataset.ecg_transform import PreprocessCfg, ecg_transform_v2
import time
import json
from sklearn.metrics import accuracy_score, roc_auc_score

def freeze_edge_layers(edge_model, freeze_layers=('stem', 'block1', 'block2')):
    """
    冻结底层参数（requires_grad=False），高层参数保持可训练
    """
    for name, param in edge_model.named_parameters():
        if any(name.startswith(layer) for layer in freeze_layers):
            param.requires_grad = False
        else:
            param.requires_grad = True
    print(f"已冻结层: {freeze_layers}")
    


def compute_per_class_auc(gt, pred):
    auc_dict = {}
    for i in range(gt.shape[1]):
        if len(np.unique(gt[:, i])) < 2:
            auc = float('nan')
        else:
            auc = roc_auc_score(gt[:, i], pred[:, i])
        auc_dict[f"class_{i}"] = auc
    mean_auc = np.nanmean(list(auc_dict.values()))
    return mean_auc, auc_dict

def compute_per_class_accuracy(gt, pred):
    acc_dict = {}
    pred_binary = (pred > 0.5).astype(int)
    for i in range(gt.shape[1]):
        acc = accuracy_score(gt[:, i], pred_binary[:, i])
        acc_dict[f"class_{i}"] = acc
    mean_acc = np.mean(list(acc_dict.values()))
    return mean_acc, acc_dict

def log_json_stats(log_path, stats_dict):
    with open(log_path, "a") as f:
        f.write(json.dumps(stats_dict, ensure_ascii=False) + "\n")

def train(model, labeled_loader, unlabeled_loader, optimizer,
          criterion_sup, criterion_distill, alpha, device, epoch_idx, max_epoch):
    model.train()
    n_batches = len(labeled_loader)
    unlabeled_iter = cycle(unlabeled_loader)
    total_loss, total_sup, total_distill = 0, 0, 0
    start_time = time.time()
    for i, (x_l, y_l) in enumerate(labeled_loader):
        x_u, soft_label_u = next(unlabeled_iter)
        x_l, y_l = x_l.to(device), y_l.to(device)
        x_u, soft_label_u = x_u.to(device), soft_label_u.to(device)
        optimizer.zero_grad()
        logits_l = model(x_l)
        loss_sup = criterion_sup(logits_l, y_l)
        logits_u = model(x_u)
        loss_distill = criterion_distill(logits_u, soft_label_u)
        loss = loss_sup + alpha * loss_distill
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_sup += loss_sup.item()
        total_distill += loss_distill.item()
        if (i+1) % 50 == 0 or (i+1) == n_batches:
            elapsed = time.time() - start_time
            percent = (i+1) / n_batches
            eta = elapsed / percent - elapsed
            lr = optimizer.param_groups[0]['lr']
            print(
                f"Train Epoch: [{epoch_idx+1}/{max_epoch}] "
                f"[{i+1:5d}/{n_batches}] "
                f"eta: {int(eta//60):02d}:{int(eta%60):02d} "
                f"lr: {lr:.6f} "
                f"loss: {loss.item():.6f} "
                f"avg_loss: {total_loss/(i+1):.6f} "
                f"avg_sup: {total_sup/(i+1):.6f} "
                f"avg_distill: {total_distill/(i+1):.6f}"
            )
    avg_loss = total_loss / n_batches
    avg_sup = total_sup / n_batches
    avg_distill = total_distill / n_batches
    print(f"Train: Avg Loss {avg_loss:.4f}, Supervised {avg_sup:.4f}, Distill {avg_distill:.4f}")
    return avg_loss, avg_sup, avg_distill

def valid_on_ptb(model, val_loader, criterion, device, return_preds=False):
    model.eval()
    gt = []
    pred = []
    val_losses = []
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            val_loss = criterion(logits, y)
            val_losses.append(val_loss.item())
            probs = torch.sigmoid(logits)
            pred.append(probs.cpu().numpy())
            gt.append(y.cpu().numpy())
    gt = np.concatenate(gt, axis=0)
    pred = np.concatenate(pred, axis=0)
    auc = safe_roc_auc_score(gt, pred)
    metrics = {"Mean ROC AUC": auc}
    if return_preds:
        return np.mean(val_losses), auc, metrics, gt, pred
    else:
        return np.mean(val_losses), auc, metrics

def safe_roc_auc_score(y_true, y_pred):
    scores = []
    for i in range(y_true.shape[1]):
        if len(np.unique(y_true[:, i])) < 2:
            scores.append(np.nan)
        else:
            scores.append(roc_auc_score(y_true[:, i], y_pred[:, i]))
    valid_scores = [s for s in scores if not np.isnan(s)]
    if len(valid_scores) > 0:
        return np.mean(valid_scores)
    else:
        return np.nan

def main(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EdgeXResNet1d(
        num_classes=config['num_classes'],
        pool_type='avg'
    ).to(device)


    # 标签数据加载
    df_train = pd.read_csv(config['labeled_csv'])
    train_signal = df_train['path'].tolist()
    train_labels = df_train['label']
    df_val = pd.read_csv(config['val_csv'])
    val_signal = df_val['path'].tolist()
    val_labels = df_val['label']
    df_test = pd.read_csv(config['test_csv'])
    test_signal = df_test['path'].tolist()
    test_labels = df_test['label']

    preprocess_cfg = PreprocessCfg(seq_length=5000, duration=10, sampling_rate=500)
    train_transform = ecg_transform_v2(cfg=preprocess_cfg, is_train=True)
    val_transform = ecg_transform_v2(cfg=preprocess_cfg, is_train=False)
    test_transform = ecg_transform_v2(cfg=preprocess_cfg, is_train=False)

    train_labeled_dataset = EdgeLabeledDataset(train_signal, train_labels, transforms=train_transform)
    val_dataset = EdgeLabeledDataset(val_signal, val_labels, transforms=val_transform)
    test_dataset = EdgeLabeledDataset(test_signal, test_labels, transforms=test_transform)

    train_labeled_loader = DataLoader(train_labeled_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

    import re
    def extract_num(file):
        nums = re.findall(r'(\d+)', os.path.basename(file))
        return int(nums[-1]) if nums else -1

    data_dir = config.get('data_dir', './')
    update_csv_list = glob.glob(os.path.join(data_dir, 'update_*.csv'))
    update_npy_list = glob.glob(os.path.join(data_dir, 'ecg_softlabel_save_update_*.npy'))

    update_csv_list = sorted(update_csv_list, key=extract_num)
    update_npy_list = sorted(update_npy_list, key=extract_num)

    best_test_auc = 0.0
    checkpoint_path = config['best_ckpt']
    global_best_auc = 0.0
    global_best_ckpt = os.path.join(os.path.dirname(config['best_ckpt']), "global_best.pt")

    log_path = os.path.join(os.path.dirname(config['best_ckpt']), "log.txt")

    for batch_idx, (update_csv, update_npy) in enumerate(zip(update_csv_list, update_npy_list)):
        # 每一批都用历史最优权重初始化
        if batch_idx == 0:
            state_dict = torch.load(config['pretrain_best_ckpt'], map_location=device)
            model.load_state_dict(state_dict)
        else:
            if os.path.exists(global_best_ckpt):
                state_dict = torch.load(global_best_ckpt, map_location=device)
                model.load_state_dict(state_dict)
            else:
                state_dict = torch.load(checkpoint_path, map_location=device)
                model.load_state_dict(state_dict)
            
        # 每批次都要确保底层参数被冻结
        freeze_edge_layers(model, freeze_layers=('stem', 'block1', 'block2'))

        print(f"\n=== 当前无标签数据: {update_csv}, {update_npy} ===")
        batch_stats = {
            "batch_idx": batch_idx,
            "unlabeled_csv": update_csv,
            "unlabeled_npy": update_npy,
        }
        log_json_stats(log_path, batch_stats)

        # 加载无标签数据
        df_unlabeled = pd.read_csv(update_csv)
        train_signal_un = df_unlabeled['path'].tolist()
        soft_labels = np.load(update_npy)
        train_unlabeled_dataset = EdgeUnlabeledDataset(train_signal_un, soft_labels, transforms=train_transform)
        train_unlabeled_loader = DataLoader(train_unlabeled_dataset, batch_size=config['batch_size'], shuffle=True)

        params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = optim.Adam(params, lr=config['lr'])
        # ===== 加入ReduceLROnPlateau调度器 ======
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=2, verbose=True
        )

        # ==== 早停机制相关变量 ====
        early_stop_patience = 5  # 早停容忍轮数，建议大于scheduler的patience
        no_improve_count = 0
        best_val_auc = 0.0

        for epoch in range(config['epochs']):
            train_loss, train_sup, train_distill = train(
                model, train_labeled_loader, train_unlabeled_loader,
                optimizer, nn.BCEWithLogitsLoss(), nn.BCEWithLogitsLoss(),
                config.get('distill_alpha', 1.0), device, epoch, config['epochs'])

            val_loss, val_auc, val_metrics, val_gt, val_pred = valid_on_ptb(model, val_loader, nn.BCEWithLogitsLoss(), device, return_preds=True)
            val_mean_auc, val_auc_dict = compute_per_class_auc(val_gt, val_pred)
            val_mean_acc, val_acc_dict = compute_per_class_accuracy(val_gt, val_pred)

            print(f"Epoch {epoch} | TrainLoss: {train_loss:.4f} | TrainSup: {train_sup:.4f} | TrainDistill: {train_distill:.4f} | ValLoss: {val_loss:.4f} | ValAUC: {val_mean_auc:.4f} | ValACC: {val_mean_acc:.4f}")

            stats = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_sup": train_sup,
                "train_distill": train_distill,
                "val_loss": val_loss,
                "val_mean_auc": val_mean_auc,
                "val_mean_acc": val_mean_acc,
                "lr": optimizer.param_groups[0]['lr'],  # 记录当前学习率
            }
            log_json_stats(log_path, stats)

            # ==== 调度器步进 ====
            scheduler.step(val_mean_auc)

            # ==== 早停判定 ====
            if val_mean_auc > best_val_auc:
                best_val_auc = val_mean_auc
                torch.save(model.state_dict(), config['best_ckpt'])
                print("保存为该模型的最佳权重")
                no_improve_count = 0
            else:
                no_improve_count += 1
                print(f"验证集AUC连续未提升: {no_improve_count}轮")
            if no_improve_count >= early_stop_patience:
                print(f"早停: 连续{early_stop_patience}轮验证集AUC未提升，终止训练")
                break

        # 当前批次训练完，加载本批次最佳权重，验证AUC
        if os.path.exists(config['best_ckpt']):
            state_dict = torch.load(config['best_ckpt'], map_location=device)
            model.load_state_dict(state_dict)
            val_loss, val_auc, val_metrics, val_gt, val_pred = valid_on_ptb(model, val_loader, nn.BCEWithLogitsLoss(), device, return_preds=True)
            val_mean_auc, val_auc_dict = compute_per_class_auc(val_gt, val_pred)

            # 比较历史最优AUC与本批次最优AUC
            if val_mean_auc > global_best_auc:
                global_best_auc = val_mean_auc
                torch.save(model.state_dict(), global_best_ckpt)
                print(f"更新全局最优权重, val_mean_auc提升为: {global_best_auc:.4f}")
            else:
                # 恢复历史最优权重文件到best_ckpt
                if os.path.exists(global_best_ckpt):
                    state_dict = torch.load(global_best_ckpt, map_location=device)
                    torch.save(state_dict, config['best_ckpt'])
                print(f"本批次未提升, 使用历史最优权重继续后续批次训练")

        # 测试前加载当前批次最佳权重（即此时的best_ckpt已保证是更优）
        if os.path.exists(config['best_ckpt']):
            state_dict = torch.load(config['best_ckpt'], map_location=device)
            try:
                model.load_state_dict(state_dict)
            except Exception as e:
                print(f"Warning: could not load checkpoint weights, error: {e}")

        test_loss, test_auc, test_metrics, test_gt, test_pred = valid_on_ptb(model, test_loader, nn.BCEWithLogitsLoss(), device, return_preds=True)
        test_mean_auc, test_auc_dict = compute_per_class_auc(test_gt, test_pred)
        test_mean_acc, test_acc_dict = compute_per_class_accuracy(test_gt, test_pred)
        print(f"当前无标签批次 Test | Loss: {test_loss:.4f} | AUC: {test_mean_auc:.4f} | ACC: {test_mean_acc:.4f}")

        test_stats = {
            "test_loss": test_loss,
            "test_mean_auc": test_mean_auc,
            "test_mean_acc": test_mean_acc,
        }
        log_json_stats(log_path, test_stats)

        if test_mean_auc > best_test_auc:
            best_test_auc = test_mean_auc
            print(f"继续下一批无标签数据，当前test_mean_auc提升为: {best_test_auc:.4f}")
        else:
            print(f"test_mean_auc未提升，继续下一批无标签数据。")

if __name__ == "__main__":
    config = {
        'input_channels': 12,
        'num_classes': 39,
        'high_layers_channels': (512, 768, 1024, 1536),
        'lr': 1e-3,
        'batch_size': 32,
        'epochs': 20,
        'pretrain_best_ckpt': '/data_C/sdb1/lyi/ecg-chat/ECG-Chat-master/edge_2/save_trained/save_pretrained/edge_best.pt',
        'best_ckpt': '/data_C/sdb1/lyi/ecg-chat/ECG-Chat-master/edge_2/save_trained/save_update/edge_best.pt',
        'distill_alpha': 1,
        'labeled_csv': '/data_C/sdb1/lyi/ecg-chat/ECG-Chat-master/edge_2/dataset/sph/split/split_1_1000.csv',
        'val_csv': '/data_C/sdb1/lyi/ecg-chat/ECG-Chat-master/edge_2/dataset/sph/split/split_2_1000.csv',
        'test_csv': '/data_C/sdb1/lyi/ecg-chat/ECG-Chat-master/edge_2/dataset/sph/split/split_3_1000.csv',
        'data_dir': '/data_C/sdb1/lyi/ecg-chat/ECG-Chat-master/edge_2/dataset/sph/split',
    }
    main(config)