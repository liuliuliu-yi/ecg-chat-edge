import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from model.edge_model import EdgeXResNet1d
import pandas as pd
import time
import json
from dataset.ecg_transform import PreprocessCfg, ecg_transform_v2
from sklearn.metrics import accuracy_score, roc_auc_score
from Dataset import EdgeLabeledDataset

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

def train(model, loader, optimizer, criterion, device, epoch_idx, max_epoch):
    model.train()
    total_loss = 0
    start_time = time.time()
    batch_num = len(loader)
    for i, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if (i+1) % 50 == 0 or (i+1) == batch_num:
            elapsed = time.time() - start_time
            percent = (i+1) / batch_num
            eta = elapsed / percent - elapsed
            lr = optimizer.param_groups[0]['lr']
            print(
                f"Train Epoch: [{epoch_idx+1}/{max_epoch}] "
                f"[{i+1:5d}/{batch_num}] "
                f"eta: {int(eta//60):02d}:{int(eta%60):02d} "
                f"lr: {lr:.6f} "
                f"loss: {loss.item():.6f} "
                f"avg_loss: {total_loss/(i+1):.6f}"
            )
    avg_loss = total_loss / batch_num
    print(f"Train: Avg Loss {avg_loss:.4f}")
    return avg_loss

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

def freeze_edge_layers(edge_model, freeze_layers=('stem', 'block1', 'block2')):
    for name, param in edge_model.named_parameters():
        if any(name.startswith(layer) for layer in freeze_layers):
            param.requires_grad = False
        else:
            param.requires_grad = True
    print(f"已冻结层: {freeze_layers}")

def extract_num(file):
    import re
    nums = re.findall(r'(\d+)', os.path.basename(file))
    return int(nums[-1]) if nums else -1

def main(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. 读取伪标签hard label csv列表
    hardlabel_dir = config.get('hardlabel_dir', config.get('data_dir', './'))
    hardlabel_csv_list = [os.path.join(hardlabel_dir, f) for f in os.listdir(hardlabel_dir)
                          if f.startswith("ecg_hardlabel_save_update_") and f.endswith(".csv")]
    hardlabel_csv_list = sorted(hardlabel_csv_list, key=extract_num)

    # 2. 原始有标签数据
    df_train = pd.read_csv(config['labeled_csv'])
    train_signal = df_train['path'].tolist()
    train_labels = df_train['label'].tolist()

    # 验证、测试集
    df_val = pd.read_csv(config['val_csv'])
    val_signal = df_val['path'].tolist()
    val_labels = df_val['label'].tolist()
    df_test = pd.read_csv(config['test_csv'])
    test_signal = df_test['path'].tolist()
    test_labels = df_test['label'].tolist()

    preprocess_cfg = PreprocessCfg(seq_length=5000, duration=10, sampling_rate=500)
    train_transform = ecg_transform_v2(cfg=preprocess_cfg, is_train=True)
    val_transform = ecg_transform_v2(cfg=preprocess_cfg, is_train=False)
    test_transform = ecg_transform_v2(cfg=preprocess_cfg, is_train=False)

    val_dataset = EdgeLabeledDataset(val_signal, val_labels, transforms=val_transform)
    test_dataset = EdgeLabeledDataset(test_signal, test_labels, transforms=test_transform)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

    best_test_auc = 0.0
    log_path = os.path.join(os.path.dirname(config['best_ckpt']), "log.txt")

    # 多轮finetune，每轮合并所有历史及本轮伪标签csv，并加载上一轮最佳权重
    merged_signal = list(train_signal)  # 累加的有标签信号路径
    merged_labels = list(train_labels)  # 累加的有标签标签

    best_ckpt = config['pretrain_best_ckpt']  # 第一轮用预训练权重，后续用上一轮best_ckpt
    for round_idx, hardlabel_csv in enumerate(hardlabel_csv_list):
        print(f"\n=== 当前finetune轮次: {round_idx+1}，硬标签文件: {hardlabel_csv} ===")
        batch_stats = {
            "batch_idx": round_idx + 1,
            "hardlabel_csv": hardlabel_csv,
        }
        log_json_stats(log_path, batch_stats)

        # 合并历史的和当前轮的伪标签csv
        df_hardlabel = pd.read_csv(hardlabel_csv)
        hardlabel_signal = df_hardlabel['path'].tolist()
        hardlabel_labels = df_hardlabel['label'].tolist()
        merged_signal += hardlabel_signal
        merged_labels += hardlabel_labels

        # 新建模型并加载上一轮最佳权重，冻结底层
        model = EdgeXResNet1d(
            num_classes=config['num_classes'],
            pool_type='avg'
        ).to(device)
        state_dict = torch.load(best_ckpt, map_location=device)
        model.load_state_dict(state_dict)
        freeze_edge_layers(model, freeze_layers=('stem', 'block1', 'block2'))

        train_dataset = EdgeLabeledDataset(merged_signal, merged_labels, transforms=train_transform)
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)

        params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = optim.Adam(params, lr=config['lr'])
        criterion = nn.BCEWithLogitsLoss()

        best_val_auc = 0

        for epoch in range(config['epochs']):
            train_loss = train(model, train_loader, optimizer, criterion, device, epoch, config['epochs'])

            val_loss, val_auc, val_metrics, val_gt, val_pred = valid_on_ptb(model, val_loader, criterion, device, return_preds=True)
            val_mean_auc, val_auc_dict = compute_per_class_auc(val_gt, val_pred)
            val_mean_acc, val_acc_dict = compute_per_class_accuracy(val_gt, val_pred)

            print(f"Epoch {epoch} | TrainLoss: {train_loss:.4f} | ValLoss: {val_loss:.4f} | ValAUC: {val_mean_auc:.4f} | ValACC: {val_mean_acc:.4f}")

            stats = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_mean_auc": val_mean_auc,
                "val_mean_acc": val_mean_acc,
                "lr": config['lr'],
            }
            log_json_stats(log_path, stats)

            # 终止条件判断（以验证集AUC为标准）
            if val_mean_auc > best_val_auc:
                best_val_auc = val_mean_auc
                torch.save(model.state_dict(), config['best_ckpt'])

        print("\nEvaluating best model on test set ...")
        model.load_state_dict(torch.load(config['best_ckpt'], map_location=device))
        test_loss, test_auc, test_metrics, test_gt, test_pred = valid_on_ptb(model, test_loader, criterion, device, return_preds=True)
        test_mean_auc, test_auc_dict = compute_per_class_auc(test_gt, test_pred)
        test_mean_acc, test_acc_dict = compute_per_class_accuracy(test_gt, test_pred)
        print(f"Test | Loss: {test_loss:.4f} | AUC: {test_mean_auc:.4f} | ACC: {test_mean_acc:.4f}")

        # 记录最终测试集结果到日志
        test_stats = {
            "round": round_idx + 1,
            "hardlabel_csv": hardlabel_csv,
            "test_loss": test_loss,
            "test_mean_auc": test_mean_auc,
            "test_mean_acc": test_mean_acc,
        }
        log_json_stats(log_path, test_stats)

        if test_mean_auc > best_test_auc:
            best_test_auc = test_mean_auc
            print(f"test_mean_auc提升为: {best_test_auc:.4f}")
        else:
            print(f"test_mean_auc未提升。")

        # 下一轮finetune加载此轮的最佳权重
        best_ckpt = config['best_ckpt']

if __name__ == "__main__":
    config = {
        'input_channels': 12,
        'num_classes': 39,
        'high_layers_channels': (512, 768, 1024, 1536),
        'pretrain_best_ckpt': '/data_C/sdb1/lyi/ecg-chat/ECG-Chat-master/edge_2/save_trained/save_pretrained/edge_best.pt',
        'labeled_csv': '/data_C/sdb1/lyi/ecg-chat/ECG-Chat-master/edge_2/dataset/sph/split/split_1_1000.csv',
        'hardlabel_dir': '/data_C/sdb1/lyi/ecg-chat/ECG-Chat-master/edge_2/dataset/sph/split',  # 存放伪标签csv的目录（与原data_dir相同可复用）
        'val_csv': '/data_C/sdb1/lyi/ecg-chat/ECG-Chat-master/edge_2/dataset/sph/split/split_2_1000.csv',
        'test_csv': '/data_C/sdb1/lyi/ecg-chat/ECG-Chat-master/edge_2/dataset/sph/split/split_3_1000.csv',
        'lr': 1e-3,
        'batch_size': 32,
        'epochs': 20,
        'best_ckpt': '/data_C/sdb1/lyi/ecg-chat/ECG-Chat-master/edge_2/save_trained/save_update_concat/edge_best.pt',
    }
    main(config)