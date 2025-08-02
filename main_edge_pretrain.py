import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from model.edge_model import EdgeXResNet1d
from Dataset import EdgeLabeledDataset
import time
from dataset.ecg_transform import PreprocessCfg, ecg_transform_v2
import pandas as pd
import json

def ked2edge_key(ked_key):
    # stem（3层）
    if ked_key.startswith('0.'):
        return 'stem.0.' + ked_key[2:]
    if ked_key.startswith('1.'):
        return 'stem.1.' + ked_key[2:]
    if ked_key.startswith('2.'):
        return 'stem.2.' + ked_key[2:]
    # block1
    if ked_key.startswith('4.'):
        rest = ked_key[2:]
        idx, remain = rest.split('.', 1)
        return f'block1.{idx}.{remain}'
    # block2
    if ked_key.startswith('5.'):
        rest = ked_key[2:]
        idx, remain = rest.split('.', 1)
        return f'block2.{idx}.{remain}'
    return None

def load_and_freeze_base(edge_model, ked_ckpt_path, freeze_layers=('stem', 'block1', 'block2')):
    ked_state_dict_raw = torch.load(ked_ckpt_path, map_location='cpu')
    if "ecg_model" in ked_state_dict_raw:
        ked_state_dict = ked_state_dict_raw["ecg_model"]
    else:
        ked_state_dict = ked_state_dict_raw
    edge_state_dict = edge_model.state_dict()
    for k_ked, v_ked in ked_state_dict.items():
        k_edge = ked2edge_key(k_ked)
        if k_edge is not None and k_edge in edge_state_dict:
            if v_ked.shape == edge_state_dict[k_edge].shape:
                edge_state_dict[k_edge] = v_ked
    edge_model.load_state_dict(edge_state_dict)
    for name, param in edge_model.named_parameters():
        if any(name.startswith(layer) for layer in freeze_layers):
            param.requires_grad = False
    return edge_model


from sklearn.metrics import accuracy_score, roc_auc_score

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

def main(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EdgeXResNet1d(
        num_classes=config['num_classes'],
        pool_type='avg'
    ).to(device)

    model = load_and_freeze_base(model, ked_ckpt_path=config['ked_ckpt'])
    params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(params, lr=config['lr'])
    criterion = nn.BCEWithLogitsLoss()

    # 训练、验证、测试集
    df_train = pd.read_csv(config['train_csv'])
    train_signal = df_train['path'].tolist()
    train_labels = df_train['label']
    df_val = pd.read_csv(config['val_csv'])
    val_signal = df_val['path'].tolist()
    val_labels = df_val['label']

    df_test = pd.read_csv(config['test_csv'])
    test_signal = df_test['path'].tolist()
    test_labels = df_test['label']
    
    print(len(train_signal))
    print(len(val_signal))
    print(len(test_signal))
    preprocess_cfg = PreprocessCfg(seq_length=5000, duration=10, sampling_rate=500)
    train_transform = ecg_transform_v2(cfg=preprocess_cfg, is_train=True)
    val_transform = ecg_transform_v2(cfg=preprocess_cfg, is_train=False)
    test_transform = ecg_transform_v2(cfg=preprocess_cfg, is_train=False)

    train_dataset = EdgeLabeledDataset(train_signal, train_labels, transforms=train_transform)
    val_dataset = EdgeLabeledDataset(val_signal, val_labels, transforms=val_transform)
    test_dataset = EdgeLabeledDataset(test_signal, test_labels, transforms=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

    best_val_auc = 0
    log_path = os.path.join(os.path.dirname(config['best_ckpt']), "log.txt")

    for epoch in range(config['epochs']):
        train_loss = train(model, train_loader, optimizer, criterion, device, epoch, config['epochs'])

        # 验证
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
    print(f"Final Test | Loss: {test_loss:.4f} | AUC: {test_mean_auc:.4f} | ACC: {test_mean_acc:.4f}")

    # 记录最终测试集结果到日志
    test_stats = {
        "test_loss": test_loss,
        "test_mean_auc": test_mean_auc,
        "test_mean_acc": test_mean_acc,
    }
    log_json_stats(log_path, test_stats)

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

from sklearn.metrics import roc_auc_score
def compute_macro_auc(gt, pred):
    aucs = []
    for i in range(gt.shape[1]):
        try:
            aucs.append(roc_auc_score(gt[:, i], pred[:, i]))
        except:
            continue
    return np.mean(aucs) if aucs else 0.0

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
    
if __name__ == "__main__":
    config = {
        'input_channels': 12,
        'num_classes': 39,
        'high_layers_channels': (512, 768, 1024, 1536),
        'ked_ckpt': '/data_C/sdb1/lyi/ked3/control-spiderman-ECGFM-KED-456810e/trained_model/checkpoints_finetune/finetune_sph.pt',
        'train_csv': '/data_C/sdb1/lyi/ecg-chat/ECG-Chat-master/edge_2/dataset/sph/split/split_1_1000.csv',
        'val_csv': '/data_C/sdb1/lyi/ecg-chat/ECG-Chat-master/edge_2/dataset/sph/split/split_2_1000.csv',
        'test_csv': '/data_C/sdb1/lyi/ecg-chat/ECG-Chat-master/edge_2/dataset/sph/split/split_3_1000.csv',
        'lr': 1e-3,
        'batch_size': 32,
        'epochs': 10,
        'best_ckpt': '/data_C/sdb1/lyi/ecg-chat/ECG-Chat-master/edge_2/save_trained/save_pretrained/edge_best.pt',
    }
    main(config)