import argparse
import os
import shutil
import tempfile
import matplotlib.pyplot as plt
import PIL
import torch
import numpy as np
from sklearn.metrics import classification_report
from monai.apps import download_and_extract, MedNISTDataset
from monai.config import print_config
from monai.data import decollate_batch
from monai.metrics import ROCAUCMetric
from monai.networks.nets import DenseNet121, resnet50
from monai.transforms import (
    Activations,
    AddChannel,
    AsDiscrete,
    Compose,
    LoadImage,
    RandFlip,
    RandRotate,
    RandZoom,
    ScaleIntensity,
    EnsureType,
)
from monai.utils import set_determinism

parser = argparse.ArgumentParser(description='AnisotropicMedicalImage Project')
parser.add_argument('--seed', default=1, type=int, help='seed for initializing training. ')
parser.add_argument('--num-workers', default=0, type=int, help='number of workers.')
parser.add_argument('--batch-size', default=256, type=int, help='mini batch size.')
parser.add_argument('--max-epoch', default=10, type=int, help='max epoch.')
parser.add_argument('--num-class', default=6, type=int, help='number of class categories')
parser.add_argument('--dir-data', default='C:/codes/mednist_selfsup/data/', type=str, help='Directory to save the data.')
parser.add_argument('--dir-results', default='C:/codes/mednist_selfsup/results/', type=str, help='Directory to save the data.')
args = parser.parse_args()

set_determinism(seed=args.seed)
train_transforms = Compose(
    [
        LoadImage(image_only=True),
        AddChannel(),
        ScaleIntensity(),
        RandRotate(range_x=np.pi / 12, prob=0.5, keep_size=True),
        RandFlip(spatial_axis=0, prob=0.5),
        RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.5),
        EnsureType(),
    ]
)

val_transforms = Compose(
    [LoadImage(image_only=True), AddChannel(), ScaleIntensity(), EnsureType()])

y_pred_trans = Compose([EnsureType(), Activations(softmax=True)])
y_trans = Compose([EnsureType(), AsDiscrete(to_onehot=args.num_class)])

train_ds = MedNISTDataset(root_dir=args.dir_data, section='training', transform=train_transforms, download=True, seed=args.seed, num_workers=args.num_workers)
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

val_ds = MedNISTDataset(root_dir=args.dir_data, section='validation', transform=val_transforms, download=True, seed=args.seed, num_workers=args.num_workers)
val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

test_ds = MedNISTDataset(root_dir=args.dir_data, section='test', transform=val_transforms, download=True, seed=args.seed, num_workers=args.num_workers)
test_loader = torch.utils.data.DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)


# val_ds = MedNISTDataset(val_x, val_y, val_transforms)
# val_loader = torch.utils.data.DataLoader(
#     val_ds, batch_size=args.batch_size, num_workers=args.num_workers)
#
# test_ds = MedNISTDataset(test_x, test_y, val_transforms)
# test_loader = torch.utils.data.DataLoader(
#     test_ds, batch_size=args.batch_size, num_workers=args.num_workers)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = DenseNet121(spatial_dims=2, in_channels=1,
#                     out_channels=num_class).to(device)
model = resnet50(spatial_dims=2, n_input_channels=1,
                    n_classes=args.num_class).to(device)

loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), 1e-5)
val_interval = 1
auc_metric = ROCAUCMetric()

best_metric = -1
best_metric_epoch = -1
epoch_loss_values = []
metric_values = []

for epoch in range(args.max_epochs):
    print("-" * 10)
    print(f"epoch {epoch + 1}/{args.max_epochs}")
    model.train()
    epoch_loss = 0
    step = 0
    for batch_data in train_loader:
        step += 1
        inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        print(
            f"{step}/{len(train_ds) // train_loader.batch_size}, "
            f"train_loss: {loss.item():.4f}")
        epoch_len = len(train_ds) // train_loader.batch_size
    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)
    print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

    if (epoch + 1) % val_interval == 0:
        model.eval()
        with torch.no_grad():
            y_pred = torch.tensor([], dtype=torch.float32, device=device)
            y = torch.tensor([], dtype=torch.long, device=device)
            for val_data in val_loader:
                val_images, val_labels = (
                    val_data[0].to(device),
                    val_data[1].to(device),
                )
                y_pred = torch.cat([y_pred, model(val_images)], dim=0)
                y = torch.cat([y, val_labels], dim=0)
            y_onehot = [y_trans(i) for i in decollate_batch(y)]
            y_pred_act = [y_pred_trans(i) for i in decollate_batch(y_pred)]
            auc_metric(y_pred_act, y_onehot)
            result = auc_metric.aggregate()
            auc_metric.reset()
            del y_pred_act, y_onehot
            metric_values.append(result)
            acc_value = torch.eq(y_pred.argmax(dim=1), y)
            acc_metric = acc_value.sum().item() / len(acc_value)
            if result > best_metric:
                best_metric = result
                best_metric_epoch = epoch + 1
                torch.save(model.state_dict(), os.path.join(
                    args.dir_results, "best_metric_model.pth"))
                print("saved new best metric model")
            print(
                f"current epoch: {epoch + 1} current AUC: {result:.4f}"
                f" current accuracy: {acc_metric:.4f}"
                f" best AUC: {best_metric:.4f}"
                f" at epoch: {best_metric_epoch}"
            )
