import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import numpy as np
import random
import os

from ResNet import ResNet, BasicBlock
from ResNet18_torchvision import build_model
from training_utils import train, validate
from utils import save_plots, get_data

best_val_acc = 0.0
best_epoch = -1
warmup_epochs = 5


parser = argparse.ArgumentParser()
parser.add_argument(
    "-m",
    "--model",
    default="scratch",
    help="choose model built from scratch or the Torchvision model",
    choices=["scratch", "torchvision"],
)
args = vars(parser.parse_args())

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
np.random.seed(seed)
random.seed(seed)

# Learning and training parameters.
epochs = 60
batch_size = 256
learning_rate = 0.1
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_loader, valid_loader = get_data(batch_size=batch_size)

# Define model based on the argument parser string.
if args["model"] == "scratch":
    print("[INFO]: Training ResNet18 built from scratch...")
    model = ResNet(img_channels=3, num_layers=18, block=BasicBlock, num_classes=10).to(
        device
    )
    plot_name = "resnet_scratch"
elif args["model"] == "torchvision":
    print("[INFO]: Training the Torchvision ResNet18 model...")
    model = build_model(pretrained=False, fine_tune=True, num_classes=10).to(device)
    plot_name = "resnet_torchvision"
else:
    raise ValueError(f"Unexpected model choice: {args['model']}")

# print(model)

# Total parameters and trainable parameters.
total_params = sum(p.numel() for p in model.parameters())
print(f"{total_params:,} total parameters.")
total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"{total_trainable_params:,} training parameters.")

# Loss function.
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

methods = ["sgd", "fgsm", "pgd", "trades"]


if __name__ == "__main__":
    for method in methods:
        model = ResNet(
            img_channels=3, num_layers=18, block=BasicBlock, num_classes=10
        ).to(device)
        optimizer = torch.optim.SGD(
            model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4
        )
        if method != "sgd":
            print(f"[INFO] Starting {warmup_epochs} warmup epochs with SGD...")

            for epoch in range(warmup_epochs):
                warmup_lr = learning_rate * (epoch + 1) / warmup_epochs
                for param_group in optimizer.param_groups:
                    param_group["lr"] = warmup_lr

                train_epoch_loss, train_epoch_acc = train(
                    model,
                    train_loader,
                    optimizer,
                    criterion,
                    device,
                    scheduler=None,
                    method="sgd",
                )
                print(
                    f"[WARMUP] Epoch {epoch+1}/{warmup_epochs}, LR={warmup_lr:.4f}, Train Acc={train_epoch_acc:.2f}"
                )

            print("[INFO] Warmup completed. Starting full training loop...\n")

        print(f"[INFO] Training {method.upper()} model")

        save_dir = os.path.join("models", method)
        os.makedirs(save_dir, exist_ok=True)

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=learning_rate,
            epochs=epochs,
            steps_per_epoch=len(train_loader),
        )

        train_loss, valid_loss = [], []
        train_acc, valid_acc = [], []

        for epoch in range(epochs):
            train_epoch_loss, train_epoch_acc = train(
                model,
                train_loader,
                optimizer,
                criterion,
                device,
                scheduler,
                method=method,
            )
            valid_epoch_loss, valid_epoch_acc = validate(
                model, valid_loader, criterion, device
            )

            train_loss.append(train_epoch_loss)
            valid_loss.append(valid_epoch_loss)
            train_acc.append(train_epoch_acc)
            valid_acc.append(valid_epoch_acc)

            print(
                f"Epoch {epoch+1}: {method.upper()} train acc {train_epoch_acc:.2f}, val acc {valid_epoch_acc:.2f}"
            )

        model_path = os.path.join(save_dir, f"{method}_model.pth")
        torch.save(model.state_dict(), model_path)
        print(f"[INFO] Saved model to {model_path}")

        save_dir = os.path.join(method)
        save_plots(
            train_acc,
            valid_acc,
            train_loss,
            valid_loss,
            name=os.path.join(save_dir, f"{method}_plots"),
        )

        print(f"[INFO] Finished training {method.upper()} model\n")

    print("[INFO] All adversarial models trained and saved!")
