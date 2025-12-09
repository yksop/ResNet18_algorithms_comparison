import torch
import torch.nn.functional as F
import torch.nn as nn

from tqdm import tqdm


def train(
    model,
    trainloader,
    optimizer,
    criterion,
    device,
    scheduler,
    method="sgd",
):
    model.train()
    print(f"Training ({method.upper()})")
    train_running_loss = 0.0
    train_running_correct = 0
    counter = 0

    for i, data in tqdm(enumerate(trainloader), total=len(trainloader)):
        counter += 1
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # --- Standard SGD ---
        if method == "sgd":
            outputs = model(images)
            loss = criterion(outputs, labels)

        # --- FGSM adversarial training ---
        elif method == "fgsm":
            outputs = model(images)
            loss_clean = criterion(outputs, labels)

            adv_images = fgsm_attack(model, images, labels)
            adv_outputs = model(adv_images)
            loss_adv = criterion(adv_outputs, labels)

            alpha = 0.5
            loss = alpha * loss_clean + (1 - alpha) * loss_adv

        # --- PGD adversarial training ---
        elif method == "pgd":
            model.eval()

            adv_images = pgd_attack(model, images, labels)

            model.train()

            outputs = model(adv_images)
            loss = criterion(outputs, labels)

        # --- TRADES adversarial training ---
        elif method == "trades":
            model.eval()

            adv_images = trades_attack(model, images)

            model.train()

            clean_logits = model(images)
            adv_logits = model(adv_images)

            loss_ce = F.cross_entropy(clean_logits, labels)
            loss_kl = F.kl_div(
                F.log_softmax(adv_logits, dim=1),
                F.softmax(clean_logits, dim=1),
                reduction="batchmean",
            )

            beta = 6.0

            loss = loss_ce + beta * loss_kl

            outputs = clean_logits

        else:
            raise ValueError(f"Unknown training method: {method}")

        # Backpropagation
        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        # Accuracy
        _, preds = torch.max(outputs.data, 1)
        train_running_correct += (preds == labels).sum().item()
        train_running_loss += loss.item()

    # Loss e accuracy per l'intero epoch
    epoch_loss = train_running_loss / counter
    epoch_acc = 100.0 * (train_running_correct / len(trainloader.dataset))

    return epoch_loss, epoch_acc


def validate(model, testloader, criterion, device):
    model.eval()
    print("Validation")
    valid_running_loss = 0.0
    valid_running_correct = 0
    counter = 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(testloader), total=len(testloader)):
            counter += 1

            image, labels = data
            image = image.to(device)
            labels = labels.to(device)
            # Forward pass.
            outputs = model(image)
            # Calculate the loss.
            loss = criterion(outputs, labels)
            valid_running_loss += loss.item()
            # Calculate the accuracy.
            _, preds = torch.max(outputs.data, 1)
            valid_running_correct += (preds == labels).sum().item()

    # Loss and accuracy for the complete epoch.
    epoch_loss = valid_running_loss / counter
    epoch_acc = 100.0 * (valid_running_correct / len(testloader.dataset))
    return epoch_loss, epoch_acc


def fgsm_attack(model, images, labels, eps=(8 / 255) / 0.5):
    device = next(model.parameters()).device

    images = images.clone().detach().to(device)
    labels = labels.to(device)

    images.requires_grad_(True)

    outputs = model(images)
    loss = nn.CrossEntropyLoss()(outputs, labels)
    model.zero_grad()
    grads = torch.autograd.grad(loss, images, retain_graph=False, create_graph=False)[0]

    adv_images = images + eps * grads.sign()
    adv_images = torch.clamp(adv_images, -1, 1)

    return adv_images.detach()


def pgd_attack(
    model, images, labels, eps=(8 / 255) / 0.5, alpha=(2 / 255) / 0.5, iters=10
):
    device = next(model.parameters()).device
    labels = labels.to(device)

    delta = torch.zeros_like(images).uniform_(-eps, eps)

    delta = torch.clamp(images + delta, -1, 1) - images

    delta.requires_grad = True

    for _ in range(iters):
        adv_images = images + delta
        outputs = model(adv_images)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        grads = torch.autograd.grad(
            loss, delta, retain_graph=False, create_graph=False
        )[0]
        delta.data = delta.data + alpha * grads.sign()
        delta.data = torch.clamp(delta.data, -eps, eps)
        delta.data = torch.clamp(images + delta.data, -1, 1) - images

    return (images + delta).detach()


def trades_attack(model, images, eps=(8 / 255) / 0.5, alpha=(2 / 255) / 0.5, iters=10):
    device = next(model.parameters()).device
    images = images.clone().detach().to(device)

    with torch.no_grad():
        clean_logits = model(images)

    delta = torch.zeros_like(images).uniform_(-eps, eps)
    adv_images = torch.clamp(images + delta, -1, 1).detach()

    for _ in range(iters):
        adv_images.requires_grad_(True)
        adv_logits = model(adv_images)

        loss_kl = F.kl_div(
            F.log_softmax(adv_logits, dim=1),
            F.softmax(clean_logits, dim=1),
            reduction="batchmean",
        )

        grads = torch.autograd.grad(
            loss_kl, adv_images, retain_graph=False, create_graph=False
        )[0]

        adv_images.data = adv_images.data + alpha * grads.sign()

        delta = torch.clamp(adv_images.data - images, -eps, eps)
        adv_images.data = torch.clamp(images + delta, -1, 1)

    return adv_images.detach()
