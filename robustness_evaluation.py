import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import argparse

from ResNet import ResNet, BasicBlock
from ResNet18_torchvision import build_model
from utils import get_data
from training_utils import fgsm_attack, pgd_attack

import os, sys

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(project_root, "autoattack"))


from autoattack import AutoAttack


def evaluate_clean_accuracy(model, dataloader, device):
    """Valuta accuracy su dati puliti (non perturbati)"""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Clean accuracy"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100.0 * correct / total
    return accuracy


def evaluate_adversarial_accuracy(
    model, dataloader, device, attack_fn, attack_name, **attack_params
):
    """Valuta accuracy su esempi adversariali"""
    model.eval()
    correct = 0
    total = 0

    for images, labels in tqdm(dataloader, desc=f"{attack_name} attack"):
        images, labels = images.to(device), labels.to(device)

        # Genera esempi adversariali
        adv_images = attack_fn(model, images, labels, **attack_params)

        # Valuta su esempi adversariali
        with torch.no_grad():
            outputs = model(adv_images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100.0 * correct / total
    return accuracy


def evaluate_robustness_curve(
    model, dataloader, device, attack_fn, attack_name, epsilon_values
):
    """Valuta accuracy per diversi valori di epsilon"""
    accuracies = []

    for eps in epsilon_values:
        print(f"\nEvaluating {attack_name} with epsilon={eps:.4f}")
        acc = evaluate_adversarial_accuracy(
            model, dataloader, device, attack_fn, attack_name, eps=eps
        )
        accuracies.append(acc)
        print(f"Accuracy: {acc:.2f}%")

    return accuracies


def autoattack_evaluation(
    model, dataloader, device, epsilon=8 / 255, batch_size=128, n_samples=1000
):
    """
    Valutazione con AutoAttack - versione migliorata
    """
    try:
        from autoattack import AutoAttack
    except ImportError:
        print(
            "❌ AutoAttack non installato. Installa con: pip install git+https://github.com/fra31/auto-attack"
        )
        return None

    model.eval()

    # Colleziona tutti i dati
    all_images = []
    all_labels = []

    for images, labels in dataloader:
        all_images.append(images)
        all_labels.append(labels)

    x_test = torch.cat(all_images, dim=0).to(device)
    y_test = torch.cat(all_labels, dim=0).to(device)

    # Limita il numero di sample per velocità
    n_samples = min(n_samples, len(x_test))
    x_test = x_test[:n_samples]
    y_test = y_test[:n_samples]

    print(f"🔧 Running AutoAttack on {n_samples} samples (ε={epsilon:.3f})...")

    try:
        # Inizializza AutoAttack
        adversary = AutoAttack(
            model,
            norm="Linf",
            eps=epsilon,
            version="standard",
            log_path=None,
            device=device,
        )

        # Configurazione per velocità
        adversary.attacks_to_run = ["apgd-ce", "apgd-t", "fab-t", "square"]

        print("   Attacks: APGD-CE, APGD-T, FAB-T, Square")

        # Esegui valutazione
        with torch.no_grad():
            x_adv = adversary.run_standard_evaluation(x_test, y_test, bs=batch_size)

        # Calcola accuracy
        with torch.no_grad():
            outputs = model(x_adv)
            _, predicted = torch.max(outputs, 1)
            correct = (predicted == y_test).sum().item()
            accuracy = 100.0 * correct / n_samples

        print(f"   ✅ AutoAttack completed")
        return accuracy

    except Exception as e:
        print(f"   ❌ AutoAttack failed: {e}")
        # Fallback: prova solo con APGD-CE
        try:
            print("   🔄 Trying APGD-CE only...")
            from autoattack import AutoAttack

            adversary = AutoAttack(
                model,
                norm="Linf",
                eps=epsilon,
                version="custom",
                log_path=None,
                device=device,
            )
            adversary.attacks_to_run = ["apgd-ce"]

            with torch.no_grad():
                x_adv = adversary.run_standard_evaluation(x_test, y_test, bs=batch_size)

            with torch.no_grad():
                outputs = model(x_adv)
                _, predicted = torch.max(outputs, 1)
                correct = (predicted == y_test).sum().item()
                accuracy = 100.0 * correct / n_samples

            return accuracy
        except Exception as e2:
            print(f"   ❌ APGD-CE also failed: {e2}")
            return None


def strong_fgsm_attack(model, images, labels, eps=8 / 255, num_restarts=10):
    """
    FGSM rinforzata con multiple random restarts
    """
    model.eval()
    device = images.device

    best_adv = images.clone()
    worst_loss = torch.zeros(images.size(0)).to(device) + float("inf")

    for i in range(num_restarts):
        # Inizializzazione random dentro la sfera ε
        delta = torch.empty_like(images).uniform_(-eps, eps)
        adv_images = torch.clamp(images + delta, 0, 1)
        adv_images.requires_grad = True

        outputs = model(adv_images)
        loss = nn.CrossEntropyLoss(reduction="none")(outputs, labels)

        # Per ogni immagine, tiene l'attacco che causa la loss più alta
        update_mask = loss > worst_loss
        worst_loss[update_mask] = loss[update_mask]
        best_adv[update_mask] = adv_images[update_mask].detach()

    return best_adv


def momentum_fgsm_attack(model, images, labels, eps=8 / 255, momentum=0.9, steps=5):
    """
    FGSM con momentum - simile a PGD ma con meno steps
    """
    model.eval()
    device = images.device

    adv_images = images.clone().detach()
    accumulated_grad = torch.zeros_like(images)

    # Inizializzazione random
    adv_images = adv_images + torch.empty_like(adv_images).uniform_(-eps, eps)
    adv_images = torch.clamp(adv_images, 0, 1).detach()

    for step in range(steps):
        adv_images.requires_grad = True

        outputs = model(adv_images)
        loss = nn.CrossEntropyLoss()(outputs, labels)

        grad = torch.autograd.grad(
            loss, adv_images, retain_graph=False, create_graph=False
        )[0]

        # Accumula il gradiente con momentum
        accumulated_grad = momentum * accumulated_grad + grad / torch.mean(
            torch.abs(grad), dim=(1, 2, 3), keepdim=True
        )

        adv_images = adv_images.detach() + (eps / steps) * torch.sign(accumulated_grad)
        delta = torch.clamp(adv_images - images, min=-eps, max=eps)
        adv_images = torch.clamp(images + delta, 0, 1).detach()

    return adv_images


def comprehensive_evaluation(model, dataloader, device, model_name):
    """Valutazione completa di un modello con AutoAttack"""
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_name}")
    print(f"{'='*60}")

    results = {}

    # 1. Clean Accuracy
    print("\n[1/10] Evaluating clean accuracy...")
    clean_acc = evaluate_clean_accuracy(model, dataloader, device)
    results["clean"] = clean_acc
    print(f"Clean Accuracy: {clean_acc:.2f}%")

    # 2. FGSM Standard
    print("\n[2/10] Evaluating Standard FGSM (eps=8/255)...")
    fgsm_acc = evaluate_adversarial_accuracy(
        model, dataloader, device, fgsm_attack, "FGSM", eps=8 / 255
    )
    results["fgsm_std_8/255"] = fgsm_acc
    print(f"Standard FGSM Accuracy: {fgsm_acc:.2f}%")

    # 3. FGSM con Random Restarts
    print("\n[3/10] Evaluating FGSM with Random Restarts (eps=8/255)...")
    fgsm_rr_acc = evaluate_adversarial_accuracy(
        model,
        dataloader,
        device,
        strong_fgsm_attack,
        "FGSM-RR",
        eps=8 / 255,
        num_restarts=10,
    )
    results["fgsm_rr_8/255"] = fgsm_rr_acc
    print(f"FGSM-RR Accuracy: {fgsm_rr_acc:.2f}%")

    # 4. PGD-20
    print("\n[4/10] Evaluating PGD-20 (eps=8/255)...")
    pgd20_acc = evaluate_adversarial_accuracy(
        model,
        dataloader,
        device,
        pgd_attack,
        "PGD-20",
        eps=8 / 255,
        alpha=2 / 255,
        iters=20,
    )
    results["pgd20_8/255"] = pgd20_acc
    print(f"PGD-20 Accuracy: {pgd20_acc:.2f}%")

    # 5. PGD-50
    print("\n[5/10] Evaluating PGD-50 (eps=8/255)...")
    pgd50_acc = evaluate_adversarial_accuracy(
        model,
        dataloader,
        device,
        pgd_attack,
        "PGD-50",
        eps=8 / 255,
        alpha=1 / 255,
        iters=50,
    )
    results["pgd50_8/255"] = pgd50_acc
    print(f"PGD-50 Accuracy: {pgd50_acc:.2f}%")

    # 6. PGD-100
    print("\n[6/10] Evaluating PGD-100 (eps=8/255)...")
    pgd100_acc = evaluate_adversarial_accuracy(
        model,
        dataloader,
        device,
        pgd_attack,
        "PGD-100",
        eps=8 / 255,
        alpha=0.5 / 255,
        iters=100,
    )
    results["pgd100_8/255"] = pgd100_acc
    print(f"PGD-100 Accuracy: {pgd100_acc:.2f}%")

    # 7. AutoAttack (ε=8/255) - STANDARD
    print("\n[7/10] Evaluating AutoAttack (eps=8/255)...")
    try:
        aa_acc = autoattack_evaluation(
            model, dataloader, device, epsilon=8 / 255, batch_size=128
        )
        if aa_acc is not None:
            results["autoattack_8/255"] = aa_acc
            print(f"AutoAttack Accuracy (eps=8/255): {aa_acc:.2f}%")
        else:
            results["autoattack_8/255"] = 0.0
            print("AutoAttack failed or not available")
    except Exception as e:
        print(f"AutoAttack error: {e}")
        results["autoattack_8/255"] = 0.0

    # 8. AutoAttack (ε=4/255) - più facile
    print("\n[8/10] Evaluating AutoAttack (eps=4/255)...")
    try:
        aa_easy_acc = autoattack_evaluation(
            model, dataloader, device, epsilon=4 / 255, batch_size=128
        )
        if aa_easy_acc is not None:
            results["autoattack_4/255"] = aa_easy_acc
            print(f"AutoAttack Accuracy (eps=4/255): {aa_easy_acc:.2f}%")
        else:
            results["autoattack_4/255"] = 0.0
    except Exception as e:
        print(f"AutoAttack error: {e}")
        results["autoattack_4/255"] = 0.0

    # 9. AutoAttack (ε=12/255) - più difficile
    print("\n[9/10] Evaluating AutoAttack (eps=12/255)...")
    try:
        aa_hard_acc = autoattack_evaluation(
            model, dataloader, device, epsilon=12 / 255, batch_size=128
        )
        if aa_hard_acc is not None:
            results["autoattack_12/255"] = aa_hard_acc
            print(f"AutoAttack Accuracy (eps=12/255): {aa_hard_acc:.2f}%")
        else:
            results["autoattack_12/255"] = 0.0
    except Exception as e:
        print(f"AutoAttack error: {e}")
        results["autoattack_12/255"] = 0.0

    # 10. FGSM Large Epsilon
    print("\n[10/10] Evaluating FGSM Large Epsilon (eps=16/255)...")
    fgsm_large_acc = evaluate_adversarial_accuracy(
        model,
        dataloader,
        device,
        strong_fgsm_attack,
        "FGSM-RR",
        eps=16 / 255,
        num_restarts=10,
    )
    results["fgsm_rr_16/255"] = fgsm_large_acc
    print(f"FGSM-RR Large Epsilon Accuracy: {fgsm_large_acc:.2f}%")

    return results


def plot_robustness_comparison(all_results, save_dir="results"):
    """Crea grafici di confronto tra i metodi con attacchi più forti"""
    os.makedirs(save_dir, exist_ok=True)

    # 1. Bar plot comparativo - USA ATTACCHI FORTI
    fig, ax = plt.subplots(figsize=(16, 6))

    methods = list(all_results.keys())
    metrics = ["clean", "pgd20_8/255", "pgd50_8/255", "pgd100_8/255"]  # Attacchi forti
    metric_labels = [
        "Clean",
        "PGD-20\n(ε=8/255)",
        "PGD-50\n(ε=8/255)",
        "PGD-100\n(ε=8/255)",
    ]

    x = np.arange(len(metrics))
    width = 0.2

    for i, method in enumerate(methods):
        values = [all_results[method].get(m, 0) for m in metrics]
        ax.bar(x + i * width, values, width, label=method, alpha=0.8)

    ax.set_xlabel("Attack Type", fontsize=12, fontweight="bold")
    ax.set_ylabel("Accuracy (%)", fontsize=12, fontweight="bold")
    ax.set_title(
        "Model Robustness Comparison - Strong Attacks", fontsize=14, fontweight="bold"
    )
    ax.set_xticks(x + width * (len(methods) - 1) / 2)
    ax.set_xticklabels(metric_labels)
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, "robustness_comparison_strong.png"),
        dpi=300,
        bbox_inches="tight",
    )
    print(f"\nSaved: {os.path.join(save_dir, 'robustness_comparison_strong.png')}")

    # 2. Heatmap con attacchi forti
    fig, ax = plt.subplots(figsize=(12, 6))

    data = []
    for method in methods:
        row = [all_results[method].get(m, 0) for m in metrics]
        data.append(row)

    df = pd.DataFrame(data, index=methods, columns=metric_labels)
    sns.heatmap(
        df,
        annot=True,
        fmt=".1f",
        cmap="RdYlGn",
        vmin=0,
        vmax=100,
        cbar_kws={"label": "Accuracy (%)"},
        ax=ax,
    )
    ax.set_title("Robustness Heatmap - Strong Attacks", fontsize=14, fontweight="bold")

    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, "robustness_heatmap_strong.png"),
        dpi=300,
        bbox_inches="tight",
    )
    print(f"Saved: {os.path.join(save_dir, 'robustness_heatmap_strong.png')}")


def plot_robustness_curves(all_curves, save_dir="results"):
    """Plotta le curve di robustness per tutti i modelli"""
    os.makedirs(save_dir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # FGSM curves
    for method, (eps_vals, fgsm_accs, _) in all_curves.items():
        eps_scaled = [e * 255 for e in eps_vals]
        ax1.plot(
            eps_scaled, fgsm_accs, marker="o", linewidth=2, label=method, markersize=4
        )

    ax1.set_xlabel("Epsilon (scaled by 255)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Accuracy (%)", fontsize=12, fontweight="bold")
    ax1.set_title("FGSM Robustness Curves", fontsize=14, fontweight="bold")
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax1.set_ylim([0, 100])

    # PGD curves
    for method, (eps_vals, _, pgd_accs) in all_curves.items():
        eps_scaled = [e * 255 for e in eps_vals]
        ax2.plot(
            eps_scaled, pgd_accs, marker="s", linewidth=2, label=method, markersize=4
        )

    ax2.set_xlabel("Epsilon (scaled by 255)", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Accuracy (%)", fontsize=12, fontweight="bold")
    ax2.set_title("PGD-20 Robustness Curves", fontsize=14, fontweight="bold")
    ax2.legend()
    ax2.grid(alpha=0.3)
    ax2.set_ylim([0, 100])

    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, "robustness_curves.png"), dpi=300, bbox_inches="tight"
    )
    print(f"\nSaved: {os.path.join(save_dir, 'robustness_curves.png')}")


def save_results_to_csv(all_results, save_dir="results"):
    """Salva i risultati in formato CSV"""
    os.makedirs(save_dir, exist_ok=True)

    df = pd.DataFrame(all_results).T
    df.to_csv(os.path.join(save_dir, "robustness_results.csv"))
    print(f"\nSaved: {os.path.join(save_dir, 'robustness_results.csv')}")

    # Stampa tabella formattata
    print("\n" + "=" * 80)
    print("ROBUSTNESS EVALUATION RESULTS")
    print("=" * 80)
    print(df.to_string())
    print("=" * 80)


def load_model(model_path, model_type, device, num_classes=10):
    """Carica un modello salvato con gestione di diversi formati di checkpoint"""
    if model_type == "scratch":
        model = ResNet(
            img_channels=3, num_layers=18, block=BasicBlock, num_classes=num_classes
        )
    elif model_type == "torchvision":
        model = build_model(pretrained=False, fine_tune=True, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Carica il checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Gestisci diversi formati di checkpoint
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            print("✓ Caricato da model_state_dict")
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
            print("✓ Caricato da state_dict")
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
            print("✓ Caricato da model")
        else:
            # Se è un dict ma non ha chiavi standard, prova a usarlo direttamente
            state_dict = checkpoint
            print("✓ Caricato direttamente dal dict")
    else:
        state_dict = checkpoint
        print("✓ Caricato direttamente i pesi")

    # Pulisci le chiavi (rimuovi 'module.' se presente da DataParallel)
    cleaned_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace("module.", "") if key.startswith("module.") else key
        cleaned_state_dict[new_key] = value

    # Carica i pesi (modalità non strict per gestire differenze)
    try:
        model.load_state_dict(cleaned_state_dict, strict=True)
        print("✓ Modello caricato in modalità strict")
    except RuntimeError as e:
        print(f"⚠ Warning: {e}")
        print("✓ Provando modalità non strict...")
        model.load_state_dict(cleaned_state_dict, strict=False)
        print("✓ Modello caricato in modalità non strict")

    model = model.to(device)
    model.eval()
    return model


def plot_autoattack_comparison(all_results, save_dir="results"):
    """Confronto specifico sui risultati AutoAttack"""
    os.makedirs(save_dir, exist_ok=True)

    methods = list(all_results.keys())

    # Metriche AutoAttack
    aa_metrics = [
        "pgd50_8/255",
        "autoattack_4/255",
        "autoattack_8/255",
        "autoattack_12/255",
    ]
    metric_labels = [
        "PGD-50\n(ε=8/255)",
        "AutoAttack\n(ε=4/255)",
        "AutoAttack\n(ε=8/255)",
        "AutoAttack\n(ε=12/255)",
    ]

    fig, ax = plt.subplots(figsize=(14, 6))

    x = np.arange(len(aa_metrics))
    width = 0.15

    for i, method in enumerate(methods):
        values = [all_results[method].get(m, 0) for m in aa_metrics]
        ax.bar(x + i * width, values, width, label=method, alpha=0.8)

    ax.set_xlabel("Attack Type", fontsize=12, fontweight="bold")
    ax.set_ylabel("Accuracy (%)", fontsize=12, fontweight="bold")
    ax.set_title("AutoAttack vs PGD Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks(x + width * (len(methods) - 1) / 2)
    ax.set_xticklabels(metric_labels)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, "autoattack_comparison.png"),
        dpi=300,
        bbox_inches="tight",
    )
    print(f"Saved: {os.path.join(save_dir, 'autoattack_comparison.png')}")


def plot_autoattack_strength_progression(all_results, save_dir="results"):
    """Progressione AutoAttack con epsilon diversi"""
    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    epsilons = [4 / 255, 8 / 255, 12 / 255]
    epsilon_labels = ["4/255", "8/255", "12/255"]

    for method in all_results.keys():
        accs = []
        for eps in epsilons:
            key = f"autoattack_{int(eps*255)}/255"
            acc = all_results[method].get(key, 0)
            accs.append(acc)

        ax.plot(epsilon_labels, accs, "o-", linewidth=3, markersize=8, label=method)

    ax.set_xlabel("Epsilon Value", fontsize=12, fontweight="bold")
    ax.set_ylabel("AutoAttack Accuracy (%)", fontsize=12, fontweight="bold")
    ax.set_title("AutoAttack Robustness vs Epsilon", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, "autoattack_epsilon_progression.png"),
        dpi=300,
        bbox_inches="tight",
    )
    print(f"Saved: {os.path.join(save_dir, 'autoattack_epsilon_progression.png')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model robustness")
    parser.add_argument(
        "--model_type",
        type=str,
        default="scratch",
        choices=["scratch", "torchvision"],
        help="Type of model architecture",
    )
    parser.add_argument(
        "--batch_size", type=int, default=256, help="Batch size for evaluation"
    )
    parser.add_argument(
        "--generate_curves",
        action="store_true",
        help="Generate robustness curves (slower)",
    )
    parser.add_argument(
        "--data_dir", type=str, default="./data", help="Directory for dataset"
    )

    args = parser.parse_args()

    # Setup
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Carica test data
    _, test_loader = get_data(batch_size=args.batch_size)

    # DEFINISCI I PERCORSI DEI MODELLI CON STRUTTURA FLESSIBILE
    model_paths = {
        "SGD": [
            "models/SGD/model.pth",
        ],
        "FGSM": [
            "models/fgsm/model.pth",
        ],
        "PGD": [
            "models/pgd/model.pth",
        ],
        "TRADES": [
            "models/trades/model.pth",
        ],
    }

    # Valuta tutti i modelli
    all_results = {}
    all_curves = {}

    for method_name, possible_paths in model_paths.items():
        model_path = None

        # Cerca il primo percorso esistente
        for path in possible_paths:
            if os.path.exists(path):
                model_path = path
                break

        if model_path is None:
            print(f"\n❌ Warning: No model found for {method_name}. Tried:")
            for path in possible_paths:
                print(f"   - {path}")
            continue

        print(f"\n{'='*60}")
        print(f"Loading model: {method_name}")
        print(f"Path: {model_path}")
        print(f"{'='*60}")

        try:
            model = load_model(model_path, args.model_type, device)

            # Valutazione completa
            results = comprehensive_evaluation(model, test_loader, device, method_name)
            all_results[method_name] = results

            # Genera curve di robustezza (opzionale)
            # if args.generate_curves:
            #     print(f"\nGenerating robustness curves for {method_name}...")
            #     eps_vals, fgsm_accs, pgd_accs = evaluate_robustness_curves(
            #         model, test_loader, device, method_name
            #     )
            #     all_curves[method_name] = (eps_vals, fgsm_accs, pgd_accs)

        except Exception as e:
            print(f"❌ Error evaluating {method_name}: {e}")
            import traceback

            traceback.print_exc()
            continue

    # Salva e visualizza risultati
    # Nel main, aggiungi queste visualizzazioni dopo la valutazione:
    if all_results:
        save_results_to_csv(all_results)
        plot_robustness_comparison(all_results)

        # Nuove visualizzazioni AutoAttack
        plot_autoattack_comparison(all_results)
        plot_autoattack_strength_progression(all_results)

    if all_curves:
        plot_robustness_curves(all_curves)

        print("\n" + "=" * 80)
        print("EVALUATION COMPLETE!")
        print("=" * 80)

        # Stampa riepilogo finale
        print("\nMODELS EVALUATED SUCCESSFULLY:")
        for method in all_results.keys():
            clean_acc = all_results[method]["clean"]
            fgsm_acc = all_results[method]["fgsm_8/255"]
            print(f"  {method}: Clean={clean_acc:.1f}%, FGSM={fgsm_acc:.1f}%")

        print("\nCheck the 'results' folder for detailed plots and CSV file.")
    else:
        print("\n❌ No models were successfully evaluated!")

        # Suggerimenti per risolvere il problema
        print("\nTROUBLESHOOTING:")
        print("1. Verifica che i percorsi dei modelli siano corretti")
        print("2. Controlla che i file .pth esistano")
        print("3. Verifica il formato dei checkpoint con:")
        print(
            "   import torch; checkpoint = torch.load('model.pth'); print(checkpoint.keys())"
        )
