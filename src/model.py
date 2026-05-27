from __future__ import annotations

from typing import Literal

from torch import nn


def _freeze_all_except(module: nn.Module, trainable_prefixes: tuple[str, ...]) -> None:
    for name, param in module.named_parameters():
        param.requires_grad = any(name.startswith(prefix) for prefix in trainable_prefixes)


def build_vit_model(
    model_name: Literal["vit_b_16", "vit_b_32"],
    num_classes: int,
    dropout: float = 0.2,
    pretrained: bool = True,
    train_mode: Literal["head_only", "finetune"] = "head_only",
) -> nn.Module:
    try:
        from torchvision.models import ViT_B_16_Weights, ViT_B_32_Weights, vit_b_16, vit_b_32
    except Exception as exc:
        raise ImportError(
            "Vision Transformer cần torchvision cài đúng cặp phiên bản với torch. "
            "Hãy kiểm tra lại `pip install torch torchvision`."
        ) from exc

    if model_name == "vit_b_16":
        weights = ViT_B_16_Weights.DEFAULT if pretrained else None
        model = vit_b_16(weights=weights)
    elif model_name == "vit_b_32":
        weights = ViT_B_32_Weights.DEFAULT if pretrained else None
        model = vit_b_32(weights=weights)
    else:
        raise ValueError(f"Unsupported ViT model: {model_name}")

    in_features = model.heads.head.in_features
    model.heads.head = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_features, num_classes),
    )

    if train_mode == "head_only":
        _freeze_all_except(model, ("heads",))
    elif train_mode == "finetune":
        for param in model.parameters():
            param.requires_grad = True
    else:
        raise ValueError("train_mode must be 'head_only' or 'finetune'.")

    return model


def build_resnet_model(
    model_name: Literal["resnet18", "resnet50"],
    num_classes: int,
    dropout: float = 0.2,
    pretrained: bool = True,
    train_mode: Literal["head_only", "finetune"] = "head_only",
) -> nn.Module:
    try:
        from torchvision.models import ResNet18_Weights, ResNet50_Weights, resnet18, resnet50
    except Exception as exc:
        raise ImportError(
            "ResNet cần torchvision cài đúng cặp phiên bản với torch. "
            "Hãy kiểm tra lại `pip install torch torchvision`."
        ) from exc

    if model_name == "resnet18":
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = resnet18(weights=weights)
    elif model_name == "resnet50":
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        model = resnet50(weights=weights)
    else:
        raise ValueError(f"Unsupported ResNet model: {model_name}")

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_features, num_classes),
    )

    if train_mode == "head_only":
        _freeze_all_except(model, ("fc",))
    elif train_mode == "finetune":
        for param in model.parameters():
            param.requires_grad = True
    else:
        raise ValueError("train_mode must be 'head_only' or 'finetune'.")

    return model


def build_model(
    model_name: str,
    train_mode: str,
    num_classes: int,
    dropout: float = 0.2,
    pretrained: bool = True,
) -> nn.Module:
    if model_name in {"vit_b_16", "vit_b_32"}:
        return build_vit_model(
            model_name=model_name,
            num_classes=num_classes,
            dropout=dropout,
            pretrained=pretrained,
            train_mode=train_mode,
        )

    if model_name in {"resnet18", "resnet50"}:
        return build_resnet_model(
            model_name=model_name,
            num_classes=num_classes,
            dropout=dropout,
            pretrained=pretrained,
            train_mode=train_mode,
        )

    raise ValueError(f"Unsupported model_name: {model_name}")


def count_parameters(model: nn.Module) -> tuple[int, int, float]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ratio = trainable / total if total > 0 else 0.0
    return total, trainable, ratio
