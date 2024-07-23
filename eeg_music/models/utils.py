import torch


def transform_labels_to_one_hot(
    labels: torch.Tensor,
    bins: int = 5,
) -> torch.Tensor:
    batch, dimensions = labels.shape
    labels = (labels - 1).to(torch.long)  # batch, dimensions

    transformed_labels = torch.zeros(batch, dimensions, bins).to(
        device=labels.device
    )  # batch, dimensions, likert
    indices = labels.unsqueeze(-1)
    return transformed_labels.scatter_(2, indices, 1)


def transform_labels_classes(
    labels: torch.Tensor,
) -> torch.Tensor:
    """
    transform labels to 3 classes
    """
    labels = (labels - 1).to(torch.long)  # batch, dimensions
    labels_mapped = torch.empty_like(labels)
    labels_mapped = torch.where(
        labels == 0, torch.tensor(0, device=labels.device), labels_mapped
    )
    labels_mapped = torch.where(
        labels == 1, torch.tensor(0, device=labels.device), labels_mapped
    )
    labels_mapped = torch.where(
        labels == 2, torch.tensor(1, device=labels.device), labels_mapped
    )
    labels_mapped = torch.where(
        labels == 3, torch.tensor(1, device=labels.device), labels_mapped
    )
    labels_mapped = torch.where(
        labels == 4, torch.tensor(2, device=labels.device), labels_mapped
    )
    labels = labels_mapped
    return labels
