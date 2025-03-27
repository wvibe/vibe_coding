"""
Visualization utilities for YOLOv3 training and evaluation
"""

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import torch

import wandb


def visualize_predictions(model, images, targets, detections, dataloader, img_idx):
    """
    Create visualization of predictions vs ground truth

    Args:
        model: YOLOv3 model
        images: Batch of images
        targets: Ground truth targets
        detections: Model predictions
        dataloader: DataLoader for validation data
        img_idx: Index of image in batch

    Returns:
        matplotlib figure: Visualization image
    """
    # Prepare image for visualization
    img = images[img_idx].cpu()
    # Denormalize image
    img = img * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = img + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    img = (img * 255).byte().permute(1, 2, 0).numpy()

    # Create figure for visualization
    plt.figure(figsize=(10, 10))
    plt.imshow(img)

    # Draw ground truth boxes in green
    for box, label in zip(targets["boxes"][img_idx], targets["labels"][img_idx]):
        x1, y1, x2, y2 = box.cpu().numpy()
        rect = patches.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            linewidth=2,
            edgecolor="g",
            facecolor="none",
        )
        plt.gca().add_patch(rect)
        plt.text(
            x1,
            y1 - 5,
            f"GT: {dataloader.dataset.class_names[label]}",
            color="g",
        )

    # Draw predicted boxes in red
    for det in detections[img_idx]:
        if det.size(0) > 0:  # If there are detections
            x1, y1, x2, y2, conf, cls_id = det
            rect = patches.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=2,
                edgecolor="r",
                facecolor="none",
            )
            plt.gca().add_patch(rect)
            # Get class name and format prediction text
            class_name = dataloader.dataset.class_names[int(cls_id)]
            pred_text = f"Pred: {class_name} {conf:.2f}"

            plt.text(
                x1,
                y2 + 15,
                pred_text,
                color="r",
            )

    plt.axis("off")
    fig = plt
    return fig


def generate_validation_visualizations(
    model, dataloader, device, max_batches=4, max_images_per_batch=2
):
    """
    Generate visualizations for validation predictions

    Args:
        model: YOLOv3 model
        dataloader: DataLoader for validation data
        device: Device to use
        max_batches: Maximum number of batches to visualize
        max_images_per_batch: Maximum number of images per batch to visualize

    Returns:
        list: Visualizations as wandb.Image objects
    """
    validation_images = []

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= max_batches:
            break

        # Get batch data
        images = batch["images"].to(device)
        targets = {
            "boxes": [boxes.to(device) for boxes in batch["boxes"]],
            "labels": [labels.to(device) for labels in batch["labels"]],
        }

        # Get predictions
        detections = model.predict(images)

        # Process each image in batch
        for img_idx in range(min(max_images_per_batch, len(images))):
            fig = visualize_predictions(model, images, targets, detections, dataloader, img_idx)
            validation_images.append(wandb.Image(fig))
            plt.close()

    return validation_images
