"""
Dataset creation and data loading utilities for YOLOv3 training
"""

import os

from torch.utils.data import DataLoader, Subset

from src.data_loaders.cv import DummyDetectionDataset, ImprovedVOCDataset, PascalVOCDataset


def _limit_batches(dataloader, num_batches):
    """
    Limit a dataloader to a specific number of batches

    Args:
        dataloader: The dataloader to limit
        num_batches: Maximum number of batches to yield

    Returns:
        A generator that yields limited batches
    """

    class LimitedBatchSampler:
        def __init__(self, dataloader, num_batches):
            self.dataloader = dataloader
            self.num_batches = num_batches

        def __iter__(self):
            counter = 0
            for batch in self.dataloader:
                if counter >= self.num_batches:
                    break
                yield batch
                counter += 1

        def __len__(self):
            return min(self.num_batches, len(self.dataloader))

    return LimitedBatchSampler(dataloader, num_batches)


def create_datasets(args):
    """
    Create datasets and dataloaders

    Args:
        args: Command line arguments

    Returns:
        tuple: train_loader, val_loader, test_loader (None if not requested)
    """
    # Check if we should use dummy dataset (for debugging or if VOC isn't available)
    use_dummy_data = args.use_dummy_data or args.fast_dev_run

    if args.dataset == "voc" and not use_dummy_data:
        # Check if VOC dataset exists at data_dir path
        voc_path = os.path.join(args.data_dir, f"VOC{args.year.split(',')[0]}")
        if not os.path.exists(voc_path):
            print(f"VOC dataset not found at {voc_path}")
            print("Falling back to dummy dataset for debugging...")
            use_dummy_data = True

    # Use dummy dataset for debugging
    if use_dummy_data:
        print("Using dummy dataset for debugging")
        num_samples = 10 if args.fast_dev_run else 100

        train_dataset = DummyDetectionDataset(num_samples=num_samples)
        val_dataset = DummyDetectionDataset(num_samples=num_samples // 2)
        test_dataset = DummyDetectionDataset(num_samples=num_samples // 2)

        train_collate_fn = train_dataset.collate_fn
        val_collate_fn = val_dataset.collate_fn
        test_collate_fn = test_dataset.collate_fn
    elif args.dataset == "voc":
        # Parse years into a list
        years = args.year.split(",")

        # Get debug parameters
        is_debug = (
            args.fast_dev_run
            or args.max_images is not None
            or args.subset_percent is not None
            or args.debug_mode
        )

        # Use improved dataset with augmentations for training
        train_dataset = ImprovedVOCDataset(
            years=years,
            split_file=f"{args.train_split}.txt",
            sample_pct=args.subset_percent,
            data_dir=args.data_dir,
            # Enable advanced augmentations
            use_mosaic=not is_debug,  # Disable in debug mode
            mosaic_prob=0.5,
            use_hsv=True,
            hsv_prob=0.5,
        )
        train_collate_fn = train_dataset.collate_fn

        # Create validation dataset (standard version without augmentations)
        val_dataset = PascalVOCDataset(
            years=years,
            split_file=f"{args.val_split}.txt",
            sample_pct=args.subset_percent,
            data_dir=args.data_dir,
        )
        val_collate_fn = val_dataset.collate_fn

        # Create test dataset if specified
        test_dataset = None
        test_collate_fn = None
        if args.test_split:
            test_dataset = PascalVOCDataset(
                years=years,
                split_file=f"{args.test_split}.txt",
                sample_pct=args.subset_percent,
                data_dir=args.data_dir,
            )
            test_collate_fn = test_dataset.collate_fn

        # Create class-specific dataset if required
        if args.class_name:
            class_train_file = f"{args.class_name}_{args.train_split}.txt"
            class_val_file = f"{args.class_name}_{args.val_split}.txt"

            # Create class-specific train dataset with improved augmentations
            train_dataset = ImprovedVOCDataset(
                years=years,
                split_file=class_train_file,
                sample_pct=args.subset_percent,
                data_dir=args.data_dir,
                # Enable advanced augmentations
                use_mosaic=not is_debug,  # Disable in debug mode
                mosaic_prob=0.5,
                use_hsv=True,
                hsv_prob=0.5,
            )
            train_collate_fn = train_dataset.collate_fn

            # Create class-specific validation dataset
            val_dataset = PascalVOCDataset(
                years=years,
                split_file=class_val_file,
                sample_pct=args.subset_percent,
                data_dir=args.data_dir,
            )
            val_collate_fn = val_dataset.collate_fn

            # Create class-specific test dataset if specified
            if args.test_split and args.class_name:
                class_test_file = f"{args.class_name}_{args.test_split}.txt"
                test_dataset = PascalVOCDataset(
                    years=years,
                    split_file=class_test_file,
                    sample_pct=args.subset_percent,
                    data_dir=args.data_dir,
                )
                test_collate_fn = test_dataset.collate_fn
    else:
        raise NotImplementedError(f"Dataset {args.dataset} is not implemented yet")

    # Handle debug mode options
    if args.max_images is not None:
        print(f"DEBUG MODE: Limiting each dataset to {args.max_images} images")
        # Create a subset for each dataset
        train_indices = list(range(min(args.max_images, len(train_dataset))))
        val_indices = list(range(min(args.max_images, len(val_dataset))))

        train_dataset = Subset(train_dataset, train_indices)
        val_dataset = Subset(val_dataset, val_indices)

        if test_dataset:
            test_indices = list(range(min(args.max_images, len(test_dataset))))
            test_dataset = Subset(test_dataset, test_indices)

    # Set batch size for fast dev run
    batch_size = args.batch_size
    if args.fast_dev_run:
        print("DEBUG MODE: Fast dev run with minimal batches")
        # Use extremely small batch size for fast dev run
        batch_size = min(2, batch_size)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=train_collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=val_collate_fn,
    )

    # Create test loader if test dataset exists
    test_loader = None
    if test_dataset:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=args.workers,
            collate_fn=test_collate_fn,
        )

    # Print dataset sizes
    train_size = len(train_dataset)
    val_size = len(val_dataset)
    print(f"Created datasets: Train ({train_size}), Val ({val_size})")
    if test_loader:
        test_size = len(test_dataset)
        print(f"Test ({test_size})")

    # Further limit batches for fast dev run
    if args.fast_dev_run:
        # Wrap loaders to limit to just a few batches
        train_loader = _limit_batches(train_loader, 3)
        val_loader = _limit_batches(val_loader, 2)
        if test_loader:
            test_loader = _limit_batches(test_loader, 1)

    return train_loader, val_loader, test_loader
