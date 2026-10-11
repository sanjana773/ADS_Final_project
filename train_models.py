import os
import copy

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset

from sklearn.metrics import (
    classification_report,
    confusion_matrix
)

import matplotlib.pyplot as plt


# ============================================================
# 1. PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Leaf dataset
LEAF_DATA_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "leaf"
)

# Fruit dataset
FRUIT_DATA_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "fruit"
)

# Model folder
MODEL_DIR = os.path.join(
    BASE_DIR,
    "model"
)

os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# 2. TRAINING SETTINGS
# ============================================================

BATCH_SIZE = 32

# ------------------------------------------------------------
# FIRST TEST:
# Keep this as 1.
#
# After confirming everything works:
# change to 10
# ------------------------------------------------------------
EPOCHS = 1

LEARNING_RATE = 0.0001

# 70% train
# 15% validation
# 15% test
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

IMAGE_SIZE = 224


# ============================================================
# 3. DEVICE
# ============================================================

if torch.cuda.is_available():

    device = torch.device("cuda")

    print("\nCUDA GPU detected.")
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

else:

    device = torch.device("cpu")

    print("\nCUDA GPU not detected.")
    print("Using CPU.")

print("Device:", device)


# ============================================================
# 4. IMAGE TRANSFORMS
# ============================================================

# Training images use augmentation
train_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.RandomHorizontalFlip(),

    transforms.RandomRotation(10),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# Validation and test images
# DO NOT use random augmentation here
val_test_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# 5. CREATE TRAIN / VALIDATION / TEST SPLIT
# ============================================================

def create_datasets(data_dir):

    print("\nLoading dataset:")
    print(data_dir)

    # --------------------------------------------------------
    # Dataset WITHOUT augmentation
    # Used to create fixed indexes
    # --------------------------------------------------------

    base_dataset = datasets.ImageFolder(
        root=data_dir,
        transform=None
    )

    class_names = base_dataset.classes

    total_size = len(base_dataset)

    print("\nClasses:")
    print("-" * 60)

    for index, class_name in enumerate(class_names):

        print(
            f"{index} -> {class_name}"
        )

    print("-" * 60)

    print(
        "Number of classes:",
        len(class_names)
    )

    print(
        "Total images:",
        total_size
    )

    # --------------------------------------------------------
    # Validate dataset
    # --------------------------------------------------------

    if len(class_names) < 2:

        raise ValueError(
            "At least 2 classes are required."
        )

    if total_size < 10:

        print(
            "\nWARNING: Very few images found."
        )

    # --------------------------------------------------------
    # Calculate split sizes
    # --------------------------------------------------------

    train_size = int(
        TRAIN_RATIO * total_size
    )

    val_size = int(
        VAL_RATIO * total_size
    )

    test_size = (
        total_size
        - train_size
        - val_size
    )

    print("\nDataset split:")

    print(
        "Training images:",
        train_size
    )

    print(
        "Validation images:",
        val_size
    )

    print(
        "Testing images:",
        test_size
    )

    # --------------------------------------------------------
    # Create random indexes
    # --------------------------------------------------------

    generator = torch.Generator().manual_seed(42)

    all_indices = torch.randperm(
        total_size,
        generator=generator
    ).tolist()

    train_indices = all_indices[
        :train_size
    ]

    val_indices = all_indices[
        train_size:
        train_size + val_size
    ]

    test_indices = all_indices[
        train_size + val_size:
    ]

    # --------------------------------------------------------
    # Create separate datasets
    # --------------------------------------------------------

    train_full = datasets.ImageFolder(
        root=data_dir,
        transform=train_transform
    )

    val_full = datasets.ImageFolder(
        root=data_dir,
        transform=val_test_transform
    )

    test_full = datasets.ImageFolder(
        root=data_dir,
        transform=val_test_transform
    )

    train_dataset = Subset(
        train_full,
        train_indices
    )

    val_dataset = Subset(
        val_full,
        val_indices
    )

    test_dataset = Subset(
        test_full,
        test_indices
    )

    return (
        train_dataset,
        val_dataset,
        test_dataset,
        class_names
    )


# ============================================================
# 6. EVALUATION FUNCTION
# ============================================================

def evaluate_model(
    model,
    data_loader,
    class_names,
    dataset_name
):

    print("\n")
    print("=" * 70)
    print(
        f"{dataset_name.upper()} EVALUATION"
    )
    print("=" * 70)

    model.eval()

    all_predictions = []
    all_labels = []

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in data_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, predictions = torch.max(
                outputs,
                1
            )

            total += labels.size(0)

            correct += (
                predictions == labels
            ).sum().item()

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

    # --------------------------------------------------------
    # Accuracy
    # --------------------------------------------------------

    accuracy = (
        100.0 * correct / total
        if total > 0
        else 0.0
    )

    print(
        f"\n{dataset_name} Accuracy: "
        f"{accuracy:.2f}%"
    )

    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    print("\nClassification Report:")
    print("-" * 70)

    report = classification_report(
        all_labels,
        all_predictions,
        labels=list(
            range(len(class_names))
        ),
        target_names=class_names,
        zero_division=0
    )

    print(report)

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    cm = confusion_matrix(
        all_labels,
        all_predictions,
        labels=list(
            range(len(class_names))
        )
    )

    print("Confusion Matrix:")
    print(cm)

    # --------------------------------------------------------
    # Save confusion matrix
    # --------------------------------------------------------

    plt.figure(
        figsize=(12, 10)
    )

    plt.imshow(cm)

    plt.title(
        f"{dataset_name} Confusion Matrix"
    )

    plt.colorbar()

    plt.xticks(
        range(len(class_names)),
        class_names,
        rotation=90
    )

    plt.yticks(
        range(len(class_names)),
        class_names
    )

    plt.xlabel("Predicted Label")

    plt.ylabel("True Label")

    plt.tight_layout()

    confusion_matrix_path = os.path.join(
        MODEL_DIR,
        f"{dataset_name.lower().replace(' ', '_')}_confusion_matrix.png"
    )

    plt.savefig(
        confusion_matrix_path
    )

    plt.close()

    print(
        "\nConfusion matrix saved at:"
    )

    print(
        confusion_matrix_path
    )

    return accuracy


# ============================================================
# 7. TRAINING FUNCTION
# ============================================================

def train_model(
    data_dir,
    model_filename,
    epochs=EPOCHS
):

    print("\n")
    print("=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)

    print("\nDataset:")
    print(data_dir)

    print("\nModel:")
    print(model_filename)

    # --------------------------------------------------------
    # Check dataset
    # --------------------------------------------------------

    if not os.path.exists(data_dir):

        print(
            "\nERROR: Dataset folder not found!"
        )

        print(data_dir)

        return False

    # --------------------------------------------------------
    # Load datasets
    # --------------------------------------------------------

    try:

        (
            train_dataset,
            val_dataset,
            test_dataset,
            class_names
        ) = create_datasets(
            data_dir
        )

    except Exception as e:

        print(
            "\nERROR while loading dataset:"
        )

        print(e)

        return False

    num_classes = len(
        class_names
    )

    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    print("\n")
    print("Batch size:", BATCH_SIZE)

    print(
        "Training batches:",
        len(train_loader)
    )

    print(
        "Validation batches:",
        len(val_loader)
    )

    print(
        "Testing batches:",
        len(test_loader)
    )

    # ========================================================
    # LOAD RESNET18
    # ========================================================

    print("\nLoading ResNet18...")

    try:

        weights = (
            models.ResNet18_Weights.DEFAULT
        )

        model = models.resnet18(
            weights=weights
        )

        print(
            "Pretrained ResNet18 loaded."
        )

    except Exception as e:

        print(
            "\nWARNING:"
        )

        print(
            "Could not load pretrained weights."
        )

        print(e)

        print(
            "\nUsing ResNet18 without pretrained weights."
        )

        model = models.resnet18(
            weights=None
        )

    # --------------------------------------------------------
    # Replace final layer
    # --------------------------------------------------------

    model.fc = nn.Linear(
        model.fc.in_features,
        num_classes
    )

    model = model.to(device)

    # ========================================================
    # LOSS
    # ========================================================

    criterion = nn.CrossEntropyLoss()

    # ========================================================
    # OPTIMIZER
    # ========================================================

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # ========================================================
    # BEST MODEL
    # ========================================================

    best_val_accuracy = 0.0

    best_model_weights = copy.deepcopy(
        model.state_dict()
    )

    # ========================================================
    # TRAINING LOOP
    # ========================================================

    print("\n")
    print("=" * 70)
    print("TRAINING STARTED")
    print("=" * 70)

    for epoch in range(epochs):

        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        model.train()

        running_loss = 0.0

        train_correct = 0
        train_total = 0

        print(
            f"\nEpoch [{epoch + 1}/{epochs}]"
        )

        for batch_index, (
            images,
            labels
        ) in enumerate(train_loader):

            images = images.to(device)
            labels = labels.to(device)

            # Clear gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)

            # Loss
            loss = criterion(
                outputs,
                labels
            )

            # Backpropagation
            loss.backward()

            # Update model
            optimizer.step()

            # Statistics
            running_loss += loss.item()

            _, predicted = torch.max(
                outputs,
                1
            )

            train_total += labels.size(0)

            train_correct += (
                predicted == labels
            ).sum().item()

            # Progress
            if (
                batch_index + 1
            ) % 10 == 0 or (
                batch_index + 1
            ) == len(train_loader):

                print(
                    f"  Batch "
                    f"{batch_index + 1}/"
                    f"{len(train_loader)} "
                    f"- Loss: "
                    f"{loss.item():.4f}"
                )

        # ----------------------------------------------------
        # Training statistics
        # ----------------------------------------------------

        train_accuracy = (
            100.0
            * train_correct
            / train_total
            if train_total > 0
            else 0.0
        )

        average_train_loss = (
            running_loss
            / len(train_loader)
            if len(train_loader) > 0
            else 0.0
        )

        # ====================================================
        # VALIDATION
        # ====================================================

        model.eval()

        val_correct = 0
        val_total = 0

        val_running_loss = 0.0

        with torch.no_grad():

            for images, labels in val_loader:

                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)

                loss = criterion(
                    outputs,
                    labels
                )

                val_running_loss += (
                    loss.item()
                )

                _, predicted = torch.max(
                    outputs,
                    1
                )

                val_total += labels.size(0)

                val_correct += (
                    predicted == labels
                ).sum().item()

        # ----------------------------------------------------
        # Validation statistics
        # ----------------------------------------------------

        val_accuracy = (
            100.0
            * val_correct
            / val_total
            if val_total > 0
            else 0.0
        )

        average_val_loss = (
            val_running_loss
            / len(val_loader)
            if len(val_loader) > 0
            else 0.0
        )

        # ----------------------------------------------------
        # Print epoch result
        # ----------------------------------------------------

        print("\n" + "-" * 70)

        print(
            f"Epoch [{epoch + 1}/{epochs}] completed"
        )

        print(
            f"Train Loss: "
            f"{average_train_loss:.4f}"
        )

        print(
            f"Train Accuracy: "
            f"{train_accuracy:.2f}%"
        )

        print(
            f"Validation Loss: "
            f"{average_val_loss:.4f}"
        )

        print(
            f"Validation Accuracy: "
            f"{val_accuracy:.2f}%"
        )

        print("-" * 70)

        # ----------------------------------------------------
        # Save best model
        # ----------------------------------------------------

        if val_accuracy > best_val_accuracy:

            best_val_accuracy = val_accuracy

            best_model_weights = copy.deepcopy(
                model.state_dict()
            )

            print(
                "\nNEW BEST MODEL!"
            )

            print(
                f"Best validation accuracy: "
                f"{best_val_accuracy:.2f}%"
            )

    # ========================================================
    # LOAD BEST MODEL
    # ========================================================

    print("\n")
    print("=" * 70)
    print("LOADING BEST MODEL")
    print("=" * 70)

    model.load_state_dict(
        best_model_weights
    )

    # ========================================================
    # TEST MODEL
    # ========================================================

    test_accuracy = evaluate_model(
        model=model,
        data_loader=test_loader,
        class_names=class_names,
        dataset_name="Test"
    )

    # ========================================================
    # SAVE MODEL
    # ========================================================

    model_path = os.path.join(
        MODEL_DIR,
        model_filename
    )

    checkpoint = {

        "model_state_dict":
            model.state_dict(),

        "class_names":
            class_names,

        "num_classes":
            num_classes,

        "image_size":
            IMAGE_SIZE,

        "best_val_accuracy":
            best_val_accuracy,

        "test_accuracy":
            test_accuracy
    }

    torch.save(
        checkpoint,
        model_path
    )

    # ========================================================
    # TRAINING COMPLETE
    # ========================================================

    print("\n")
    print("=" * 70)
    print("MODEL TRAINING COMPLETED")
    print("=" * 70)

    print("\nModel saved at:")
    print(model_path)

    print(
        f"\nBest validation accuracy: "
        f"{best_val_accuracy:.2f}%"
    )

    print(
        f"Test accuracy: "
        f"{test_accuracy:.2f}%"
    )

    print("\nClasses saved:")

    for index, class_name in enumerate(
        class_names
    ):

        print(
            f"{index} -> {class_name}"
        )

    print("=" * 70)

    return True


# ============================================================
# 8. MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    print("\n")
    print("=" * 70)
    print("PLANORA AI - MODEL TRAINING")
    print("=" * 70)

    print("\nProject directory:")
    print(BASE_DIR)

    print("\nModel directory:")
    print(MODEL_DIR)

    # ========================================================
    # TRAIN LEAF MODEL
    # ========================================================

    leaf_success = train_model(

        data_dir=LEAF_DATA_DIR,

        model_filename=
        "tomato_disease_model.pth",

        epochs=EPOCHS
    )

    # ========================================================
    # TRAIN FRUIT MODEL
    # ========================================================

    fruit_success = train_model(

        data_dir=FRUIT_DATA_DIR,

        model_filename=
        "fruit_disease_model.pth",

        epochs=EPOCHS
    )

    # ========================================================
    # FINAL RESULT
    # ========================================================

    print("\n")
    print("=" * 70)
    print("ALL TRAINING FINISHED")
    print("=" * 70)

    print("\nLeaf model:")

    print(
        "SUCCESS"
        if leaf_success
        else "FAILED"
    )

    print("\nFruit model:")

    print(
        "SUCCESS"
        if fruit_success
        else "FAILED"
    )

    print("\nModel files should be located here:")

    print(
        os.path.join(
            MODEL_DIR,
            "tomato_disease_model.pth"
        )
    )

    print(
        os.path.join(
            MODEL_DIR,
            "fruit_disease_model.pth"
        )
    )

    print("\n")
    print("=" * 70)
    print("DONE")
    print("=" * 70)