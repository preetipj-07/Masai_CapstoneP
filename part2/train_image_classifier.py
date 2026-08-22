import os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from PIL import Image
import warnings
warnings.filterwarnings("ignore")

import builtins
_orig_print = builtins.print
def _fprint(*a, **k):
    k["flush"] = True
    _orig_print(*a, **k)
builtins.print = _fprint

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]
NUM_CLASSES = 10
INPUT_SIZE = 64
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
BATCH_SIZE = 256
LR_HEAD = 1e-3
EPOCHS_HEAD = 10
LR_FINETUNE = 1e-4
EPOCHS_FINETUNE = 2

print("=" * 60)
print("TASK 1: Load Fashion-MNIST")
print("=" * 60)

transform_base = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

train_full = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform_base)
test_dataset = datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform_base)

targets = np.array(train_full.targets)
train_idx, val_idx = train_test_split(range(len(train_full)), test_size=0.1, stratify=targets, random_state=42)
train_dataset = Subset(train_full, train_idx)
val_dataset = Subset(train_full, val_idx)

print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")
print(f"Input size: {INPUT_SIZE}x{INPUT_SIZE} RGB (grayscale replicated to 3 channels)")
print(f"Normalization: ImageNet mean={IMAGENET_MEAN}, std={IMAGENET_STD}")

print("\n" + "=" * 60)
print("TASK 2 & 3: Build ResNet-18 + Cache Features")
print("=" * 60)

resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
for param in resnet.parameters():
    param.requires_grad = False

num_features = resnet.fc.in_features
resnet.fc = nn.Linear(num_features, NUM_CLASSES)
resnet = resnet.to(DEVICE)

total_p = sum(p.numel() for p in resnet.parameters())
trainable_p = sum(p.numel() for p in resnet.parameters() if p.requires_grad)
print(f"Total params: {total_p:,} | Trainable (head only): {trainable_p:,}")


def extract_features(model, dataset, desc=""):
    model.eval()
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    feats, labels = [], []
    t0 = time.time()
    with torch.no_grad():
        for i, (images, lbls) in enumerate(loader):
            images = images.to(DEVICE)
            x = model.conv1(images)
            x = model.bn1(x)
            x = model.relu(x)
            x = model.maxpool(x)
            x = model.layer1(x)
            x = model.layer2(x)
            x = model.layer3(x)
            x = model.layer4(x)
            x = model.avgpool(x)
            x = torch.flatten(x, 1)
            feats.append(x.cpu().numpy())
            labels.append(lbls.numpy())
            if (i + 1) % 50 == 0:
                print(f"  {desc} batch {i+1}... ({time.time()-t0:.1f}s)")
    return np.concatenate(feats), np.concatenate(labels)


t0 = time.time()
print("Extracting train features...")
train_features, train_labels = extract_features(resnet, train_dataset, "train")
print(f"  Done in {time.time()-t0:.1f}s")

t0 = time.time()
print("Extracting val features...")
val_features, val_labels = extract_features(resnet, val_dataset, "val")
print(f"  Done in {time.time()-t0:.1f}s")

t0 = time.time()
print("Extracting test features...")
test_features, test_labels = extract_features(resnet, test_dataset, "test")
print(f"  Done in {time.time()-t0:.1f}s")

print(f"Feature shapes: train={train_features.shape}, val={val_features.shape}, test={test_features.shape}")

print("\nTraining classifier head on cached features...")


class ClassifierHead(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)
    def forward(self, x):
        return self.fc(x)


head = ClassifierHead(train_features.shape[1], NUM_CLASSES).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(head.parameters(), lr=LR_HEAD)

train_t = torch.tensor(train_features, dtype=torch.float32)
train_l = torch.tensor(train_labels, dtype=torch.long)
val_t = torch.tensor(val_features, dtype=torch.float32)
val_l = torch.tensor(val_labels, dtype=torch.long)

for epoch in range(EPOCHS_HEAD):
    head.train()
    optimizer.zero_grad()
    out = head(train_t.to(DEVICE))
    loss = criterion(out, train_l.to(DEVICE))
    loss.backward()
    optimizer.step()
    head.eval()
    with torch.no_grad():
        vpred = head(val_t.to(DEVICE)).argmax(1).cpu().numpy()
        vacc = accuracy_score(val_labels, vpred)
    print(f"  Epoch {epoch+1}/{EPOCHS_HEAD}  loss={loss.item():.4f}  val_acc={vacc:.4f}")

feature_extraction_val_acc = vacc
print(f"\nFeature extraction validation accuracy: {feature_extraction_val_acc:.4f}")

print("\n" + "=" * 60)
print("TASK 4: Fine-tuning Decision")
print("=" * 60)

fine_tuned = False
fine_tune_val_acc = feature_extraction_val_acc

if feature_extraction_val_acc < 0.80:
    print(f"Feature extraction val acc {feature_extraction_val_acc:.4f} < 0.80 — proceeding with fine-tuning...")
    fine_tuned = True

    for param in resnet.layer3.parameters():
        param.requires_grad = True
    for param in resnet.layer4.parameters():
        param.requires_grad = True
    resnet.fc = nn.Linear(num_features, NUM_CLASSES).to(DEVICE)

    train_loader_ft = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader_ft = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, resnet.parameters()), lr=LR_FINETUNE)

    for epoch in range(EPOCHS_FINETUNE):
        resnet.train()
        running_loss, correct, total = 0.0, 0, 0
        t_ep = time.time()
        for images, labels in train_loader_ft:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer_ft.zero_grad()
            outputs = resnet(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer_ft.step()
            running_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
        resnet.eval()
        vc, vt = 0, 0
        with torch.no_grad():
            for images, labels in val_loader_ft:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                vc += (resnet(images).argmax(1) == labels).sum().item()
                vt += labels.size(0)
        fine_tune_val_acc = vc / vt
        print(f"  FT Epoch {epoch+1}/{EPOCHS_FINETUNE}  loss={running_loss/total:.4f}  train_acc={correct/total:.4f}  val_acc={fine_tune_val_acc:.4f}  ({time.time()-t_ep:.1f}s)")

    print(f"\nBefore fine-tuning: {feature_extraction_val_acc:.4f}")
    print(f"After fine-tuning: {fine_tune_val_acc:.4f}")
else:
    print(f"Feature extraction val accuracy {feature_extraction_val_acc:.4f} >= 0.80 — fine-tuning NOT required.")
    print("Before/after: N/A (skipped)")

print("\n" + "=" * 60)
print("TASK 5: Test-Set Evaluation")
print("=" * 60)

if not fine_tuned:
    head.eval()
    with torch.no_grad():
        test_preds = head(torch.tensor(test_features, dtype=torch.float32).to(DEVICE)).argmax(1).cpu().numpy()
else:
    resnet.eval()
    test_loader_eval = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    all_preds = []
    with torch.no_grad():
        for images, _ in test_loader_eval:
            all_preds.append(resnet(images.to(DEVICE)).argmax(1).cpu().numpy())
    test_preds = np.concatenate(all_preds)

test_acc = accuracy_score(test_labels, test_preds)
print(f"\nTest accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")

cm = confusion_matrix(test_labels, test_preds)
print("\nConfusion Matrix:")
for i, row in enumerate(cm):
    print(f"  {CLASS_NAMES[i]:15s}  {row}")

print("\nPer-class Precision/Recall:")
print(classification_report(test_labels, test_preds, target_names=CLASS_NAMES))

print("=" * 60)
print("TASK 6: Confusion Pattern Analysis")
print("=" * 60)

cm_no_diag = cm.copy()
np.fill_diagonal(cm_no_diag, 0)
top_pairs = []
for i in range(NUM_CLASSES):
    for j in range(NUM_CLASSES):
        if i != j and cm_no_diag[i][j] > 0:
            top_pairs.append((CLASS_NAMES[i], CLASS_NAMES[j], int(cm_no_diag[i][j])))
top_pairs.sort(key=lambda x: x[2], reverse=True)

print("\nTop confusion pairs:")
for true_cls, pred_cls, count in top_pairs[:5]:
    print(f"  {true_cls:15s} -> {pred_cls:15s}: {count} misclassifications")

print("\nConfusion pair analysis:")
for true_cls, pred_cls, count in top_pairs[:2]:
    print(f"\n  {true_cls} vs {pred_cls} ({count} errors):")
    if true_cls in ["Shirt", "Pullover", "Coat", "T-shirt/top"] and pred_cls in ["Shirt", "Pullover", "Coat", "T-shirt/top"]:
        print(f"    Both are upper-body garments with similar silhouettes — long sleeves, similar outline.")
        print(f"    The model struggles to distinguish fabric textures and collar/hem details in low-res images.")
    elif true_cls in ["Sandal", "Sneaker", "Ankle boot"] and pred_cls in ["Sandal", "Sneaker", "Ankle boot"]:
        print(f"    All are footwear — similar overall shape, with differences mainly in ankle coverage and sole style.")
        print(f"    At low resolution, these fine-grained structural differences are harder to capture.")
    else:
        print(f"    These categories share visual similarities that confuse the model at this resolution.")

print("\n" + "=" * 60)
print("TASK 7: Save Model")
print("=" * 60)

os.makedirs("models", exist_ok=True)
save_dict = {
    "num_classes": NUM_CLASSES,
    "class_names": CLASS_NAMES,
    "fine_tuned": fine_tuned,
    "feature_extraction_val_acc": feature_extraction_val_acc,
    "input_size": INPUT_SIZE,
    "input_dim": num_features,
}
if fine_tuned:
    save_dict["model_state_dict"] = resnet.state_dict()
    save_dict["fine_tune_val_acc"] = fine_tune_val_acc
else:
    save_dict["head_state_dict"] = head.state_dict()

torch.save(save_dict, "models/product_classifier.pt")
print(f"Saved to models/product_classifier.pt")
print(f"  fine_tuned={fine_tuned}, val_acc={fine_tune_val_acc:.4f}")

print("\n" + "=" * 60)
print("TASK 8: Export Sample Images as .png")
print("=" * 60)

os.makedirs("data/sample_images", exist_ok=True)
raw_dataset = datasets.FashionMNIST(root="./data", train=False, download=True)

exported = []
for cls in range(NUM_CLASSES):
    cls_indices = np.where(test_labels == cls)[0]
    idx = cls_indices[0]
    img_array = np.array(raw_dataset[idx][0])
    fname = f"{cls:02d}_{CLASS_NAMES[cls].replace('/', '_').replace(' ', '_').lower()}.png"
    fpath = os.path.join("data/sample_images", fname)
    Image.fromarray(img_array).save(fpath)
    exported.append(fpath)
    print(f"  {fpath} (label: {CLASS_NAMES[cls]})")

print(f"\nExported {len(exported)} sample images to data/sample_images/")
print("\nPart 2 complete!")
