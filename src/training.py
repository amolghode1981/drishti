import torch
import clip
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import  src.model_utils

# ── 1. Dataset Class ──────────────────────────────────────────
class InstrumentDataset(Dataset):
    def __init__(self, dataframe, preprocess):
        self.dataframe = dataframe
        self.preprocess = preprocess

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]

        # Load and preprocess image
        image = Image.open(row["local_path"]).convert("RGB")
        image = self.preprocess(image)

        # Build text prompt
        instrument = row["instrument"]
        text = f"{instrument}"
        #print(f"image path = {row["local_path"]} instrumet = {text}")
        return image, text

model = None
device = None
preprocess = None
train_loader = None
val_loader = None
optimzer = None
loss_fn = None

def load_model():
    global model, device, preprocess, optimizer, loss_fn
    device = "mps" if torch.backends.mps.is_available() else "cpu"
# ──  Load Model ─────────────────────────────────────────────
    print(f"Using device: {device}")
    model, preprocess = src.model_utils.load_model("ViT-B/32")
    model = model.float()
# ── Optimizer and loss function ──────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-6,
        weight_decay=0.01
    )
    loss_fn = torch.nn.CrossEntropyLoss()


def load_data():
    global train_loader, val_loader
    # ──  Load Data ──────────────────────────────────────────────
    train_df = pd.read_csv("datasets/csvs/training/training_objects.csv")
    val_df   = pd.read_csv("datasets/csvs/training/validation_objects.csv")

    # Adjust column names to match your CSV
    train_df.columns = ["id", "instrument", "museum_index", "local_path"]
    val_df.columns   = ["id", "instrument", "museum_index", "local_path"]

    train_dataset = InstrumentDataset(train_df, preprocess)
    val_dataset   = InstrumentDataset(val_df, preprocess)

    train_loader  = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader    = DataLoader(val_dataset, batch_size=16, shuffle=False)


def init():
    load_model()
    load_data()

def run_training_loop():
# ── 5. Training Loop ──────────────────────────────────────────
    NUM_EPOCHS = 5
    for epoch in range(NUM_EPOCHS):
        # ── Training ──
        model.train()
        train_loss = 0

        for images, texts in train_loader:
            # Move images to the device
            images = images.to(device)
            print(train_loader)
            text_tokens = clip.tokenize(texts).to(device)

            # Forward pass
            image_features = model.encode_image(images)
            text_features  = model.encode_text(text_tokens)

            # Normalize
            '''
            Depending on the features, the vector lengths could varie. We dont want to do operations with the vectors
            that are in different space. So normalize them. i.e convert them to a unit vector.
            '''
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features  = text_features  / text_features.norm(dim=-1, keepdim=True)

            # Contrastive loss
            '''
            Idea here is this. Matrices get arranged diagonally, and diagonal should be having a maximum value
            and rest should have less value. In other words, Image 0 has a truth value of Text 0. So Value[0,0]
            Should be max and rest other pairs with zero should be mimimum.
            If this is not the case, it would mean the prediction is not yet acceptable.
            So now penalize non max diagonals and max non diagonals
            By the penalization process weights are adjusted.
                        Text 0    Text 1    Text 2
            Image 0 Tabla  [ 0.91,    0.12,     0.08 ]  ← Image 0 should match Text 0
            Image 1 Sitar  [ 0.10,    0.88,     0.11 ]  ← Image 1 should match Text 1
            Image 2 Pakha  [ 0.09,    0.08,     0.93 ]  ← Image 2 should match Text 2
            '''
            # Logit scale
            logit_scale = model.logit_scale.exp()

            '''
            multiply image features with text features
            image_features shape:    [16, 512]   (16 images, 512 dimensions)
            text_features shape:     [16, 512]   (16 texts, 512 dimensions)
            So first Transpose text. So it becomes
                Original text_features (3 texts, 4 dimensions):
                    dim0  dim1  dim2  dim3
                Text0 [ 0.6,  0.2,  0.8,  0.1]
                Text1 [ 0.3,  0.9,  0.1,  0.7]
                Text2 [ 0.5,  0.4,  0.2,  0.6]
                Shape: [3, 4]

                After .T:
                        Text0  Text1  Text2
                dim0  [  0.6,   0.3,   0.5]
                dim1  [  0.2,   0.9,   0.4]
                dim2  [  0.8,   0.1,   0.2]
                dim3  [  0.1,   0.7,   0.6]
                Shape: [4, 3]

            '''
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text  = logits_per_image.T
            labels = torch.arange(len(images), device=device)
            '''
            A beautiful thing to note logits_per_image and logits_per_text are same matrix data transposed.
            So
            logits_per_image:          logits_per_text (transposed):
                        Text0  Text1               Image0  Image1
            Image0      [0.91,  0.12]    Text0     [0.91,   0.10]
            Image1      [0.10,  0.88]    Text1     [0.12,   0.88]

            Is going tobe true.
            And labels = [0, 1, 2...] is the same for both — the two losses will always be mathematically identical in this code.

            loss = (same_value + same_value) / 2 = same_value
            It's carried over from the original CLIP paper where the two losses were computed slightly differently — using separate learned parameters for each direction.
            In this simplified implementation it's technically redundant. You could write:
            '''
            loss = (loss_fn(logits_per_image, labels) + loss_fn(logits_per_text, labels)) / 2

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # ── Validation ──
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for images, texts in val_loader:
                images = images.to(device)
                text_tokens = clip.tokenize(texts).to(device)

                image_features = model.encode_image(images)
                text_features  = model.encode_text(text_tokens)

                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features  = text_features  / text_features.norm(dim=-1, keepdim=True)

                logit_scale = model.logit_scale.exp()
                logits_per_image = logit_scale * image_features @ text_features.T
                logits_per_text  = logits_per_image.T

                labels = torch.arange(len(images), device=device)
                loss = (loss_fn(logits_per_image, labels) + loss_fn(logits_per_text, labels)) / 2

                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")



def save_model():
    # ── 6. Save Model ─────────────────────────────────────────────
    os.makedirs("output", exist_ok=True)
    torch.save(model.state_dict(), "output/clip_finetuned.pt")
    print("Model saved to output/clip_finetuned.pt")