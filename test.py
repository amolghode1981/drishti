from transformers import CLIPModel

#model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
model = CLIPModel.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
total = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total:,}")
print(f"Total parameters: {total/1e6:.1f}M")
