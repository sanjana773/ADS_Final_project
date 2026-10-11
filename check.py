import torch
for path in ["model/tomato_disease_model.pth", "model/fruit_disease_model.pth"]:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    print(path)
    for key in ("best_val_accuracy", "test_accuracy"):
        print("  ", key, "->", ckpt.get(key, "NOT SAVED"))
        