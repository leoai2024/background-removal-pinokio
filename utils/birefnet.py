import torch
from transformers import AutoModelForImageSegmentation
from torchvision import transforms

BIREFNET_MODEL_PATH = "ZhengPeng7/BiRefNet"


# Load BiRefNet with weights
def load_birefnet_model(
    device: str,
    model_path: str = BIREFNET_MODEL_PATH,
):
    model = AutoModelForImageSegmentation.from_pretrained(
        model_path, trust_remote_code=True
    )
    torch.set_float32_matmul_precision(["high", "highest"][0])
    model.to(device)
    return model


def run_birefnet_infer(model, input_images):
    # Prediction
    with torch.no_grad():
        preds = model(input_images)[-1].sigmoid().cpu()
    pred = preds[0].squeeze()
    pred_pil = transforms.ToPILImage()(pred)
    return pred_pil
