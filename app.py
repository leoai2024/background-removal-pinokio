import os
import gradio as gr
import torch
from glob import glob
from gradio_imageslider import ImageSlider
from loadimg import load_img
from torchvision import transforms

from utils.birefnet import load_birefnet_model, run_birefnet_infer


# check gpu support
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# set home directory
os.environ["HOME"] = os.path.expanduser("~")
print(os.environ["HOME"])

# load birefnet model
birefnet_model = load_birefnet_model(DEVICE)
# transform image
transform_image = transforms.Compose(
    [
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


# image processing function
def fn(image):
    im = load_img(image, output_type="pil")
    im = im.convert("RGB")
    image_size = im.size
    origin = im.copy()
    image = load_img(im)
    input_images = transform_image(image).unsqueeze(0).to(DEVICE)

    # run birefnet
    pred_pil = run_birefnet_infer(birefnet_model, input_images)
    mask = pred_pil.resize(image_size)
    image.putalpha(mask)

    return (image, origin)


slider1 = ImageSlider(label="birefnet", type="pil")
slider2 = ImageSlider(label="birefnet", type="pil")
image = gr.Image(label="Upload an image")
text = gr.Textbox(label="Paste an image URL")

# image examples
image_examples = [[_] for _ in glob("examples/*")][:]
# Add the option of resolution in a text box.
for idx_example, example in enumerate(image_examples):
    image_examples[idx_example].append("1024x1024")
image_examples.append(image_examples[-1].copy())
image_examples[-1][1] = "512x512"

# url examples
url_examples = [
    "https://hips.hearstapps.com/hmg-prod/images/gettyimages-1229892983-square.jpg"
]

tab1 = gr.Interface(
    fn, inputs=image, outputs=slider1, examples=image_examples, api_name="image"
)

tab2 = gr.Interface(
    fn, inputs=text, outputs=slider2, examples=url_examples, api_name="text"
)


demo = gr.TabbedInterface(
    [tab1, tab2],
    ["image", "url"],
    title="BiRefNet for background removal, model at: https://huggingface.co/ZhengPeng7/BiRefNet",
)

if __name__ == "__main__":
    demo.launch()
