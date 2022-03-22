# %%

from train import get_instance_segmentation_model
import torch
from PIL import Image
import transforms as T
from tqdm import tqdm
import numpy as np
from glob import glob

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model = get_instance_segmentation_model(2)
model.load_state_dict(torch.load("model/models/trained.pth"))
model.eval()

trans = T.Compose([
    T.ToTensor()])

def segment_image(path):
    img = Image.open(path).convert("RGB")
    transformed = trans(target=img, image = img)[0]
    with torch.no_grad():
        prediction = model([transformed.to(device)])
        
    array_out = transformed.mul(255).permute(1, 2, 0).byte().numpy()
    try:
        mask = prediction[0]['masks'][0, 0].cpu().numpy()
    except:
        return array_out.astype(np.uint8)
    mask_3d = mask[:, :, None] * np.ones(3, dtype=float)[None, None, :]
    multiplied = np.multiply(array_out, mask_3d)
    return np.round(multiplied).astype(np.uint8)


def image_is_black(image, threshold):
    return np.sum(image) < threshold
       
if __name__ == "__main__":
    for image in tqdm(sorted(glob("FFHQ/*"))):
        out = segment_image(image)
        im = Image.fromarray(out)
        if not image_is_black(out, threshold = 1000000):
            im.save("FFHQ_NoBGR/" + image.split("/")[-1])
        
# %%
