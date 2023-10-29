from transformers import ViTImageProcessor,ViTModel
from PIL import Image
import requests

url = 'http://images.cocodataset.org/val2017/000000039769.jpg'  # 两只小猫
image = Image.open(requests.get(url,stream=True).raw)

processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
inputs = processor(images=image,return_tensors = "pt")

outputs = model(**inputs)
last_hidden_states = outputs.last_hidden_state

print("aa")