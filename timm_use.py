import requests
import timm
import torch
from PIL import Image
from pprint import pprint

url = 'https://datasets-server.huggingface.co/assets/imagenet-1k/--/default/test/12/image/image.jpg'
image = Image.open(requests.get(url, stream=True).raw)

# 新建模型
model = timm.create_model('mobilenetv3_large_100', pretrained=True).eval()
# 用模型的参数新建转换器
transform = timm.data.create_transform(**timm.data.resolve_data_config(model.pretrained_cfg))
# 将图片传到转换器预处理
image_tensor = transform(image)
# 将预处理后的图片送至模型处理
output = model(image_tensor.unsqueeze(0))
# 获取预测的概率
probabilities = torch.nn.functional.softmax(output[0], dim=0)
# 获取前5的预测结果
values, indices = torch.topk(probabilities, 5)
# 获取前五的标签
IMAGENET_1k_URL = 'https://storage.googleapis.com/bit_models/ilsvrc2012_wordnet_lemmas.txt'
IMAGENET_1k_LABEL = requests.get(IMAGENET_1k_URL).text.strip().split('\n')
pprint([{'label': IMAGENET_1k_LABEL[idx], 'value': val.item()} for val, idx in zip(values, indices)])