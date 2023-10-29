from urllib.request import urlopen
from PIL import Image
import torch
import timm

img = Image.open(
    urlopen("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png"))

# 新建model
model = timm.create_model('mobilenetv3_large_100.ra_in1k',pretrained=True).eval()
# 用模型的参数新建转换器
data_config = timm.data.resolve_model_data_config(model)
transforms = timm.data.create_transform(**data_config, is_training=False)

# 将图片传到转换器预处理后的送至模型处理
output = model(transforms(img).unsqueeze(0))

#
top5_probabilities, top5_class_indices = torch.topk(output.softmax(dim=1) * 100, k=5)




print("aa")

