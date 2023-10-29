# from huggingface_hub import notebook_login
# notebook_login()

from datasets import load_dataset
from transformers import AutoImageProcessor

# 加载 food101 数据集
food = load_dataset("food101", split="train[:5000]")
food = food.train_test_split(test_size=0.2)
aaa = food["train"][0]
labels = food["train"].features["label"].names
label2id,id2label = dict(),dict()
for i ,label in enumerate(labels):
    label2id[label]= str(i)
    id2label[str(i)] = label
bbb = id2label[str(79)]

checkpoint = "google/vit-base-patch16-224-in21k"
image_processor = AutoImageProcessor.from_pretrained(checkpoint)



print(aaa)
print(bbb)