import torch
import torchvision
from PIL import Image
from model import Covid_cnn

# 1. 读取图片
img_path = "../data/test/肺炎2.jpeg"
image = Image.open(img_path)
print("原始图像模式: {}".format(image.mode))

# 2. 确保是 RGB 格式
if image.mode != "RGB":
    image = image.convert("RGB")

# 3. 预处理
transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x)
])
image = transform(image).unsqueeze(0).cuda()
print("预处理后 shape: {}".format(image.shape))

# 4. 加载模型
covid_cnn = Covid_cnn().cuda()
model_path = "../models/covid_10.pth"
covid_cnn.load_state_dict(torch.load(model_path))
covid_cnn.eval()

# 5. 预测
with torch.no_grad():
    output = covid_cnn(image)
    probability = torch.sigmoid(output).item()
print("阳性概率: {}".format(probability))
