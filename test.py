import timm
import torch
from PIL import Image
import requests
from timm.data import resolve_data_config, create_transform
from model import lsnet
import numpy as np
import math
import matplotlib.pyplot as plt
import cv2

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load the model
model = timm.create_model(
    'hf_hub:jameslahm/lsnet_t',
    pretrained=True
)
model = model.to(device)
model.eval()
feature_maps = []
def get_activation():
    def hook(model, input, output):
        # output chính là feature map ta cần
        feature_maps.append(output.detach())
    return hook

model.blocks3[-1].mixer.ska.register_forward_hook(get_activation())

# Load and transform image
# Example using a URL:
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
}
url = 'https://i.imgur.com/jqJcwLc.jpeg'
img = Image.open(requests.get(url, headers=headers, stream=True).raw)

config = resolve_data_config({}, model=model)
transform = create_transform(**config)
input_tensor = transform(img).unsqueeze(0) # transform and add batch dimension
input_tensor = input_tensor.to(device)


# Make prediction
with torch.no_grad():
    output = model(input_tensor)
# probabilities = torch.nn.functional.softmax(output[0], dim=0)

fmap = feature_maps[0].squeeze(0)
abs_fmap = torch.abs(fmap)

heatmap = torch.mean(abs_fmap, dim=0) # -> kết quả chỉ còn [56, 56]

# Chuyển sang numpy để vẽ
heatmap_np = heatmap.cpu().numpy()

print(fmap.shape)

# Dùng colormap 'jet' hoặc 'viridis' để giống ảnh nhiệt
heatmap_np = (heatmap_np - heatmap_np.min()) / (heatmap_np.max() - heatmap_np.min() + 1e-8)

# Phóng to lên 224x224 bằng Cubic Interpolation (Làm mịn)
heatmap_smooth = cv2.resize(heatmap_np, img.size)

# 6. VẼ VÀ LƯU (OVERLAY)
# Tạo màu
heatmap_uint8 = np.uint8(255 * heatmap_smooth)
heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)


plt.figure(figsize=(6, 6))
plt.imshow(heatmap_smooth, cmap='viridis')
plt.title("Heatmap (Paper's Method)")
plt.axis('off')
plt.savefig("visualize.png")
print("Đã lưu kết quả chuẩn paper vào: visualize.png")

# Get top 5 predictions
# top5_prob, top5_catid = torch.topk(probabilities, 5)
# Assuming you have imagenet labels list 'imagenet_labels'
# for i in range(top5_prob.size(0)):
#     print(imagenet_labels[top5_catid[i]], top5_prob[i].item())
