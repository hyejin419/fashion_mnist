import struct
import numpy as np
from PIL import Image
import os

def load_images(file_path):
    with open(file_path, 'rb') as f:
        magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
        assert magic == 2051
        image_data = np.frombuffer(f.read(), dtype=np.uint8)
        images = image_data.reshape(num_images, rows, cols)
    return images

# 이미지 불러오기
images = load_images("./train-images-idx3-ubyte")

# 저장할 폴더 만들기
output_dir = "sample_images"
os.makedirs(output_dir, exist_ok=True)

# 앞에서 10개 이미지 저장
for i in range(10):
    img_array = images[i].astype(np.uint8)
    img = Image.fromarray(img_array)  # mode='L' 생략 (권장)
    img.save(f"{output_dir}/image_{i}.png")

