import os
import random
import shutil

# 设置随机种子，保证可复现
random.seed(42)

# 定义数据集路径
original_dataset_dir = "../flower_photos"
base_dir = "../flower_dataset"

# 创建 `train/` 和 `val/` 目录
train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "val")

if not os.path.exists(base_dir):
    os.makedirs(train_dir)
    os.makedirs(val_dir)

# 遍历每个类别文件夹
for category in os.listdir(original_dataset_dir):
    category_path = os.path.join(original_dataset_dir, category)

    # 跳过非图片的文件，比如 LICENSE.txt
    if not os.path.isdir(category_path):
        continue

    # 获取该类别的所有图片
    images = os.listdir(category_path)
    random.shuffle(images)  # 随机打乱

    # 计算训练集和验证集大小
    train_size = int(0.8 * len(images))  # 80% 训练，20% 验证

    # 定义该类别的目标目录
    train_category_dir = os.path.join(train_dir, category)
    val_category_dir = os.path.join(val_dir, category)
    os.makedirs(train_category_dir, exist_ok=True)
    os.makedirs(val_category_dir, exist_ok=True)

    # 复制图片到训练集和验证集
    for i, image in enumerate(images):
        src = os.path.join(category_path, image)

        if i < train_size:
            dst = os.path.join(train_category_dir, image)
        else:
            dst = os.path.join(val_category_dir, image)

        shutil.copyfile(src, dst)

print("Dataset has been split into train/ and val/")