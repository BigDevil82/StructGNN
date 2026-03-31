import os
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

from src.shearwall_pred.config import data_config, training_config
from src.shearwall_pred.utils import get_file_category

# ================= 配置区域 =================
# 原始 DXF 文件夹路径
SOURCE_DIR = data_config.DXF_DIR

# 输出的目标文件夹路径 (会自动创建)
TARGET_DIR = r"data/dxf/dataset_split_8_2"

# 划分比例 (训练集占比)
TRAIN_RATIO = 0.8

# 随机种子 (保证每次划分结果一致)
SEED = training_config.RANDOM_SEED


def split_dataset():
    # 1. 初始化
    random.seed(SEED)
    src_path = Path(SOURCE_DIR)
    target_path = Path(TARGET_DIR)

    train_dir = target_path / "train"
    test_dir = target_path / "test"

    # 如果目标目录不存在，创建它；如果存在，提示用户
    if target_path.exists():
        print(f"⚠️ 警告: 目标目录 {TARGET_DIR} 已存在。")
        ans = input("是否清空并重新划分? (y/n): ")
        if ans.lower() == "y":
            shutil.rmtree(target_path)
        else:
            print("操作已取消")
            return

    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    # 2. 扫描并按类别分组
    files_by_category = defaultdict(list)
    all_files = [f for f in os.listdir(src_path) if f.endswith(".dxf")]

    if not all_files:
        print(f"❌ 错误: 源目录 {SOURCE_DIR} 中没有找到 .dxf 文件")
        return

    print(f"📂 正在扫描源目录: {SOURCE_DIR}")
    for fname in all_files:
        cat = get_file_category(fname)
        files_by_category[cat].append(fname)

    # 3. 分层划分并复制
    stats = {"train": Counter(), "test": Counter()}

    print(f"\n🚀 开始执行分层划分 (Train: {TRAIN_RATIO*100}% / Test: {(1-TRAIN_RATIO)*100}%)")
    print("-" * 60)
    print(f"{'Category':<25} | {'Total':<8} | {'Train':<8} | {'Test':<8}")
    print("-" * 60)

    for cat, files in files_by_category.items():
        # 打乱该类别的所有文件
        random.shuffle(files)

        total_count = len(files)
        train_count = int(total_count * TRAIN_RATIO)

        # 确保至少有一个测试样本（如果总数大于1）
        if total_count > 1 and (total_count - train_count) == 0:
            train_count -= 1

        train_files = files[:train_count]
        test_files = files[train_count:]

        # 记录统计
        stats["train"][cat] += len(train_files)
        stats["test"][cat] += len(test_files)

        print(f"{cat:<25} | {total_count:<8} | {len(train_files):<8} | {len(test_files):<8}")

        # 执行复制
        for f in train_files:
            shutil.copy2(src_path / f, train_dir / f)

        for f in test_files:
            shutil.copy2(src_path / f, test_dir / f)

    print("-" * 60)
    print(f"\n✅ 划分完成！")
    print(f"📁 训练集路径: {train_dir} (共 {sum(stats['train'].values())} 张)")
    print(f"📁 测试集路径: {test_dir}  (共 {sum(stats['test'].values())} 张)")
    print(f"原始文件保持不变。")


if __name__ == "__main__":
    split_dataset()
