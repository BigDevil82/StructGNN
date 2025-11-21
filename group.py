import os
import random
import shutil

random.seed(42)


def split_dxf_into_groups(src_dir=None, dst_base=None, per_group=14, ext_filter=(".dxf",)):
    """
    从 src_dir 随机把 dxf 文件分成 groups 组，每组 per_group 个文件，最后一组包含剩余文件。
    将每组文件复制到 dst_base/group_i 子文件夹中，返回每组文件数量的字典。
    """
    if src_dir is None:
        src_dir = os.path.join(os.path.dirname(__file__), "dxf", "to_process", "raw_dynamic_scale")
    if dst_base is None:
        dst_base = os.path.join(os.path.dirname(__file__), "dxf", "to_process", "group")

    if not os.path.isdir(src_dir):
        raise FileNotFoundError(f"source directory not found: {src_dir}")

    if not isinstance(per_group, int) or per_group <= 0:
        raise ValueError("per_group must be a positive integer")

    os.makedirs(dst_base, exist_ok=True)

    # 收集符合后缀的文件（不递归）
    files = [
        f
        for f in os.listdir(src_dir)
        if os.path.isfile(os.path.join(src_dir, f)) and f.lower().endswith(ext_filter)
    ]
    finished_files = ["L1L28_30.dxf", "L1L28_190.dxf", "L1L28_232.dxf"]
    files = [f for f in files if f not in finished_files]

    if not files:
        return {}

    random.shuffle(files)

    total = len(files)
    groups = (total + per_group - 1) // per_group  # 向上取整，保证最后一组包含剩余文件

    # 分组并复制
    for i in range(groups):
        group_name = f"group_{i+1}"
        group_dir = os.path.join(dst_base, group_name)
        os.makedirs(group_dir, exist_ok=True)

        start = i * per_group
        end = start + per_group
        chunk = files[start:end]

        for fname in chunk:
            src_path = os.path.join(src_dir, fname)
            dst_path = os.path.join(group_dir, fname)
            shutil.copy2(src_path, dst_path)

    # 将finished_files复制到最后一组
    if finished_files:
        group_name = f"group_{groups+1}"
        group_dir = os.path.join(dst_base, group_name)
        os.makedirs(group_dir, exist_ok=True)
        for fname in finished_files:
            src_path = os.path.join(src_dir, fname)
            dst_path = os.path.join(group_dir, fname)
            shutil.copy2(src_path, dst_path)

    # 返回各组文件数量，便于检查
    summary = {}
    for i in range(groups):
        group_name = f"group_{i+1}"
        group_dir = os.path.join(dst_base, group_name)
        count = len([n for n in os.listdir(group_dir) if os.path.isfile(os.path.join(group_dir, n))])
        summary[group_name] = count
    return summary


if __name__ == "__main__":
    src_dir = r"E:\Common\Desktop\Research\deepLearning\codes\Png2Dxf\dxf\to_process\room_raw"
    dst_base = r"E:\Common\Desktop\Research\deepLearning\codes\Png2Dxf\dxf\to_process\room_group"

    summary = split_dxf_into_groups(src_dir, dst_base, 46)
    for k, v in summary.items():
        print(f"{k}: {v} files")
