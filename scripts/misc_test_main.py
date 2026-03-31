import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import requests


def test_conditional_predict():
    from shearwall_pred.config import data_config, model_config, training_config
    from shearwall_pred.cross_validate import EnsembleShearWallGNN
    from shearwall_pred.visualize_test import test_conditional_predict

    cv_path = Path(data_config.SAVE_DIR)
    model_paths = sorted(list(cv_path.glob("fold_*/best_model.pth")))
    ensemble_model = EnsembleShearWallGNN(model_paths, model_config)
    ensemble_model.to(training_config.DEVICE)

    cate_ious = {0: [], 1: [], 2: []}
    for fname in os.listdir(data_config.DXF_DIR + "/test"):
        if fname.endswith(".dxf"):
            dxf = os.path.join(data_config.DXF_DIR, "test", fname)
            save_dir = str(cv_path / "conditional_test_results")
            os.makedirs(save_dir, exist_ok=True)

            save_path = os.path.join(save_dir, os.path.basename(dxf).replace(".dxf", ".png"))
            ious = test_conditional_predict(dxf_path=dxf, model=ensemble_model, save_path=save_path)

            for i, iou in enumerate(ious):
                cate_ious[i].append(iou)


def get_density_stats():
    from shearwall_pred.config import data_config
    from shearwall_pred.trainer import DataManager

    stats = {0: [], 1: [], 2: []}

    dm = DataManager(root=data_config.DXF_DIR)
    train_loader, _ = dm.get_train_val_loaders(val_ratio=0)

    for data in train_loader:
        avg_wall_per_node = data.y.sum(dim=1).mean().item()
        category = data.condition.argmax().item()
        stats[category].append(avg_wall_per_node)

    print("Data count per Category:", [len(stats[0]), len(stats[1]), len(stats[2])])
    print("Target Densities:", [np.mean(stats[0]), np.mean(stats[1]), np.mean(stats[2])])


def test_shapely_polygon():
    from shapely.geometry import Polygon

    poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    print("Area:", poly.area)
    print("Bounds:", poly.bounds)
    print("Centroid:", poly.centroid)


def test_axis_engine_import():
    from axisengine.graphy.graph_builder import build_graph_from_architectural, load_json_data
    from axisengine.graphy.visualize_graph import visualize_graph

    archi_data = load_json_data(
        r"E:\Common\Desktop\Research\deepLearning\codes\AxisEngine\axisengine\graphy\test_data\archi.json"
    )
    struct_data = load_json_data(
        r"E:\Common\Desktop\Research\deepLearning\codes\AxisEngine\axisengine\graphy\test_data\struct.json"
    )

    print("=" * 70)
    print("步骤1: 构建建筑图（输入）")
    print("=" * 70)
    archi_graph = build_graph_from_architectural(archi_data, beam=struct_data["beam"], tolerance=300)
    print(f"节点数: {len(archi_graph.nodes)}")
    print(f"边数: {len(archi_graph.edges)}")
    print(f"构件类型: {archi_graph._count_edge_types()}")

    print("\n" + "=" * 70)
    print("步骤3: 导出图数据")
    print("=" * 70)

    archi_output = archi_graph.to_dict()
    with open("./outputs/result/architectural_graph.json", "w", encoding="utf-8") as f:
        json.dump(archi_output, f, indent=2, ensure_ascii=False)
    print("✓ 建筑图已保存: outputs/result/architectural_graph.json")
    visualize_graph(
        archi_output,
        output_path="outputs/result/archi_graph.png",
        show_node_ids=True,
        edge_types_to_draw=["shear_wall", "beam"],
    )


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)


def get_bibtex_from_doi(doi: str, timeout: int = 10) -> str:
    api_url = f"https://www.doi2bib.org/8350e5a3e24c153df2275c9f80692773/doi2bib?id={doi.strip()}"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
    }

    try:
        response = requests.get(api_url, headers=headers, timeout=timeout)
        response.raise_for_status()
        bibtex = response.text.strip()
        if not bibtex:
            logging.warning(f"DOI {doi} 未返回任何BibTeX内容")
            return ""
        return bibtex
    except requests.exceptions.HTTPError as e:
        logging.error(f"DOI {doi} 请求失败：HTTP错误 {e.response.status_code}")
    except requests.exceptions.ConnectionError:
        logging.error(f"DOI {doi} 请求失败：网络连接错误")
    except requests.exceptions.Timeout:
        logging.error(f"DOI {doi} 请求失败：超时（{timeout}秒）")
    except Exception as e:
        logging.error(f"DOI {doi} 处理失败：未知错误 {str(e)}")

    return ""


def batch_get_bibtex(doi_file: str, output_file: str, delay: float = 1.5):
    try:
        with open(doi_file, "r", encoding="utf-8") as f:
            doi_list = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        logging.error(f"DOI文件 {doi_file} 未找到！")
        return
    except Exception as e:
        logging.error(f"读取DOI文件失败：{str(e)}")
        return

    total = len(doi_list)
    success_count = 0
    logging.info(f"开始处理 {total} 个DOI...")

    with open(output_file, "a", encoding="utf-8") as out_f:
        for idx, doi in enumerate(doi_list, 1):
            logging.info(f"正在处理 [{idx}/{total}]：{doi}")

            bibtex = get_bibtex_from_doi(doi)
            if bibtex:
                out_f.write(bibtex + "\n\n")
                success_count += 1

            if idx < total:
                time.sleep(delay)

    logging.info(f"处理完成！成功获取 {success_count}/{total} 个BibTeX条目，已保存到 {output_file}")


def main() -> None:
    print("正在读取结果数据并生成箱线图...")
    from experiments.ablation.analyze_results import plot_boxplot

    results_path = r"outputs\outputs\result\ablation_study\analysis\conditional_eval_results.json"
    with open(results_path, "r", encoding="utf-8") as f:
        results_data = json.load(f)
    plot_boxplot(results_data, r"cgs_boxplot.png")


if __name__ == "__main__":
    main()
