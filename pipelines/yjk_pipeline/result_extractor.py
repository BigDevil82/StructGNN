from __future__ import annotations

import csv
import faulthandler
import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from YJKAPI import *  # type: ignore # noqa: F401,F403

try:
    from .common import setup_file_logger
    from .config import ExtractionConfig, OutputConfig
except ImportError:
    from pipelines.yjk_pipeline.common import setup_file_logger
    from pipelines.yjk_pipeline.config import ExtractionConfig, OutputConfig


@dataclass
class StoryDriftData:
    envelope: Dict
    details: list[Dict]
    metrics: Dict


class YJKResultExtractor:
    """分析结果提取器，聚焦剪力墙层间位移角。"""

    def __init__(self, extract_config: ExtractionConfig, output_config: OutputConfig, base_dir: Path):
        self.extract_config = extract_config
        self.output_config = output_config
        self.output_dir = output_config.resolve_output_dir(base_dir)
        self.logger = setup_file_logger(
            "YJKResultExtractor",
            self.output_dir / "result_extractor.log",
            output_config.log_level,
        )
        self._fault_log_file = None
        self._init_faulthandler()

    def _init_faulthandler(self):
        try:
            fault_log = self.output_dir / "faulthandler.log"
            self._fault_log_file = open(fault_log, "w", encoding="utf-8")
            faulthandler.enable(file=self._fault_log_file, all_threads=True)
        except Exception as exc:
            self.logger.warning("Failed to enable faulthandler: %s", exc)

    @staticmethod
    def _to_float(v, default=0.0):
        try:
            return float(v)
        except Exception:
            return default

    @staticmethod
    def _flatten_numbers(value, out):
        if isinstance(value, (list, tuple)):
            for item in value:
                YJKResultExtractor._flatten_numbers(item, out)
            return
        num = YJKResultExtractor._to_float(value, None)
        if num is not None:
            out.append(num)

    @staticmethod
    def _dedup_keep_order(values, ndigits=8):
        seen = set()
        out = []
        for value in values:
            key = round(value, ndigits)
            if key in seen:
                continue
            seen.add(key)
            out.append(value)
        return out

    def _extract_first5_periods(self, data):
        if isinstance(data, (list, tuple)) and len(data) >= 3:
            period_arr = data[2]
            if isinstance(period_arr, (list, tuple)):
                periods = []
                for value in period_arr:
                    period_v = self._to_float(value, None)
                    if period_v is not None and period_v > 0.0:
                        periods.append(period_v)
                periods = self._dedup_keep_order(periods)
                if periods:
                    return periods[:5]
        return []

    def _extract_total_mass(self, value):
        v = self._to_float(value, None)
        if v is not None and v > 0.0:
            return v

        nums = []
        self._flatten_numbers(value, nums)
        positives = [x for x in nums if x > 0.0]
        if not positives:
            return 0.0
        return max(positives)

    def _extract_drift_angle(self, dir_info):
        if dir_info is None:
            return 0.0

        if isinstance(dir_info, (list, tuple)) and len(dir_info) >= 5:
            return self._to_float(dir_info[4], 0.0)

        nums = []
        self._flatten_numbers(dir_info, nums)
        if not nums:
            return 0.0

        small = [x for x in nums if 0.0 < abs(x) < 0.1]
        if small:
            return max(small, key=lambda x: abs(x))

        return nums[-1]

    @staticmethod
    def _angle_to_ratio_text(angle):
        if abs(angle) < 1.0e-12:
            return "-"
        return f"1/{1.0 / abs(angle):.0f}"

    def collect_key_metrics(self) -> Dict:
        total_mass = None
        if self.extract_config.enable_total_mass_api:
            try:
                total_mass_raw = YJKSDsnDataPy.dsnGetTolMass()  # type: ignore
                total_mass = self._extract_total_mass(total_mass_raw)
            except Exception as exc:
                self.logger.warning("dsnGetTolMass failed: %s", exc)

        periods = []
        try:
            eign_raw = YJKSDsnDataPy.dsnGetEignInfoByReader(self.extract_config.seis_module)  # type: ignore
            periods = self._extract_first5_periods(eign_raw)
        except Exception as exc:
            self.logger.warning("dsnGetEignInfoByReader failed: %s", exc)

        if not periods:
            try:
                first_period_raw = YJKSDsnDataPy.dsnGetFirstPeriod(self.extract_config.seis_module)  # type: ignore
                fp = self._to_float(first_period_raw, None)
                if fp is not None and fp > 0.0:
                    periods = [fp]
                else:
                    nums = []
                    self._flatten_numbers(first_period_raw, nums)
                    periods = [x for x in nums if 0.0 < x < 20.0][:5]
            except Exception as exc:
                self.logger.warning("dsnGetFirstPeriod failed: %s", exc)

        periods = self._dedup_keep_order([p for p in periods if p > 0.0])[:5]
        return {"total_mass": total_mass, "periods": periods}

    def collect_story_drift(self) -> tuple[Dict, list[Dict]]:
        YJKSDsnDataPy.dsnInitData()  # type: ignore
        pre_data = YJKSPrePy()  # type: ignore

        floor_count = pre_data.NZRC()
        tower_count = pre_data.TowInMdl()
        if tower_count <= 0:
            tower_count = 1

        envelope = {}
        details: list[Dict] = []

        for tower in range(1, tower_count + 1):
            for floor in range(1, floor_count + 1):
                dsx, dsy = YJKSDsnDataPy.dsnGetFlrSeisInfo(  # type: ignore
                    floor,
                    tower,
                    self.extract_config.seis_module,
                )
                x_ang = self._extract_drift_angle(dsx)
                y_ang = self._extract_drift_angle(dsy)

                envelope[(tower, floor)] = {
                    "x": x_ang,
                    "y": y_ang,
                    "x_case": f"SeisInfo_Module{self.extract_config.seis_module}",
                    "y_case": f"SeisInfo_Module{self.extract_config.seis_module}",
                }

                details.append(
                    {
                        "tower": tower,
                        "floor": floor,
                        "case_id": -1,
                        "case_name": f"SeisInfo_Module{self.extract_config.seis_module}",
                        "x": x_ang,
                        "y": y_ang,
                    }
                )

        return envelope, details

    def get_story_drift_data(self) -> StoryDriftData:
        metrics = self.collect_key_metrics()
        envelope, details = self.collect_story_drift()
        return StoryDriftData(envelope=envelope, details=details, metrics=metrics)

    def write_report_txt(self, data: StoryDriftData) -> Path:
        report_path = self.output_dir / "story_drift_report.txt"

        with report_path.open("w", encoding="utf-8") as f:
            f.write("剪力墙结构层间位移角结果\n")
            f.write("=" * 60 + "\n")
            f.write(f"固定模块: module={self.extract_config.seis_module}\n\n")

            f.write("关键指标\n")
            f.write("-" * 60 + "\n")
            if data.metrics.get("total_mass") is None:
                f.write("结构总质量: 未提取\n")
            else:
                f.write(f"结构总质量: {data.metrics['total_mass']:.6f}\n")

            if data.metrics["periods"]:
                for idx, value in enumerate(data.metrics["periods"], start=1):
                    f.write(f"第{idx}阶周期: {value:.6f} s\n")
            else:
                f.write("前5阶周期: 未读取到有效值\n")
            f.write("\n")

            f.write("包络结果\n")
            f.write("-" * 60 + "\n")
            for tower, floor in sorted(data.envelope.keys()):
                env = data.envelope[(tower, floor)]
                f.write(
                    f"层号:{floor:>2d}  塔号:{tower:>2d}  "
                    f"X:{env['x']:.6e} ({self._angle_to_ratio_text(env['x'])})  "
                    f"Y:{env['y']:.6e} ({self._angle_to_ratio_text(env['y'])})  "
                    f"工况:{env['x_case']}\n"
                )

        return report_path

    def write_summary_csv(self, data: StoryDriftData) -> Path:
        csv_path = self.output_dir / "story_drift_summary.csv"

        with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["tower", "floor", "x_ang", "x_ratio", "y_ang", "y_ratio", "case_name"])

            for tower, floor in sorted(data.envelope.keys()):
                env = data.envelope[(tower, floor)]
                writer.writerow(
                    [
                        tower,
                        floor,
                        f"{env['x']:.6e}",
                        self._angle_to_ratio_text(env["x"]),
                        f"{env['y']:.6e}",
                        self._angle_to_ratio_text(env["y"]),
                        env["x_case"],
                    ]
                )

        return csv_path

    def write_metrics_csv(self, data: StoryDriftData) -> Path:
        csv_path = self.output_dir / "key_metrics.csv"

        with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "value"])

            total_mass = data.metrics.get("total_mass")
            writer.writerow(["total_mass", "" if total_mass is None else f"{total_mass:.6f}"])

            periods = data.metrics.get("periods", [])
            for idx in range(1, 6):
                value = periods[idx - 1] if idx <= len(periods) else ""
                writer.writerow([f"period_{idx}", "" if value == "" else f"{value:.6f}"])

        return csv_path

    def plot_story_drift(self, envelope: Dict) -> Optional[Path]:
        if not self.extract_config.export_plot:
            return None

        try:
            plt = importlib.import_module("matplotlib.pyplot")

            rows = []
            for tower, floor in sorted(envelope.keys()):
                env = envelope[(tower, floor)]
                rows.append((floor, abs(env["x"]), abs(env["y"])))

            if not rows:
                return None

            floors = [r[0] for r in rows]
            x_vals = [r[1] for r in rows]
            y_vals = [r[2] for r in rows]

            fig, ax = plt.subplots(figsize=(8, 10), dpi=140)
            ax.plot(x_vals, floors, marker="o", linewidth=1.8, label="X abs")
            ax.plot(y_vals, floors, marker="s", linewidth=1.8, label="Y abs")
            ax.set_xlabel("Drift angle abs")
            ax.set_ylabel("Floor")
            ax.set_title(f"Story Drift Envelope (module={self.extract_config.seis_module})")
            ax.grid(True, linestyle="--", alpha=0.35)
            ax.legend()
            ax.invert_yaxis()
            fig.tight_layout()

            plot_path = self.output_dir / "story_drift_plot.png"
            fig.savefig(plot_path)
            plt.close(fig)
            return plot_path

        except Exception as exc:
            self.logger.error("Failed to create plot: %s", exc, exc_info=True)
            return None

    def export_reports(self, data: StoryDriftData) -> dict[str, Optional[Path]]:
        outputs = {
            "report": self.write_report_txt(data),
            "summary": self.write_summary_csv(data),
            "metrics": self.write_metrics_csv(data),
            "plot": self.plot_story_drift(data.envelope),
        }
        return outputs

    def run_extraction(self) -> tuple[StoryDriftData, dict[str, Optional[Path]]]:
        data = self.get_story_drift_data()
        outputs = self.export_reports(data)
        return data, outputs

    def close(self) -> None:
        try:
            faulthandler.disable()
            if self._fault_log_file:
                self._fault_log_file.close()
                self._fault_log_file = None
        except Exception as exc:
            self.logger.warning("Error during close: %s", exc)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
