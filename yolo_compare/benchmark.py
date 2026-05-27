"""Fine-tune and benchmark pretrained Ultralytics YOLO models on DOTA (HBB).

Independent comparison pipeline -- it does NOT use the custom backbone. It trains
off-the-shelf YOLO weights on the YOLO-format DOTA produced by
``DOTADatasetProcessor`` and tabulates mAP/precision/recall.

Run:  python -m yolo_compare.benchmark
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from ultralytics import YOLO

import config
from common.classes import DOTA_CLASSES
from .dataset_processor import DOTADatasetProcessor


class YOLOModelTrainer:
    def __init__(self, train_dataset_yaml_path, val_dataset_yaml_path, results_dir="results"):
        self.train_dataset_yaml_path = train_dataset_yaml_path
        self.val_dataset_yaml_path = val_dataset_yaml_path
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        # Model name -> weights path/spec. Add more (yolov8m, yolov9s, ...) as needed.
        self.models = {
            "yolov8s": "yolov8s.pt",
        }
        self.training_results = {}
        self.evaluation_results = {}

    def download_and_load_model(self, model_name, model_file):
        try:
            print(f"Loading {model_name} ({model_file})...")
            return YOLO(model_file)
        except Exception as e:
            print(f"Error loading {model_name}: {e}")
            return None

    def train_model(self, model_name, model, epochs=100, batch_size=16, img_size=640):
        print(f"\n{'=' * 50}\nTraining {model_name}\n{'=' * 50}")
        try:
            results = model.train(
                data=self.train_dataset_yaml_path, epochs=epochs, batch=batch_size,
                imgsz=img_size, project=str(self.results_dir), name=f"{model_name}_training",
                patience=20, save=True, plots=True, val=True,
            )
            self.training_results[model_name] = {
                "results": results,
                "best_model_path": Path(results.save_dir) / "weights" / "best.pt",
            }
            return results
        except Exception as e:
            print(f"Error training {model_name}: {e}")
            return None

    def evaluate_model(self, model_name):
        if model_name not in self.training_results:
            print(f"No training results found for {model_name}")
            return None
        try:
            model = YOLO(str(self.training_results[model_name]["best_model_path"]))
            val_results = model.val(
                data=self.val_dataset_yaml_path, project=str(self.results_dir),
                name=f"{model_name}_validation",
            )
            self.evaluation_results[model_name] = {
                "val_results": val_results,
                "metrics": {
                    "mAP50": val_results.box.map50,
                    "mAP50-95": val_results.box.map,
                    "precision": val_results.box.p.mean(),
                    "recall": val_results.box.r.mean(),
                    "f1": val_results.box.f1.mean(),
                },
            }
            return val_results
        except Exception as e:
            print(f"Error evaluating {model_name}: {e}")
            return None

    def train_all_models(self, epochs=100):
        print("Starting training for all models...")
        for model_name, model_file in self.models.items():
            model = self.download_and_load_model(model_name, model_file)
            if model is not None:
                self.train_model(model_name, model, epochs=epochs)
                self.evaluate_model(model_name)
        print("\nAll models training completed!")

    def create_results_summary(self):
        if not self.evaluation_results:
            print("No evaluation results available")
            return None
        rows = []
        for model_name, results in self.evaluation_results.items():
            m = results["metrics"]
            rows.append({
                "Model": model_name, "mAP@0.5": m["mAP50"], "mAP@0.5:0.95": m["mAP50-95"],
                "Precision": m["precision"], "Recall": m["recall"], "F1-Score": m["f1"],
            })
        df = pd.DataFrame(rows)
        df.to_csv(self.results_dir / "model_comparison.csv", index=False)
        self.create_performance_plots(df)
        return df

    def create_performance_plots(self, df_results):
        plt.style.use("seaborn-v0_8")
        metrics = ["mAP@0.5", "mAP@0.5:0.95", "Precision", "Recall", "F1-Score"]
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle("YOLO Models Performance Comparison on DOTA", fontsize=16)
        for i, metric in enumerate(metrics):
            ax = axes[i // 3, i % 3]
            bars = ax.bar(df_results["Model"], df_results[metric],
                          color=plt.cm.Set3(np.linspace(0, 1, len(df_results))))
            ax.set_title(f"{metric} Comparison")
            ax.set_ylabel(metric)
            ax.tick_params(axis="x", rotation=45)
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2.0, h + 0.01, f"{h:.3f}",
                        ha="center", va="bottom")
        ax = axes[1, 2]
        x = np.arange(len(df_results["Model"].values))
        width = 0.15
        for i, metric in enumerate(metrics):
            ax.bar(x + i * width, df_results[metric], width, label=metric, alpha=0.8)
        ax.set_xlabel("Models")
        ax.set_ylabel("Score")
        ax.set_title("All Metrics Comparison")
        ax.set_xticks(x + width * 2)
        ax.set_xticklabels(df_results["Model"].values, rotation=45)
        ax.legend()
        plt.tight_layout()
        plt.savefig(self.results_dir / "performance_comparison.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(df_results.set_index("Model")[metrics].T, annot=True, cmap="YlOrRd", fmt=".3f", ax=ax)
        ax.set_title("Performance Heatmap - All Models and Metrics")
        plt.tight_layout()
        plt.savefig(self.results_dir / "performance_heatmap.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    @staticmethod
    def create_class_wise_analysis():
        print("Class-wise analysis would require detailed validation results")
        print("Available classes:", list(DOTA_CLASSES))


def main():
    epochs = config.NUM_EPOCHS

    print("YOLO Models Fine-tuning on DOTA Dataset\n" + "=" * 50)
    # Train and validate on DATA_ROOT by default; point val_yaml at a separate
    # processed dataset if you have one.
    train_processor = DOTADatasetProcessor(config.DATA_ROOT)
    train_yaml = train_processor.prepare_dataset()
    val_yaml = train_yaml  # set to a separate processed dataset if you have one

    trainer = YOLOModelTrainer(train_yaml, val_yaml)
    trainer.train_all_models(epochs=epochs)
    results_df = trainer.create_results_summary()
    if results_df is not None:
        print("\nFinal Results Summary:\n" + "=" * 80)
        print(results_df.to_string(index=False))
        best = results_df.loc[results_df["mAP@0.5:0.95"].idxmax()]
        print(f"\nBest model: {best['Model']}  mAP@0.5:0.95={best['mAP@0.5:0.95']:.4f}")
    trainer.create_class_wise_analysis()
    print(f"\nAll results saved in: {trainer.results_dir}")


if __name__ == "__main__":
    main()
