from ultralytics import YOLO
import shutil
import os
import matplotlib.pyplot as plt
import pandas as pd

# ------------------ TRAINING ------------------
# Load base YOLOv8 model
model = YOLO("yolov8n.pt")

# Train the model
results = model.train(
    data="dataset/data.yaml",   # make sure the path is correct
    epochs=20,
    imgsz=640,
    batch=4,
    device="cpu"
)

# ------------------ SAVE TRAINED MODEL ------------------
default_path = model.trainer.best
os.makedirs("model", exist_ok=True)
custom_path = os.path.join("model", "best.pt")
shutil.copy(default_path, custom_path)
print(f"✅ Model saved successfully at: {os.path.abspath(custom_path)}")

# ------------------ LOAD TRAINING METRICS ------------------
# YOLO saves results.csv automatically (contains all metrics per epoch)
results_csv = os.path.join(model.trainer.save_dir, "results.csv")

if os.path.exists(results_csv):
    df = pd.read_csv(results_csv)

    # Make sure graphs directory exists
    os.makedirs("graphs", exist_ok=True)

    # ------------------ GRAPH 1: Training Loss ------------------
    plt.figure()
    plt.plot(df["epoch"], df["train/box_loss"], label="Box Loss")
    plt.plot(df["epoch"], df["train/cls_loss"], label="Class Loss")
    plt.plot(df["epoch"], df["train/dfl_loss"], label="DFL Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss per Epoch")
    plt.legend()
    plt.grid(True)
    plt.savefig("graphs/training_loss.png")
    plt.close()

    # ------------------ GRAPH 2: Precision ------------------
    plt.figure()
    plt.plot(df["epoch"], df["metrics/precision(B)"], label="Precision")
    plt.xlabel("Epoch")
    plt.ylabel("Precision")
    plt.title("Precision per Epoch")
    plt.legend()
    plt.grid(True)
    plt.savefig("graphs/precision.png")
    plt.close()

    # ------------------ GRAPH 3: Recall ------------------
    plt.figure()
    plt.plot(df["epoch"], df["metrics/recall(B)"], label="Recall", color="orange")
    plt.xlabel("Epoch")
    plt.ylabel("Recall")
    plt.title("Recall per Epoch")
    plt.legend()
    plt.grid(True)
    plt.savefig("graphs/recall.png")
    plt.close()

    # ------------------ GRAPH 4: mAP ------------------
    plt.figure()
    plt.plot(df["epoch"], df["metrics/mAP50(B)"], label="mAP@50")
    plt.plot(df["epoch"], df["metrics/mAP50-95(B)"], label="mAP@50-95")
    plt.xlabel("Epoch")
    plt.ylabel("mAP")
    plt.title("Mean Average Precision per Epoch")
    plt.legend()
    plt.grid(True)
    plt.savefig("graphs/mAP.png")
    plt.close()

    # ------------------ PRINT FINAL RESULTS ------------------
    last_row = df.iloc[-1]
    print("\n📊 Final Model Performance:")
    print(f" Precision: {last_row['metrics/precision(B)']:.4f}")
    print(f" Recall: {last_row['metrics/recall(B)']:.4f}")
    print(f" mAP@50: {last_row['metrics/mAP50(B)']:.4f}")
    print(f" mAP@50-95: {last_row['metrics/mAP50-95(B)']:.4f}")

    print("\n📈 Graphs saved in:", os.path.abspath("graphs"))

else:
    print("⚠️ Could not find results.csv. Make sure YOLO training completed successfully.")
