import shutil
from train import do_learning

#0.0001
DATA_DIR = "/ssd1/xuefeng/"
ARTIFACT_DIR = "/home/xuefeng/stoic2021/training/output/fold0_swin"

if __name__ == "__main__":
    # Substitute do_learning for your training function.
    # It is recommended to write artifacts (e.g. model weights) to ARTIFACT_DIR during training.
    artifacts = do_learning(DATA_DIR, ARTIFACT_DIR)

    # When the learning has completed, any artifacts should have been saved to ARTIFACT_DIR.
    # Alternatively, you can copy artifacts to ARTIFACT_DIR after the learning has completed:
    for artifact in artifacts:
        shutil.copy(artifact, ARTIFACT_DIR)

    print("Training completed.")
