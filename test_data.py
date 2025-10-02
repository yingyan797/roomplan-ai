from ml_pipeline import load_data, train_multistage_pipeline
import os

pipeline = load_data(filenames=["dataset/"+fname for fname in os.listdir("dataset/") if fname.startswith("training_")])