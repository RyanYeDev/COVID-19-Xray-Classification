import kagglehub

# Download latest version
path = kagglehub.dataset_download("khoongweihao/covid19-xray-dataset-train-test-sets")

print("Path to dataset files:", path)