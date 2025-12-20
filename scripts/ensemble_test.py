import os
import glob
import zipfile
import numpy as np
import pandas as pd

PRED_DIR = "preds"
files = sorted(glob.glob(os.path.join(PRED_DIR, "prediction_*_test.csv")))
assert len(files) > 0, "Aucun fichier prediction_*_test.csv trouvé."

dfs = [pd.read_csv(f) for f in files]

base_id = dfs[0]["ID"].values
for i, df in enumerate(dfs[1:], 1):
    if not np.array_equal(base_id, df["ID"].values):
        raise ValueError(f"IDs non alignés: {files[i]}")

stack = np.stack([df["Task2"].values for df in dfs], axis=1)  # (N, M)
ens = stack.mean(axis=1)

out_csv = os.path.join(PRED_DIR, "prediction_ensemble_test.csv")
out_zip = os.path.join(PRED_DIR, "submission_ensemble_test.zip")

out = pd.DataFrame({"ID": base_id, "Task2": ens})
out.to_csv(out_csv, index=False)

with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
    zf.write(out_csv, arcname=os.path.basename(out_csv))

print("✅ models:", len(files))
print("✅ ensemble csv:", out_csv)
print("✅ ensemble zip:", out_zip)
