import os
import pandas as pd
import shutil

image_dir = os.path.join("/ssd1/xuefeng/data/mha/")
image_out_dir = os.path.join("/ssd1/xuefeng/metadata/test_mha/")
reference_test_path = os.path.join("/ssd1/xuefeng/metadata/test/test.csv")
df_test = pd.read_csv(reference_test_path)
df_test_list = df_test.apply(lambda row: os.path.join(image_dir, str(row["PatientID"]) + ".mha"), axis=1).to_list()
for i in range(len(df_test_list)):
    file_name = df_test_list[i].split('/')[-1]
    shutil.copyfile(df_test_list[i], image_out_dir + file_name)

