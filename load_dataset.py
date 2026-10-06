from datasets import load_dataset

dataset = load_dataset("batuhanmtl/job_resume_fit")

print(dataset)

train_data = dataset["train"]

print(train_data)

print(train_data.column_names)

print(train_data[0])

df_fit = train_data.to_pandas()

print(df_fit.head())
print(df_fit.shape)
print(df_fit.columns)

df_fit.info()