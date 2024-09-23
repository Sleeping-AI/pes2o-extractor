import os
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
from datasets import load_from_disk

dataset = load_from_disk("/ammar_storage/pesco")

def extract_abstract(dataset, filter_source_value=None):
    extract_all = input("Do you want to extract all entries? (yes/no): ").strip().lower()

    if extract_all == "yes":
        num_entries = len(dataset)
    else:
        num_entries = 50

    checkpoint_interval = int(input("Enter the checkpoint interval: "))

    output_folder = input("Enter the checkpoint folder name: ")

    while True:
        final_parquet_file_name = input("Enter the final Parquet file name (with .parquet extension): ")
        if final_parquet_file_name.lower().endswith('.parquet'):
            break
        else:
            print("Invalid file name. Please ensure the file name ends with '.parquet'.")

    all_entries = dataset.select(range(num_entries))

    if filter_source_value:
        filtered_entries = [entry for entry in all_entries if entry['source'] == filter_source_value]
        for i, entry in enumerate(filtered_entries):
            print(f"Found and Extracted entry {i + 1}")
    else:
        filtered_entries = all_entries
        for i, entry in enumerate(filtered_entries):
            print(f"Extracted entry {i + 1}")

    data = {key: [] for key in dataset.features.keys()}
    data['title'] = []

    os.makedirs(output_folder, exist_ok=True)
    print(f"Folder '{output_folder}' created in the working directory.")

    checkpoint_counter = 0
    checkpoint_files = []
    total_entries_processed = 0

    for i, entry in enumerate(filtered_entries):
        text = entry['text']
        title = text.split('\n')[0] if '\n' in text else text
        data['title'].append(title)

        for key in dataset.features.keys():
            data[key].append(entry[key])

        total_entries_processed += 1

        if total_entries_processed % checkpoint_interval == 0:
            checkpoint_counter += 1
            df = pd.DataFrame(data)
            columns = ['title', 'text'] + [key for key in dataset.features.keys() if key not in ['title', 'text']]
            df = df[columns]

            table = pa.Table.from_pandas(df)
            checkpoint_file_path = os.path.join(output_folder, f"checkpoint_{checkpoint_counter}.parquet")
            pq.write_table(table, checkpoint_file_path)

            checkpoint_files.append(checkpoint_file_path)
            print(f"Checkpoint {checkpoint_counter} saved to {checkpoint_file_path}")

            data = {key: [] for key in dataset.features.keys()}
            data['title'] = []

    if total_entries_processed % checkpoint_interval != 0:
        df = pd.DataFrame(data)
        columns = ['title', 'text'] + [key for key in dataset.features.keys() if key not in ['title', 'text']]
        df = df[columns]

        table = pa.Table.from_pandas(df)
        final_checkpoint_file_path = os.path.join(output_folder, "final_checkpoint.parquet")
        pq.write_table(table, final_checkpoint_file_path)

        checkpoint_files.append(final_checkpoint_file_path)
        print(f"Final checkpoint saved to {final_checkpoint_file_path}")

    combined_dfs = [pq.read_table(file).to_pandas() for file in checkpoint_files]
    final_df = pd.concat(combined_dfs, ignore_index=True)
    final_table = pa.Table.from_pandas(final_df)

    final_parquet_file_path = os.path.join(os.getcwd(), final_parquet_file_name)
    pq.write_table(final_table, final_parquet_file_path)

    print(f"Final Parquet file '{final_parquet_file_name}' saved to {final_parquet_file_path} in the working directory.")

extract_abstract(dataset, filter_source_value="s2orc/train")
