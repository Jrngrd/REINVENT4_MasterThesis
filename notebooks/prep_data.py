import os
from datasets import load_dataset
from reinvent.chemistry.standardization.rdkit_standardizer import RDKitStandardizer

def ensure_directory_exists(path):
    """Creates the directory if it doesn't already exist."""
    if not os.path.isdir(path):
        os.makedirs(path)

def split_dataset(dataset, split):
    """Splits a Hugging Face dataset into training and validation (hold out) sets."""
    if split == "train":
        return dataset[dataset["SMILES_Held_Out"] == False]["SMILES"]
    else:
        return dataset[dataset["SMILES_Held_Out"] == True]["SMILES"]

def standardize_smiles(dataset):
    """Standardizes SMILES strings in a dataset using RDKit and removes any that cannot be standardized."""
    standardizer = RDKitStandardizer(filter_configs=None, isomeric=False)
    ds = dataset.apply(standardizer.apply_filter)
    return ds.dropna()

def filter_and_save(dataset, allowed_tokens, base_path, filename):
    """Filters out SMILES strings that contain tokens not in the allowed set, and saves the resulting dataset to a file."""
    dataset = dataset[dataset.apply(lambda x: all(token in allowed_tokens for token in x))]
    dataset.to_csv(f"{base_path}/{filename}", sep="\t", index=False, header=False)
    

def process_tack_data(base_path, allowed_tokens):
    """Downloads, filters, and splits the TACK dataset."""
    print("Downloading tack data")
    tack_ds = load_dataset("ailab-bio/TACK")
    tack_ds = tack_ds["train"].to_pandas()

    tack_smiles_train = split_dataset(tack_ds, "train")
    print(f"Before standardization, TACK training dataset has {len(tack_smiles_train)} molecules")
    tack_smiles_val = split_dataset(tack_ds, "val")
    
    # RDKit standarization
    tack_smiles_train = standardize_smiles(tack_smiles_train)
    tack_smiles_val = standardize_smiles(tack_smiles_val)

    print(f"After standardization, TACK training dataset has {len(tack_smiles_train)} molecules")
    print(f"After standardization, TACK held-out dataset has {len(tack_smiles_val)} molecules")
    
    # Filter out SMILES that contain tokens not supported by the model
    filter_and_save(tack_smiles_train, allowed_tokens, base_path, "tack_train.smi")
    filter_and_save(tack_smiles_val, allowed_tokens, base_path, "tack_validation.smi")
    print(f"Finished writing curated data to folder {base_path}")

def process_synthetic_data(base_path, allowed_tokens):
    """Downloads, filters, and saves the PROTAC synthetic dataset."""
    print("Downloading synthetic data")
    synthetic_ds = load_dataset("ailab-bio/PROTAC-Splitter-Dataset", "clustered")
    
    for split in synthetic_ds:
        # Remove labels and filter by tokens
        ds_split = synthetic_ds[split].remove_columns("labels")

        # standardize SMILES and filter out those that contain tokens not supported by the model
        standardizer = RDKitStandardizer(filter_configs=None, isomeric=False)
        ds_split = ds_split.map(lambda x: {"text": standardizer.apply_filter(x["text"])})

        # remove all None values that may have been introduced by the standardization step
        ds_split = ds_split.filter(lambda x: x["text"] is not None)
        
        # Filter out SMILES that contain tokens not supported by the model
        ds_split = ds_split.filter(lambda x: all(token in allowed_tokens for token in x["text"]))
        
        print(f"After filtering, {split} split has {len(ds_split)} molecules")
        ds_split.to_csv(f"{base_path}/synthetic_{split}.smi", header=False)
    
    print(f"Finished writing synthetic data to {base_path}")

def prep_data(args):
    """Main orchestration method for data preparation."""
    base_path = os.path.join(os.getcwd(), args.data_folder)
    ensure_directory_exists(base_path)

    # FIXME: temporary solution to filter out SMILES with unallowed tokens for reinvent.prior
    reinvent_prior_allowed_tokens = {
        '[S+]', '[N+]', '[N-]', '[O-]', '[n+]', '[nH]', '%10', 'Cl',')', 'S', '^', '2', 'O',  '4', 
        '=', 'C', '1', '9', '6', 's',  '5', 'Br', 
        'o', '7', '(', 'n', '-', '8', 'N', 'F', '3',  'c',  '#', '$'
    }

    process_tack_data(base_path, reinvent_prior_allowed_tokens)
    process_synthetic_data(base_path, reinvent_prior_allowed_tokens)

            

