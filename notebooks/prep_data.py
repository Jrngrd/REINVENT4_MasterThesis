import os
from datasets import load_dataset
from reinvent.chemistry.standardization.rdkit_standardizer import RDKitStandardizer

def ensure_directory_exists(path):
    """Creates the directory if it doesn't already exist."""
    if not os.path.isdir(path):
        os.makedirs(path)

def process_tack_data(base_path, allowed_tokens):
    """Downloads, filters, and splits the TACK dataset."""
    print("Downloading tack data")
    tack_ds = load_dataset("ailab-bio/TACK")
    tack_ds_train = tack_ds["train"].to_pandas()
    
    tack_smiles = tack_ds_train["SMILES"]

    #FIXME Temporary: Replace all @ with $ to avoid tokenization issues with the model

    # RDKit standarization
    standardizer = RDKitStandardizer(filter_configs=None, isomeric=False)
    tack_smiles = tack_smiles.apply(standardizer.apply_filter)

    # Remove all None values that may have been introduced by the standardization step
    tack_smiles = tack_smiles.dropna()

    print(f"After standardization, TACK dataset has {len(tack_smiles)} molecules", tack_smiles)
    # Filter out SMILES that contain tokens not supported by the model
    tack_smiles = tack_smiles[tack_smiles.apply(lambda x: all(token in allowed_tokens for token in x))]

    n_head = int(0.8 * len(tack_smiles))
    n_tail = len(tack_smiles) - n_head
    print(f"number of molecules for: training={n_head}, validation={n_tail}")

    train, validation = tack_smiles.head(n_head), tack_smiles.tail(n_tail)
    
    train.to_csv(f"{base_path}/tack_train.smi", sep="\t", index=False, header=False)
    validation.to_csv(f"{base_path}/tack_validation.smi", sep="\t", index=False, header=False)
    print(f"Finished writing curated data to folder {base_path}")

def process_synthetic_data(base_path, allowed_tokens):
    """Downloads, filters, and saves the PROTAC synthetic dataset."""
    print("Downloading synthetic data")
    synthetic_ds = load_dataset("ailab-bio/PROTAC-Splitter-Dataset", "clustered")
    
    for split in synthetic_ds:
        # Remove labels and filter by tokens
        ds_split = synthetic_ds[split].remove_columns("labels")

        #FIXME Temporary: Replace all @ with $ to avoid tokenization issues with the model

        # standardize SMILES and filter out those that contain tokens not supported by the model
        standardizer = RDKitStandardizer(filter_configs=None, isomeric=False)
        ds_split = ds_split.map(lambda x: {"text": standardizer.apply_filter(x["text"])})

        # remove all None values that may have been introduced by the standardization step
        ds_split = ds_split.filter(lambda x: x["text"] is not None)
        

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

            

