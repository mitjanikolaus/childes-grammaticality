import os
import pandas as pd

from grammaticality_annotation.data import DATA_FILE_ANNOTATED_CHILDES_DB
from load_childes_db_data import DB_VERSION
from utils import PROJECT_ROOT_DIR

DATA_FILE_CHILDES_DB_METADATA = os.path.join(PROJECT_ROOT_DIR, "data", "manual_annotation", "childes_db_export", "metadata.csv")
DATA_FILE_CHILDES_DB_VARIABLES = os.path.join(PROJECT_ROOT_DIR, "data", "manual_annotation", "childes_db_export", "variables.csv")
DATA_FILE_ANNOTATED_ALL = os.path.join(PROJECT_ROOT_DIR, "data", "manual_annotation", "childes_db_export", "data.csv")


if __name__ == "__main__":
    metadata = pd.DataFrame.from_records([{
        "dataset_name": "grammaticality",
        "entity_type": "utterances",
        "childes_db_version": DB_VERSION,
        "dataset_version": "1",
        "tag_type": "manual",
        "model_version": "1",
        "date_of_release": "2024-05-01",
        "contact": "mitja.nikolaus@posteo.de",
        "citation": "https://doi.org/10.48550/arXiv.2403.14208",
    }])
    metadata.to_csv(DATA_FILE_CHILDES_DB_METADATA, index=False)

    variables_meta = pd.DataFrame.from_records([{
        "variable_id": "1",
        "variable_name": "is_grammatical",
        "data_type": "float",
        "values": "[-1,0,1]",
    }])
    variables_meta.to_csv(DATA_FILE_CHILDES_DB_VARIABLES, index=False)

    print("creating single file with all annotated utterances..")

    print('loading data...')
    data_manual_annotations = pd.read_csv(DATA_FILE_ANNOTATED_CHILDES_DB)

    data_manual_annotations.rename(columns={"id": "utterance_id", "transcript_file": "transcript_id"}, inplace=True)
    data_manual_annotations = data_manual_annotations[["utterance_id", "is_grammatical"]]
    data_manual_annotations.dropna(inplace=True)

    print(f"saving {len(data_manual_annotations)} utterances")
    data_manual_annotations.to_csv(DATA_FILE_ANNOTATED_ALL, index=False)
