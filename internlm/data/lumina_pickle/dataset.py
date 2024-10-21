import os
import pickle
import yaml

import torch
from torch.utils.data import Dataset

from internlm.core.context import ParallelMode
from internlm.core.context import global_context as gpc

from internlm.utils.logger import get_logger

logger = get_logger(__file__)

class LuminaPickleDataset(Dataset):
    def __init__(self, data_yaml: str):
        logger.info(f"read data meta yaml from {data_yaml}")
        # TODO(zhenghuihuang): cache to disk?
        self.meta_list, self.record_list = self._load_data_yaml(data_yaml)

    def __len__(self) -> int:
        total_len = sum(meta["len"] for meta in meta_list)
        return total_len

    def __getitem__(self, idx: int):
        meta_idx, idx_in_meta = self.tie_index_to_meta(index)

        try:
            return self.get_item_func(meta_idx, idx_in_meta)
        except Exception as e:
            logger.info(
                f"Item {index} errored, record:\n"
                f"{self.record_list[meta_idx][idx_in_meta]}\n"
                f"Error:\n"
                f"{traceback.format_exc()}"
            )
            if idx_in_meta != 0:
                return self[index - 1]
            else:
                return self[index + self.meta_list[meta_idx]["len"] - 1]

    def _load_data_yaml(self, data_yaml: str):
        meta_list = []
        record_list = []
        with open(data_yaml, "r") as yaml_fin:
            data_meta = yaml.load(yaml_fin, Loader=yaml.FullLoader)
            for meta in data_meta["META"]:
                record_json_path = meta["path"]
                with open(record_json_path) as record_json_fin:
                    record_json = json.load(record_json_fin)
                    record_list.append(record_json)
                    meta["len"] = len(record_json)

                if "type" not in meta:
                    meta["type"] = "default"
                 meta["item_len_list"] = [r["len"] for r in record_json]
        return meta_list, record_list

    def tie_index_to_meta(self, idx: int):
        # Initialize the starting index
        start_idx = 0

        # Iterate through the list of dictionaries
        for i, meta in enumerate(self.meta_list):
            # Calculate the ending index for the current collection
            end_idx = start_idx + meta["len"]

            # Check if the given index falls within the current collection
            if start_idx <= idx < end_idx:
                # Calculate the new index within the current collection
                new_index = idx - start_idx
                return i, new_index

            # Update the starting index for the next collection
            start_idx = end_idx

        # If the index is out of range of all collections, raise an error
        raise IndexError("Index out of range")

    def get_item_func(self, meta_idx, idx_in_meta):
        record_item = self.record_list[meta_idx][idx_in_meta]
        # Why origin code has deepcopy?
        #data_item = copy.deepcopy(record_item)

        with open(record_item["file"], "rb") as f:
            data_item = pickle.load(f)
        okens = data_item["token"]
        labels = data_item["label"]
        assert len(tokens) == len(labels)

        return tokens, labels
