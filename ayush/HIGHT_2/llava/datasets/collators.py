from dataclasses import dataclass
from typing import Dict, Optional, Sequence, List, Any

import torch 
from torch_geometric.data import Batch, Data
import transformers

from llava.constants import IGNORE_INDEX

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images

        return batch
    

@dataclass
class GraphDataCollatorForSupervisedDataset(object):
    """Collate graph-QA examples for supervised fine-tuning."""
    
    tokenizer: transformers.PreTrainedTokenizer
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if 'graph' in instances[0]:
            g = Batch.from_data_list([self._convert_dict_to_Data(instance["graph"]) for instance in instances])
            batch['graphs'] = g
        return batch
    
    def add_node_attr(self,
        data: Data,
        value: Any,
        attr_name: Optional[str] = None,
    ) -> Data:
        # TODO Move to `BaseTransform`.
        if attr_name is None:
            if data.x is not None:
                x = data.x.view(-1, 1) if data.x.dim() == 1 else data.x
                data.x = torch.cat([x, value.to(x.device, x.dtype)], dim=-1)
            else:
                data.x = value
        else:
            data[attr_name] = value

        return data
    def _convert_dict_to_Data(self, data_dict: Dict) -> Data:
        data = Data(
            x=torch.asarray(data_dict['node_feat']),
            edge_attr=torch.asarray(data_dict['edge_feat']),
            edge_index=torch.asarray(data_dict['edge_index']),
        )
        if "lap_pe" in data_dict.keys():
            data = self.add_node_attr(data, data_dict["lap_pe"], attr_name="lap_pe")
        if "num_part" in data_dict.keys():
            data = self.add_node_attr(data, data_dict["num_part"], attr_name="num_part")
        return data
