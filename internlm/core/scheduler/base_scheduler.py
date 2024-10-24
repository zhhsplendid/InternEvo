#!/usr/bin/env python
# -*- encoding: utf-8 -*-

# adopted from https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/engine

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Iterable

import torch

from internlm.apis import InferenceParams
from internlm.core.engine import Engine


class BaseScheduler(ABC):
    """A basic helper class to control the process of training or evaluation.
    It mainly composes of forward_backward_step for gradient backward and
    optimizer_step for parameters update.
    For the convenience to enable FP16, we aggregate all codes that contain the
    control of FP16 in class schedule.

    Args:
        data_process_func (Callable, optional): The preprocessing function which receives a batch of data and arranges
            them into data and label.
    """

    def __init__(self, data_process_func: Callable = None):
        self.data_process_func = data_process_func

    @abstractmethod
    def pre_processing(self, engine: Engine):
        """To perform actions before running the schedule.

        Args:
           engine (internlm.core.Engine): InternLM engine for training and inference.
        """
        pass

    def _load_micro_batch(self, data: Dict, label: torch.Tensor, offset: int, bsz_stride: int):
        """
        For pp, it will cut one fully batch into micro batch in pipeline concept.
        For nopp, it will cut one fully batch into small batch based on gradient accumulate size.

        A special case is that pp uses a 'non-packed-dateset' (such as evaluation dataset),
        so the data of batch is unpacked and 'bsz_stride' is equal to 'micro_bsz'.
        In all other cases 'bsz_stride' should be equal to 1.
        """
        assert isinstance(data, dict)

        micro_batch_data = {}
        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                micro_batch_data[k] = v[offset : offset + bsz_stride]
            elif isinstance(v, InferenceParams):
                v.set_batch_offset(offset, bsz_stride)
                micro_batch_data[k] = v
            elif isinstance(v, (list, tuple)):
                micro_batch_data[k] = v[offset : offset + bsz_stride]
            else:
                raise NotImplementedError(f"value of type {type(v)} is not supported")

        if isinstance(label, torch.Tensor):
            micro_batch_label = label[offset : offset + bsz_stride]
        elif isinstance(label, Dict):
            micro_batch_label = {k: v[offset : offset + bsz_stride] if v.dim() > 0 else v for k, v in label.items()}
        else:
            micro_batch_label = label
        return micro_batch_data, micro_batch_label

    @abstractmethod
    def forward_backward_step(
        self,
        engine: Engine,
        data_iter: Iterable,
        forward_only: bool,
        return_loss: bool = True,
        return_output_label: bool = True,
    ):
        """The process function over a batch of dataset for training or evaluation.

        Args:
            engine (internlm.core.Engine): InternLM engine for training and inference.
            data_iter (Iterable): Data iterator from which get a batch of data, obtained by calling iter(dataloader).
            forward_only (bool): If True, the process won't include backward.
            return_loss (bool, optional): If False, the loss won't be returned.
            return_output_label (bool, optional): If False, the output and label won't be returned.
        """
        pass

    @staticmethod
    def _call_engine(engine: Engine, inputs: Any):
        """Calls the engine with the given inputs.

        Args:
            engine (internlm.core.Engine): InternLM engine for training and inference.
            inputs (Any): The inputs to the engine, can be of type torch.Tensor, list, tuple, or dict.
        """
        if isinstance(inputs, torch.Tensor):
            return engine(inputs)
        elif isinstance(inputs, (list, tuple)):
            return engine(*inputs)
        elif isinstance(inputs, dict):
            return engine(**inputs)
        else:
            raise TypeError(
                f"Expected engine inputs to be of type torch.Tensor, list, tuple, or dict, but got {type(inputs)}"
            )

    @staticmethod
    def _call_engine_criterion(engine: Engine, outputs: Any, labels: Any):
        """Calls the engine's criterion with the given outputs and labels.

        Args:
            engine (internlm.core.Engine): InternLM engine for training and inference.
            outputs (Any): The outputs from the model, can be of type torch.Tensor, list, tuple, or dict.
            labels (Any): The labels for the outputs, can be of type torch.Tensor, list, tuple, or dict.
        """
        assert isinstance(
            outputs, (torch.Tensor, list, tuple, dict)
        ), f"Expect output of model is (torch.Tensor, list, tuple), got {type(outputs)}"
        if isinstance(outputs, torch.Tensor):
            outputs = (outputs,)
        if isinstance(labels, torch.Tensor):
            labels = (labels,)

        if isinstance(outputs, (tuple, list)) and isinstance(labels, (tuple, list)):
            return engine.criterion(*outputs, *labels)
        elif isinstance(outputs, (tuple, list)) and isinstance(labels, dict):
            return engine.criterion(*outputs, **labels)
        elif isinstance(outputs, dict) and isinstance(labels, dict):
            return engine.criterion(**outputs, **labels)
        elif isinstance(outputs, dict) and isinstance(labels, (list, tuple)):
            raise ValueError(f"Expected labels to be a dict when the model outputs are dict, but got {type(labels)}")
        else:
            raise TypeError(
                f"Expected model outputs and labels to be of type torch.Tensor ' \
                '(which is auto-converted to tuple), list, tuple, or dict, ' \
                'but got {type(outputs)} (model outputs) and {type(labels)} (labels)"
            )
