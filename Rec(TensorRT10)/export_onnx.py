from rec.data.loaders import build_dataset
from rec.data.mix_wrappers import MixDictBatch
from mt_tools.dataloader.orc_data import ORCIterDataset
from rec.models.scaling_law_rank_wo_rt_seq_hstu.model import GRRankingModel
from torch.utils.data import DataLoader, IterableDataset
import torch
from torch import nn
from torchrec import KeyedJaggedTensor, JaggedTensor
from typing import Dict, List
import onnx

cols_dense = ""
cols_kjt = ""

def build_dataset(batch_wrapper=MixDictBatch):
    eval_dataset = ORCIterDataset(
        path = "",
        batch_size=1,
        cols_dense=cols_dense.split(","),
        cols_kjt=cols_kjt.split(","),
        batch_wrapper=batch_wrapper,
    )
    return eval_dataset

def get_dataloader(dataset: IterableDataset):
    dataloader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=2,
        pin_memory=True,
        prefetch_factor=4,
        collate_fn=lambda x: x,
    )
    return dataloader

class GRRankingWrappedModel(nn.Module):
    def __init__(self, model, model_input_names, kjt_name_map):
        super(GRRankingWrappedModel, self).__init__()
        self.model = model
        self.model_input_names = model_input_names
        self.kjt_name_map = kjt_name_map

    def forward(self, inputs):
        feat = {}
        inputs = list(inputs)
        idx = 0
        for name in self.model_input_names:
            if name in self.kjt_name_map.keys():
                keys = [t[0] for t in  self.kjt_name_map[name]]
                values = [inputs[t[1]] for t in  self.kjt_name_map[name]]
                lens = [len(v) for v in values]
                feat[name] = KeyedJaggedTensor(keys=keys, values=torch.cat(values), lengths=torch.IntTensor(lens)).to('cuda:0', non_blocking=True)
                idx += len(keys)
            else:
                feat[name] = inputs[idx].to('cuda:0', non_blocking=True)
                idx += 1
        # print(f'inputs: {len(inputs)}')
        # print(f'idx: {idx}')
        # print(f'feat: {len(feat)}')
        # print(f'target_poi_seq_length shape1: {feat["target_poi_seq_length"].shape}')
        return self.model(feat)

eval_data = build_dataset()
model = GRRankingModel()
loader = get_dataloader(eval_data)
model.eval()
model.to('cuda:0')

onnx_input_names = []
onnx_input_tensors = []
model_input_names = []
kjt_name_map = {}
dynamic_names = []
batch = next(iter(loader))
for k, v in batch.items():
    if isinstance(v, torch.Tensor):
        model_input_names.append(k)
        onnx_input_names.append(k)
        onnx_input_tensors.append(v)
        if v.shape[0] % 28 == 0:
            dynamic_names.append(k)
            if v.dim() != 1:
                raise ValueError(f"dim of {k} should be 1")
    elif isinstance(v, KeyedJaggedTensor):
        model_input_names.append(k)
        kjt : Dict[str, JaggedTensor] = v.to_dict()
        for kjt_k, kjt_v in kjt.items():
            onnx_input_names.append(f'{k}-kjt-{kjt_k}')
            onnx_input_tensors.extend(kjt_v.to_dense())
            # for item in kjt_v.to_dense():
            #     # print(f"k:{kjt_k}, vshape: {item.shape[0]}")
            if k in kjt_name_map.keys():
                kjt_name_map[k].append((kjt_k, len(onnx_input_tensors)-1))
            else:
                kjt_name_map[k] = [(kjt_k, len(onnx_input_tensors)-1)]

onnx_output_names = ['ctr_prob', 'cxr_prob', 'logits', 'loss']
# onnx_names = onnx_input_names+onnx_output_names
# dynamic_axes = {name: {0: 'batch_size'} for name in onnx_names}

input_tensors = tuple(onnx_input_tensors)
wrapped_model = GRRankingWrappedModel(model, model_input_names, kjt_name_map)

ctr_prob, cxr_prob, logits, loss = wrapped_model(input_tensors)
# print(ctr_prob)
# print(dynamic_names)
torch.onnx.export(
    wrapped_model,
    (input_tensors,),
    "onnx_bs1_dynamicshape/GRRankingModel.onnx",
    do_constant_folding=True,
    opset_version=12,
    input_names=onnx_input_names,
    output_names=onnx_output_names,
    verbose=True,
    dynamic_axes={name: {0: 'seq_length'} for name in dynamic_names},
)
