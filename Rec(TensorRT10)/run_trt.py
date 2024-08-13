from rec.data.loaders import build_dataset
from rec.data.mix_wrappers import MixDictBatch
from mt_tools.dataloader.orc_data import ORCIterDataset
from rec.models.scaling_law_rank_wo_rt_seq_hstu.model import GRRankingModel
from torch.utils.data import DataLoader, IterableDataset
import torch
from torch import nn
from torchrec import KeyedJaggedTensor, JaggedTensor
from typing import Dict, List
import tensorrt as trt
import cupy as cp
import numpy as np
from cal_diff import diff_outputs
cols_dense = ""

def build_dataset(batch_wrapper=MixDictBatch):
    eval_dataset = ORCIterDataset(
        path="",
        batch_size=1,
        cols_dense=cols_dense.split(","),
        cols_kjt=cols_kjt.split(","),
        batch_wrapper=batch_wrapper,
    )
    return eval_dataset

def get_dataloader(dataset: IterableDataset):
    dataloader = DataLoader(
        dataset,
        batch_size=1,
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
        return self.model(feat)

eval_data = build_dataset()
model = GRRankingModel()
loader = get_dataloader(eval_data)
model.eval()
model.to('cuda:0')

wrapped_model = None
data_loader_iterator = iter(loader)
max_diff = 0
for _ in range(100):
    onnx_input_names = []
    onnx_input_tensors = []
    model_input_names = []
    kjt_name_map = {}
    batch = next(data_loader_iterator)
    # print(batch)
    for k, v in batch[0].items():
        if isinstance(v, torch.Tensor):
            model_input_names.append(k)
            onnx_input_names.append(k)
            onnx_input_tensors.append(v)
        elif isinstance(v, KeyedJaggedTensor):
            model_input_names.append(k)
            kjt : Dict[str, JaggedTensor] = v.to_dict()
            for kjt_k, kjt_v in kjt.items():
                onnx_input_names.append(f'{k}-kjt-{kjt_k}')
                onnx_input_tensors.extend(kjt_v.to_dense())
                if k in kjt_name_map.keys():
                    kjt_name_map[k].append((kjt_k, len(onnx_input_tensors)-1))
                else:
                    kjt_name_map[k] = [(kjt_k, len(onnx_input_tensors)-1)]

    onnx_output_names = ['ctr_prob', 'cxr_prob', 'logits', 'loss']
    onnx_names = onnx_input_names+onnx_output_names

    input_tensors = tuple(onnx_input_tensors)
    if not wrapped_model:
        wrapped_model = GRRankingWrappedModel(model, model_input_names, kjt_name_map)

    # test inference
    ctr_prob, cxr_prob, logits, loss = wrapped_model(input_tensors)
    # print(f"pytorch ouput: {ctr_prob}")

    outputs = [ctr_prob, cxr_prob, logits, loss, loss]

    output_dict = {
        'ctr_prob': [outputs[0]],
        'cxr_prob': [outputs[1]],
    }
    def load_engine(engine_path):
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())


    engine = load_engine('grranking_bs1.trt')
    context = engine.create_execution_context()
    stream = cp.cuda.Stream()
    context.set_optimization_profile_async(0, stream.ptr)

    input_tensor_mp = {}
    for i in range(len(onnx_input_names)):
        input_tensor_mp[onnx_input_names[i]] = input_tensors[i]

    engine_input_names = []
    for i in range(engine.num_io_tensors):
        engine_input_names.append(engine.get_tensor_name(i))

    d_inputs = []
    h_outputs = []
    d_outputs = []
    for engine_input_name in engine_input_names:
        if engine_input_name in input_tensor_mp.keys():
            tensor = input_tensor_mp[engine_input_name]
            input_nbytes = trt.volume(tensor.shape) * tensor.dtype.itemsize
            d_input = cp.cuda.alloc(input_nbytes)
            d_inputs.append(d_input)
            h_input = cp.asarray(np.ascontiguousarray(tensor.cpu().numpy().ravel()))
            cp.cuda.runtime.memcpyAsync(d_input.ptr, h_input.data.ptr, h_input.nbytes, cp.cuda.runtime.memcpyHostToDevice, stream.ptr)

    stream.synchronize()

    for output in outputs:
        h_output = cp.asarray(np.ascontiguousarray(ctr_prob.cpu().detach().numpy().ravel()))
        h_outputs.append(h_output)
        d_outputs.append(cp.cuda.alloc(h_output.nbytes))

    bindings = [int(d_input.ptr) for d_input in d_inputs] + [int(d_output.ptr) for d_output in d_outputs]

    for i in range(engine.num_io_tensors):
        if i < len(bindings):
            context.set_tensor_address(engine.get_tensor_name(i), bindings[i])

    context.execute_async_v3(stream_handle=stream.ptr)
    stream.synchronize()

    for idx, d_output in enumerate(d_outputs):
        cp.cuda.runtime.memcpyAsync(h_outputs[idx].data.ptr, d_output.ptr, h_outputs[idx].nbytes, cp.cuda.runtime.memcpyDeviceToHost, stream.ptr)
    stream.synchronize()
    h_dict = {}
    iter = 0
    name_dict=['ctr_prob', 'cxr_prob']
    for h_output in h_outputs:
        h_tensor = torch.tensor(h_output)
        h_dict[name_dict[iter]] = [h_tensor]
        iter += 1
        if (iter == 2):
            break

    max_diff = max(max_diff, diff_outputs(output_dict,h_dict))
    print(f"cur_diff:{diff_outputs(output_dict,h_dict)}")
    if(diff_outputs(output_dict,h_dict) > 1):
        raise Exception(f"diff too large:trt_result:{h_dict}, pytorch_result:{output_dict}")
print(f"max diff:{max_diff}")
