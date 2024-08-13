import time
import os

import torch
from torch import nn
from transformers import AutoTokenizer, BertConfig, BertModel
import numpy as np

import onnxruntime as ort

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

MAX_SEQ_LENGTH = 64

class WmBERT(nn.Module):
    def __init__(self, path):
        super(WmBERT, self).__init__()
        config = BertConfig.from_pretrained(os.path.join(path, "config.json"))
        self.bert = BertModel(config)
        self.linear = nn.Linear(config.hidden_size, 3)
        self.dropout = nn.Dropout(0.1)


    def forward(self, input_ids, attention_mask, token_type_ids):
        output = self.bert(input_ids, attention_mask, token_type_ids)
        cls_output = output.pooler_output
        logits = self.linear(self.dropout(cls_output))
        probs = nn.functional.softmax(logits, dim=1)
        return probs

def encoding2ids(texts, tokenizer):
    input_text_list = []
    for text in texts:
        query, poi, spu = text
        text_a = query
        text_b = poi + tokenizer.sep_token + spu
        input_text_list.append((text_a[:MAX_SEQ_LENGTH // 4], text_b))
    encoding = tokenizer(input_text_list, padding='max_length', return_tensors="pt",
                                      max_length=MAX_SEQ_LENGTH, truncation=True)
    return encoding

def get_onnx_output(ort_session, input_ids, attention_mask, token_type_ids):
    ort_inputs = {
        'input_ids': input_ids.numpy().astype(np.int32),
        'attention_mask': attention_mask.numpy().astype(np.int32),
        'token_type_ids': token_type_ids.numpy().astype(np.int32)
    }
    ort_outs = ort_session.run(None, ort_inputs)
    return ort_outs[0]

def get_pytorch_output(model, input_ids, attention_mask, token_type_ids):
    with torch.no_grad():
        return model(input_ids.cuda(), attention_mask.cuda(), token_type_ids.cuda()).cpu().numpy()

def load_engine(engine_path):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())
    
def allocate_buffer(context):
    input_shape = (1, 64)
    input_nbytes = trt.volume(input_shape) * trt.int32.itemsize
    for name in ["input_ids", "token_type_ids", "attention_mask"]:
        context.set_input_shape(name, input_shape)
    assert len(context.infer_shapes()) == 0

    d_inputs = [cuda.mem_alloc(input_nbytes) for binding in range(3)]

    h_output = cuda.pagelocked_empty(tuple(context.get_tensor_shape("output")), dtype=np.float32)
    d_output = cuda.mem_alloc(h_output.nbytes)
    return d_inputs, d_output, h_output

def get_trt_output(engine, context, stream, d_inputs, d_output, h_output, input_ids, attention_mask, token_type_ids):
    # ref: https://github.com/NVIDIA/TensorRT/blob/release/8.6/demo/BERT/inference.py
    h_input_ids = cuda.register_host_memory(np.ascontiguousarray(input_ids.cpu().numpy().astype(np.int32).ravel()))
    h_token_type_ids = cuda.register_host_memory(np.ascontiguousarray(token_type_ids.cpu().numpy().astype(np.int32).ravel()))
    h_attention_mask = cuda.register_host_memory(np.ascontiguousarray(attention_mask.cpu().numpy().astype(np.int32).ravel()))
    
    cuda.memcpy_htod_async(d_inputs[0], h_input_ids, stream)
    cuda.memcpy_htod_async(d_inputs[1], h_attention_mask, stream)
    cuda.memcpy_htod_async(d_inputs[2], h_token_type_ids, stream)

    binding_idx_offset = 0
    bindings = [0 for _ in range(binding_idx_offset)] + [int(d_inp) for d_inp in d_inputs] + [int(d_output)]

    for i in range(engine.num_io_tensors): 
        context.set_tensor_address(engine.get_tensor_name(i), bindings[i + binding_idx_offset])
    
    context.execute_async_v3(stream_handle=stream.handle)
    stream.synchronize()
    cuda.memcpy_dtoh_async(h_output, d_output, stream)
    stream.synchronize()
    return h_output

def convert_int64_to_int32_pytorch(model):
    for name, param in model.named_parameters():
        if param.dtype == torch.int64:
            print(f"Converting parameter {name} from INT64 to INT32")
            param.data = param.data.to(torch.int32)

def export_onnx(onnx_export_path, model, encoding):
    convert_int64_to_int32_pytorch(model) # TRT不支持INT64，这里参数没有INT64的，但输入是INT64的，需要在下面进行转换

    torch.onnx.export(
        model, 
        (encoding['input_ids'].to(torch.int32), encoding['attention_mask'].to(torch.int32), encoding['token_type_ids'].to(torch.int32)), 
        onnx_export_path,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input_ids', 'attention_mask', 'token_type_ids'],
        output_names=['output'],
        dynamic_axes={'input_ids': {0: 'batch_size'},
                    'attention_mask': {0: 'batch_size'},
                    'token_type_ids': {0: 'batch_size'},
                    'output': {0: 'batch_size'}}
    )
    print(f"Export onnx to {onnx_export_path} successfully")

    # TODO: dynamo_export
    # onnx_program = torch.onnx.dynamo_export(model, **encoding)
    # onnx_program.save("bert.onnx")

if __name__ == "__main__":
    data_path = "/workdir/repos/bert_opt/benchmark-593.txt"
    model_path = "/workdir/repos/bert_opt/model"
    targets = []
    results = []
    texts = []
    cnt = 0
    MAX_CNT = 1
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            split_line = line.strip("\n").split("\t")
            texts.append(split_line[:3])
            cnt += 1
            if cnt == MAX_CNT:
                break
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    encoding = encoding2ids(texts, tokenizer)
    ckpts = torch.load(os.path.join(model_path, "pytorch_model.bin"), map_location="cpu")
    model = WmBERT(model_path)
    model.load_state_dict(ckpts, strict=False)
    # export_onnx("bert_ts_i32.onnx", model, encoding) 
    
    model.cuda().eval()
    start = time.time()
    pytorch_output = get_pytorch_output(model, **encoding)
    print(f"finished pytorch runing using: {time.time()-start} seconds")

    # ONNX runtime
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    providers = ['CUDAExecutionProvider']
    ort_session = ort.InferenceSession('bert_ts_i32.onnx', sess_options=options, providers=providers)
    start = time.time()
    onnx_output = get_onnx_output(ort_session, **encoding)
    print(f"finished onnx runing using: {time.time()-start} seconds")

    # TensorRT
    engine = load_engine('bert_fp16.trt')
    context = engine.create_execution_context()
    stream = cuda.Stream()
    context.set_optimization_profile_async(0, stream.handle)
    d_inputs, d_output, h_output = allocate_buffer(context)
    start = time.time()
    trt_output = get_trt_output(engine, context, stream, d_inputs, d_output, h_output, **encoding)
    print(f"finished trt runing using: {time.time()-start} seconds")

    # 计算差异
    difference_onnx = np.abs(pytorch_output - onnx_output)
    max_difference_onnx = np.max(difference_onnx)

    difference_trt = np.abs(pytorch_output - trt_output)
    max_difference_trt = np.max(difference_trt)

    tolerance = 1e-4

    print("PyTorch 输出:", pytorch_output)
    print("ONNX 输出:", onnx_output)
    print("TRT 输出:", trt_output)
    print("ONNX 最大单元素差异:", max_difference_onnx)
    print("TRT 最大单元素差异:", max_difference_trt)
    if np.allclose(pytorch_output, onnx_output, atol=tolerance):
        print("结果一致，导出的 ONNX 模型正确运行。")
    else:
        print("结果不一致，请检查导出的 ONNX 模型。")
    if np.allclose(pytorch_output, trt_output, atol=tolerance):
        print("结果一致，导出的 TRT 模型正确运行。")
    else:
        print("结果不一致，请检查导出的 TRT 模型。")