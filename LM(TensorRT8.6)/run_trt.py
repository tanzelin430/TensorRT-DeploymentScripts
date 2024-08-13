import time

from transformers import AutoTokenizer
import numpy as np

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

TRT_LOGGER = trt.Logger(trt.Logger.INFO)

MAX_SEQ_LENGTH = 64

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

def get_input():
    data_path = "/workdir/repos/bert_opt/benchmark-593.txt"
    model_path = "/workdir/repos/bert_opt/model"
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
    return encoding

if __name__ == "__main__":
    encoding = get_input()
    input_ids = encoding['input_ids']
    token_type_ids = encoding['token_type_ids']
    attention_mask = encoding['attention_mask']

    with open('bert_py.trt', 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime, \
        runtime.deserialize_cuda_engine(f.read()) as engine, engine.create_execution_context() as context:

        selected_profile = -1
        for idx in range(engine.num_optimization_profiles):
            profile_shape = engine.get_tensor_profile_shape(name = "input_ids", profile_index = idx)
            if profile_shape[0][0] <= 1 and profile_shape[2][0] >= 1 and profile_shape[0][1] <= 64 and profile_shape[2][1] >= 64:
                selected_profile = idx
                break
        if selected_profile == -1:
            raise RuntimeError("Could not find any profile that can run batch size {}.".format(1))
        
        stream = cuda.Stream()
        context.set_optimization_profile_async(selected_profile, stream.handle)
        binding_idx_offset = selected_profile * engine.num_io_tensors

        input_shape = (1, 64)
        input_nbytes = trt.volume(input_shape) * trt.int32.itemsize
        for name in ["input_ids", "token_type_ids", "attention_mask"]:
            context.set_input_shape(name, input_shape)
        assert len(context.infer_shapes()) == 0

        d_inputs = [cuda.mem_alloc(input_nbytes) for binding in range(3)]

        h_output = cuda.pagelocked_empty(tuple(context.get_tensor_shape("output")), dtype=np.float32)
        d_output = cuda.mem_alloc(h_output.nbytes)

        # h_input_ids = cuda.pagelocked_empty(trt.volume(context.get_tensor_shape("input_ids")), dtype=np.float32)
        # h_attention_mask = cuda.pagelocked_empty(trt.volume(context.get_tensor_shape("attention_mask")), dtype=np.float32)
        # h_token_type_ids = cuda.pagelocked_empty(trt.volume(context.get_tensor_shape("token_type_ids")), dtype=np.float32)
        # np.copyto(h_input_ids, input_ids.cpu().numpy().astype(np.float32).ravel())
        # np.copyto(h_attention_mask, attention_mask.cpu().numpy().astype(np.float32).ravel())
        # np.copyto(h_token_type_ids, token_type_ids.cpu().numpy().astype(np.float32).ravel())

        h_input_ids = cuda.register_host_memory(np.ascontiguousarray(input_ids.cpu().numpy().astype(np.int32).ravel()))
        h_token_type_ids = cuda.register_host_memory(np.ascontiguousarray(token_type_ids.cpu().numpy().astype(np.int32).ravel()))
        h_attention_mask = cuda.register_host_memory(np.ascontiguousarray(attention_mask.cpu().numpy().astype(np.int32).ravel()))

        print(h_input_ids)

        eval_start_time = time.time()
        cuda.memcpy_htod_async(d_inputs[0], h_input_ids, stream)
        cuda.memcpy_htod_async(d_inputs[1], h_token_type_ids, stream)
        cuda.memcpy_htod_async(d_inputs[2], h_attention_mask, stream)

        bindings = [0 for _ in range(binding_idx_offset)] + [int(d_inp) for d_inp in d_inputs] + [int(d_output)]

        for i in range(engine.num_io_tensors): 
            context.set_tensor_address(engine.get_tensor_name(i), bindings[i + binding_idx_offset])

        context.execute_async_v3(stream_handle=stream.handle)

        # Synchronize the stream
        stream.synchronize()
        eval_time_elapsed = (time.time() - eval_start_time)
        print(eval_time_elapsed)

        cuda.memcpy_dtoh_async(h_output, d_output, stream)
        stream.synchronize()

        print(h_output)