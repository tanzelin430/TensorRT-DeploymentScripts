import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.INFO)

explicit_batch_flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

with trt.Builder(TRT_LOGGER) as builder, builder.create_network(explicit_batch_flag) as network, builder.create_builder_config() as builder_config, trt.OnnxParser(network, TRT_LOGGER) as parser, trt.Runtime(TRT_LOGGER) as runtime:
    builder_config.max_workspace_size = 1 << 30
    builder_config.set_flag(trt.BuilderFlag.FP16)
    builder_config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED

    with open('.onnx', 'rb') as model:
        print('Beginning ONNX file parsing')
        if not parser.parse(model.read()):
            print ('ERROR: Failed to parse the ONNX file.')
            for error in range(parser.num_errors):
                print (parser.get_error(error))

    profile = builder.create_optimization_profile()
    for i in range(network.num_inputs):
        binding = network.get_input(i).name
        shape = network.get_input(i).shape
        if shape[0] == -1:
            profile.set_shape(binding, [1, 64], [1, 64], [1, 64])

    builder_config.add_optimization_profile(profile)

    try:
        engine = builder.build_engine(network, builder_config)
        if engine is None:
            print("Failed to build the engine!")
        print("Engine successfully built.")
        with open('bert_fp16.trt', "wb") as f:
            f.write(engine.serialize())
            print(f"Engine serialized and saved as bert_py.trt")
    except Exception as e:
        print(f"Exception during engine build: {e}")
