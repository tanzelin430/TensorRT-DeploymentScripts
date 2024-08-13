import tensorrt as trt
import onnx
TRT_LOGGER = trt.Logger(trt.Logger.INFO)

explicit_batch_flag = 1
dynamic_name = []

with trt.Builder(TRT_LOGGER) as builder, builder.create_network(explicit_batch_flag) as network, builder.create_builder_config() as builder_config, trt.OnnxParser(network, TRT_LOGGER) as parser, trt.Runtime(TRT_LOGGER) as runtime:
    # pass
    builder_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 35)
    builder_config.set_flag(trt.BuilderFlag.FP16)
    builder_config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    parser.parse_from_file('.onnx')
    model = onnx.load('.onnx')
    # for i in range(network.num_inputs):
    #    print(network.get_input(i).shape)
    profile = builder.create_optimization_profile()
    print(f"network:{network.num_inputs}")
    for i in range(network.num_inputs):
        binding = network.get_input(i).name
        if binding in dynamic_name:
            shape = network.get_input(i).shape
            shape = trt.Dims(shape)
            # dynamic shape
            min_shape = [1]
            opt_shape = [100]
            max_shape = [200]
        else:
            min_shape = shape
            opt_shape = shape
            max_shape = shape
        try:
            profile.set_shape(binding, min_shape, opt_shape, max_shape)
        except Exception as e:
            print(f"Exception during profile set_shape for binding {binding}: {e}")

    # # profile.set_shape('target_poi_seq_length',(16,),(16,),(16,))
    builder_config.add_optimization_profile(profile)
    # print(profile)
    try:
        # print(dir(builder))
        # engine = builder.build_serialized_network(network, builder_config)
        # print(builder.is_network_supported(network, builder_config))
        if (builder.is_network_supported(network, builder_config)):
            print("Network is supported")
            engine = builder.build_serialized_network(network, builder_config)
        if engine is None:
            print("Failed to build the engine!")
        else:
            print("Engine successfully built.")
            with open('grranking_bs1_dynamicshape.trt', "wb") as f:
                f.write(engine)
                print(f"Engine serialized and saved as grranking.trt")
    except Exception as e:
        print(f"Exception during engine build: {e}")




