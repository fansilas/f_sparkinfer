# CPU composition test — pure C++, validates the decode-layer wiring math.
add_executable(decode_layer_cpu_test decode_layer_cpu_test.cpp)
set_target_properties(decode_layer_cpu_test PROPERTIES CXX_STANDARD 17)
add_test(NAME decode_layer_cpu_test COMMAND decode_layer_cpu_test)

# GPU integration test — runs the real DecodeRunner; skips without a device.
add_executable(decode_runner_gpu_test decode_runner_gpu_test.cpp)
target_link_libraries(decode_runner_gpu_test PRIVATE sparkinfer_runtime CUDA::cudart)
add_test(NAME decode_runner_gpu_test COMMAND decode_runner_gpu_test)
