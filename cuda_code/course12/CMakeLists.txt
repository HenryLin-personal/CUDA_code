add_executable(conv im2col.cu)
add_executable(base_conv base_conv.cu)

target_link_libraries(conv PRIVATE ${CUDA_LIBRARIES} cublas)
target_link_libraries(base_conv PRIVATE ${CUDA_LIBRARIES} cublas)

target_compile_options(base_conv PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-G -g>
)