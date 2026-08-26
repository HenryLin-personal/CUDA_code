add_executable(prefix_sum1 kogge_stone.cu)
add_executable(prefix_sum2 kogge_stone_shared.cu)
add_executable(prefix_sum3 work_efficient_scan.cu)
add_executable(prefix_sum4 multi_block_scan.cu)

target_compile_options(prefix_sum1 PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-G -g>
)


target_compile_options(prefix_sum2 PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-G -g>
)

target_compile_options(prefix_sum3 PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-G -g>
)

target_compile_options(prefix_sum4 PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-G -g>
)