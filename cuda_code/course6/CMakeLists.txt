add_executable(race race.cu)
add_executable(no_race no_race.cu)
add_executable(hist histogram.cu)
add_executable(hist_shared histogram_shared.cu)

target_link_libraries(race PRIVATE ${CUDA_LIBRARIES})
target_link_libraries(no_race PRIVATE ${CUDA_LIBRARIES})
target_link_libraries(hist PRIVATE ${CUDA_LIBRARIES})
