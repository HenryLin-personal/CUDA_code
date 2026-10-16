mkdir -p reports
# fa0 每次 call_attention 发射 3 个 kernel (qk / softmax / pv), --launch-count 3 正好抓第一次调用
for p in fa0; do ncu --launch-count 3 --metrics gpu__time_duration.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed,l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_requests_pipe_lsu_mem_global_op_st.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum -o reports/$p ../build/$p; done
# for p in fa0; do ncu --launch-count 3 --set full -o reports/$p ./build/$p; done
