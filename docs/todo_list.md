Current Works

- vllm optimize

- 固定 extraction model 去做 ablation study


Future Works

- prompt 優化 (所有prompt都需要)
    - build_direct_ir_prompt 內容對齊 build_ir_generation_prompt

- 現在一次跑太多 sample 的話，記憶體會不會爆掉?

- json ICL?

- checkpoint 不確定需不需要
- [warning] failed sample 79/100: nw_RC158e6f4713899ca8167a0567d44fa7682eae55c0f522addf3731c2df
    - ir-pipeline-gemini-2.5-flash-lite 跑到這邊會 timeout
    