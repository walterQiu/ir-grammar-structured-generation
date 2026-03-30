Current Works
- 換更強的 extraction model

- In context learning 嘗試

- 優化 IR model，盡量讓他的輸出乾淨點 (參考論文筆記中那堆奇怪的 IR model 輸出)

- extraction_prompt 裡面有 briefly explain，是不是不要讓 model 幹這件事比較好?

- config yaml 重新命名 & sample 數確認都是100



Future Works
- 資料集處理
    - 如果是 few shot 或是 zero shot，那可以把所有資料集都串起來

- prompt 優化 (所有prompt都需要)
    - extraction 的去看 G&O 論文
    - ir 的去看 code4struct 或另一篇還沒看的論文 或 G&O 的 O 部分

- build_baseline_event_extraction_prompt 的那種寫法好像沒辦法處理多層 schema ?

- 現在一次跑太多 sample 的話，記憶體會不會爆掉?

- checkpoint 不確定需不需要
- [warning] failed sample 79/100: nw_RC158e6f4713899ca8167a0567d44fa7682eae55c0f522addf3731c2df
    - ir-pipeline-gemini-2.5-flash-lite 跑到這邊會 timeout
    