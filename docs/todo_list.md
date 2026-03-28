Current Works


Future Works
- 資料集處理
    - 如果是 few shot 或是 zero shot，那可以把所有資料集都串起來
- prompt 優化 (所有prompt都需要)
    - extraction 的去看 G&O 論文
    - ir 的去看 code4struct 或另一篇還沒看的論文 或 G&O 的 O 部分
- metrics、model 等等的實驗 factor 添加
- build_baseline_event_extraction_prompt 的那種寫法好像沒辦法處理多層 schema ?
- 另一個 EAE dataset 嘗試
- EAE span candidate 欄位處理問題
    - 詳情請看"論文筆記"
        - search "傳統 EAE tasks 會給 model span candidate"
    - 簡單說就是傳統 IE model 會吃到 candidate span，再從候選中選 span；但我用的是 LM，所以 inference 時不需要看到這個候選，但這樣可能造成比較上的不公平?
- In context learning 嘗試
- 地端模型嘗試

- checkpoint 不確定需不需要
