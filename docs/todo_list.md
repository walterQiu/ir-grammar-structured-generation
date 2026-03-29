Current Works
- 先換更強的 extraction model & 把 \" "\ 的部分拆掉試試看

- parsed_output 裡面有 None (看起來是 ir model 在搞)
     "parsed_output": {
        "event_type": "conflict.yield.n/a",
        "arguments": [
            {
                "role": "yielder",
                "text": "Japan"
            },
            {
                "role": "recipient",
                "text": "None"
            },
            {
                "role": "place",
                "text": "None"
            }
        ]
    },


- 看一下為什麼Qwen表現那麼爛 
    - extraction sentence 跟 ir sentence 的 sentence 都放到 prompt 最後面試試看
        - 目前看起來主要問題是 extraction model 語意理解有問題
    - 試試看 ICL (extraction 跟 ir 都要)
    - 真沒辦法優化，就拿更好的 LLM 作為 extraction model，再拿 Qwen 當作 IR model

- metrics 還有其他要加的嗎?
    - semantic 部分應該就這樣了
    - schema 要去看看 "部分 match" 這件事情有沒有意義 (目前想法是只看 exact match)



Future Works
- 資料集處理
    - 如果是 few shot 或是 zero shot，那可以把所有資料集都串起來
- prompt 優化 (所有prompt都需要)
    - extraction 的去看 G&O 論文
    - ir 的去看 code4struct 或另一篇還沒看的論文 或 G&O 的 O 部分
- metrics、model 等等的實驗 factor 添加
    - schema similarity 那篇
- build_baseline_event_extraction_prompt 的那種寫法好像沒辦法處理多層 schema ?
- EAE span candidate 欄位處理問題
    - 詳情請看"論文筆記"
        - search "傳統 EAE tasks 會給 model span candidate"
    - 簡單說就是傳統 IE model 會吃到 candidate span，再從候選中選 span；但我用的是 LM，所以 inference 時不需要看到這個候選，但這樣可能造成比較上的不公平?
- In context learning 嘗試
- 現在一次跑太多 sample 的話，記憶體會不會爆掉?

- checkpoint 不確定需不需要
- [warning] failed sample 79/100: nw_RC158e6f4713899ca8167a0567d44fa7682eae55c0f522addf3731c2df
    - ir-pipeline-gemini-2.5-flash-lite 跑到這邊會 timeout
    