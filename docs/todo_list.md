Current Works
- find span position function 可以拿掉? (可以)
    - _find_span_by_text()
    - 教授如果覺得我的 RAMS dataset 使用方式是OK的就行
    - predictions 輸出的 span position encode 部分也都可以拿掉
    - 跟教授討論完後記得更新 RAMS dataset 的說明文件
- validator 跟 metrics 的功能可能需要 decouple (主要是最開始AI寫的那3個metrics)
- 加 time.sleep()、retry mechanism
    - 或是 checkpoint


Future Works
- 資料集處理
    - 如果是 few shot 或是 zero shot，那可以把所有資料集都串起來
- prompt 優化 (所有prompt都需要)
    - extraction 的去看 G&O 論文
    - ir 的去看 code4struct 或另一篇還沒看的論文 或 G&O 的 O 部分
- metrics、model 等等的實驗 factor 添加
- json_parser.py 中的 _extract_braced_object() 是不是作弊?
- build_baseline_event_extraction_prompt 的那種寫法好像沒辦法處理多層 schema ?
- 另一個 EAE dataset 嘗試
- EAE span candidate 欄位處理問題
    - 詳情請看"論文筆記"
- In context learning 嘗試
- 地端模型嘗試
- Validator、最早三個指標拿掉?
    - 或許某些可以留，例如 schema 相似度之類的 (但這可能需要重新實作，實作方式要向其他 metrics 那樣)
    - json 解析成功與否不用評估於有 ir compiler 的 pipeline，畢竟它的結構是用 deterministic 的方式生出來的，但還是可以留著測測看 baseline
    - 甚至 evaluations/ 裡面只需要有 metrics 相關的功能，然後把不同 metrics 拆成不同的 .py 之類的?

- metrics 的命名要改一下
    - {"rows": 3, "metrics": {"json_valid_rate": 1.0, "schema_valid_rate": 1.0, "exact_match_rate": 0.3333333333333333, "arg_i_precision": 0.5, "arg_i_recall": 0.3333333333333333, "arg_i_f1": 0.4, "arg_c_precision": 0.5, "arg_c_recall": 0.3333333333333333, "arg_c_f1": 0.4, "content_soft_precision": 0.5809850096702576, "content_soft_recall": 0.4283756564060847, "content_similarity": 0.49314351795984923, "bemeae_soft_precision": 0.6352649827798208, "bemeae_soft_recall": 0.4842991679906845, "bemeae": 0.549603704981349}}