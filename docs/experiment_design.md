# 論文實驗設計總覽（IR Grammar Comparison）

## 總體研究目標
本研究旨在探討：
> 在固定語義抽取（Extraction）條件下，不同 IR grammar 如何影響 structured generation 的穩定性與表現。

核心策略：
- 採用 two-stage pipeline（semantic extraction → IR generation）
- 將「語義」與「結構生成」分離
- 針對 IR grammar 進行 controlled comparison

---

# Main Results

## M1：IR Grammar Comparison（核心實驗）

## 目的
在固定 two-stage pipeline 下，比較不同 IR grammar 對生成品質的影響，並找出表現最好的 IR (Best IR)。

## 設定
- Pipeline：Two-stage
- Extraction model：gemini-3.1-pro-preview（固定）
- IR model：Mistral-7B-Instruct-v0.3 (固定)
- IR 類型：JSON / Dot-notation / CODE4STRUCT

## 補充分析
抽樣 30–50 筆進行 error analysis：
- 上游 notes 是否錯誤
- IR generation 是否正確反映 notes

---

# Ablation Studies

## A1：Pipeline Comparison

### 目的
驗證 two-stage pipeline 是否優於 one-stage 方法。

### 比較
1. One-stage JSON VS. Two-stage JSON  
2. One-stage best IR VS. Two-stage best IR  

### 設定
- 都用 Mistral-7B-Instruct-v0.3

## A2：JSON VS. Best IR

### 目的
檢驗在固定 pipeline 下，Best IR 是否優於 JSON 表示法。

### 比較
1. One-stage JSON VS. One-stage best IR  
2. Two-stage JSON VS. Two-stage best IR

### 設定
- 都用 Mistral-7B-Instruct-v0.3

## A3：ICL(In-Context Learning) VS. Non-ICL

### 設定
- Pipeline: two-stage
- IR：JSON / Dot-notation / CODE4STRUCT
- 模型：Gemini 、 Mistral
- 0-shot vs 4-shot

### ICL examples 包含以下三種情境
1. 每個 role 對應單一 span
2. 部分 role 對應多個 span
3. 完全沒有合法 role

---

# Practical Evaluation

## P1：Practical System Comparison

## 目的
驗證 two-stage IR pipeline 在實務上是否具競爭力

## 比較
1. One-stage JSON
    - Gemini 3.1 Pro  
    - gpt-5.3-chat (未啟用)
    - Claude Sonnet 4.6 (未啟用)
2. Two-stage JSON — Gemini + Mistral  
3. Two-stage best IR — Gemini + Mistral  
4. Two-stage best IR — Mistral + Mistral (可選，但效果應該會很差)

---

# 總結
1. Main Results：IR Grammar Comparison
2. Ablation Studies：Pipeline / JSON vs Best IR / ICL 分析
3. Practical Evaluation：實務系統比較

---

# Future Session：未來實驗規劃（目前不納入主實驗）

> 本區塊屬於主實驗全部完成後，才評估是否追加執行的延伸實驗。

## F1：IR model 擴充

目前主實驗中的地端 model 以 Mistral 為主。  
未來可加入 Qwen，觀察在相同設定下的差異。

## F2：M1 補充分析擴充（錯誤類型表）

在 M1 的抽樣分析中，除了檢查 extraction notes 品質外，
再定義一個簡單的 error taxonomy，觀察不同錯誤類型對 IR model 的影響。

### 建議錯誤類型（初版）
- Upstream note omission
- Wrong span copied from notes
- Hallucinated role
- Multiplicity violation
- Correct IR but wrong semantics already present in notes

### 目的
- 分析不同上游 extration model 的錯誤類型對 IR model 會有什麼影響

## F3：M1 將 IR model 換成 Gemini 3.1 Pro

### 目的
- 分析對於能力很強的 LLM 來說，不同 IR 的表現是否對齊地端 model 的表現

## F4：A1、A2 將所有 model 換成 Gemini 3.1 Pro

### 目的
- 分析對於能力很強的 LLM 來說，不同 pipeline、IR 是否會影響到模型能力
