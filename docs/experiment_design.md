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
- IR model：
    - gemini-3.1-pro-preview
    - gemini-2.5-flash
    - Mistral-7B-Instruct-v0.3
- IR 類型：JSON / Incremental-Assignment / CODE4STRUCT

---

# Ablation Studies

## A1：Pipeline Comparison

### 目的
驗證 two-stage pipeline 是否優於 one-stage 方法。

### 比較
1. One-stage JSON VS. Two-stage JSON  
2. One-stage best IR VS. Two-stage best IR  

### 設定
- 模型：Gemini 3.1 Pro、Gemini 2.5 flash、Mistral (Two-stage的 extraction/ir model 需要相同)

## A2：JSON VS. Best IR

### 目的
檢驗在固定 pipeline 下，Best IR 是否優於 JSON 表示法。

### 比較
1. One-stage JSON VS. One-stage best IR  
2. Two-stage JSON VS. Two-stage best IR

### 設定
- 模型：Gemini 3.1 Pro、Gemini 2.5 flash、Mistral

## A3：ICL(In-Context Learning) VS. Non-ICL

### 設定
- Pipeline: two-stage
- IR：JSON / Incremental-Assignment / CODE4STRUCT
- 模型：Gemini 3.1 Pro、Gemini 2.5 flash、Mistral
- 0-shot vs 4-shot

### ICL examples 包含以下四種情境
1. 每個 role 對應單一 span
2. 部分 role 對應多個 span，但實際都只有一個 span
3. 部分 role 對應多個 span，且實際上也有多個 spans
4. 完全沒有合法 role
