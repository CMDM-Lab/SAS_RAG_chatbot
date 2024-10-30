# SAS_RAG_2024

Retrieval-Augmented Generation (RAG) for The SAS System in 2024.

![RAG](./.gitignore/SAS_RAG.png)
## 環境設置

### 0. 安裝 Ollama

請先到 Ollama 官網下載並安裝對應系統的 Ollama 軟體：[Ollama 官網](https://ollama.com)

安裝完成後，啟動 Ollama，並確認模型可以在本地端運行：

```bash
ollama run llama3
```

### 1. 創建 Conda 環境

```bash
conda create -n SAS_env python=3.11
```

### 2. 進入 Conda 環境

```bash
conda activate SAS_env
```

### 3. 安裝相關套件

```bash
pip install -r requirements.txt
```

## 執行 RAG

### 啟動指令

```bash
streamlit run chatbot.py #chemical_mapping No. --server.port #xxxx
```

### 範例

```bash
streamlit run chatbot.py 59 --server.port 2024
```

#### 🟡新增CHATGPT版本 使用前須在 retriever_chain_openai.py 輸入API KEY

```bash
streamlit run chatbot_openai.py 59 --server.port 2024
```

## 數據處理

### 1. 下載化合物資訊並轉檔

將 SAS 平台上化合物資訊下載後，放入 `./SAS_file` 資料夾，並轉檔為 txt 格式：

```bash
python load_csv_to_txt.py
```

### 2. 轉換化合物 txt 檔為向量資料庫

更改 `vectorstore.py` 中的 `file_path` 路徑，並執行以下指令：

```bash
python vectorstore.py
```

## 設定與調整

- **更改 RAG 人物設定的 Prompt 內容**：調整 `retriever_chain.py`。
- **更改 Nemoguardrails 限制條件**：編輯 `./config/config.yml`。

## 化合物資料

化合物向量資料庫與 SAS 平台流水號的對應文件位於 `chemical_mapping.txt`。

## 後續維護：新增化合物

1. 將新化合物的 CSV 檔放入 `./SAS_file` 資料夾。
2. 使用 `load_csv_to_txt.py` 進行轉檔。
3. 利用 `vectorstore.py` 轉換為向量資料庫。
   - 由於每份文件差異，需嘗試不同的文字分割塊尺寸與重疊數，建議初始設定：`chunk_size=1000`，`chunk_overlap=200`，`SAS chemical number=對應流水號`。
4. 將對應流水號新增至 `chemical_mapping.txt`。
5. 執行 `chatbot.py` 啟動 RAG。
