import streamlit as st
import pandas as pd
import numpy as np
import requests
from io import StringIO
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# -------------------------------
# 缓存模型和计算结果，提高重复运行效率
# -------------------------------
@st.cache_resource(show_spinner=False)
def load_dense_model():
    # 加载 SentenceTransformer 模型（这里使用 all-MiniLM-L6-v2）
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model

@st.cache_resource(show_spinner=False)
def compute_dense_embeddings(model, texts):
    # 计算稠密向量
    return model.encode(texts, convert_to_numpy=True)

@st.cache_resource(show_spinner=False)
def compute_tfidf_matrix(texts):
    # 使用 TfidfVectorizer 计算文档的 TF-IDF 矩阵
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)
    return vectorizer, tfidf_matrix

# ------------------------------------------------------------------------------
# 腾讯云混合检索 API 封装函数
# ------------------------------------------------------------------------------
def tencent_cloud_hybrid_search(query, docs, dense_weight, sparse_weight, vdb_url, vdb_key):
    """
    调用腾讯云混合检索接口：
      - query: 单个查询字符串
      - docs: 待检索的内容片段列表
      - dense_weight, sparse_weight: 权重设置
      - vdb_url, vdb_key: 腾讯云服务的连接参数
    假设接口请求发送到 {vdb_url}/hybrid_search，并返回以下 JSON 格式：
      {
          "dense_scores": [score1, score2, ...],
          "sparse_scores": [score1, score2, ...],
          "hybrid_scores": [score1, score2, ...]
      }
    """
    payload = {
        "query": query,
        "documents": docs,
        "dense_weight": dense_weight,
        "sparse_weight": sparse_weight
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {vdb_key}"
    }
    try:
        response = requests.post(f"{vdb_url}/hybrid_search", json=payload, headers=headers, timeout=10)
        response.raise_for_status()  # 若返回状态异常则抛出异常
    except Exception as e:
        st.error(f"请求腾讯云混合检索服务失败: {e}")
        return None

    try:
        data = response.json()
    except Exception as e:
        st.error(f"解析返回数据失败: {e}")
        return None
    return data

# ------------------------------------------------------------------------------
# Streamlit 主程序
# ------------------------------------------------------------------------------
st.title("腾讯云混合检索 Demo")
st.write("本 Demo 演示如何直接调用腾讯云向量数据库服务进行混合检索。请配置腾讯云服务参数后，导入或输入查询与内容片段，系统将调用腾讯云接口返回检索得分。")

# ------------------------------------------------------------------------------
# 侧边栏：腾讯云混合检索配置
# ------------------------------------------------------------------------------
st.sidebar.header("腾讯云混合检索配置")
# 你可以选择将配置写入 Streamlit secrets 中，本示例允许直接在页面上填入
vdb_url = st.sidebar.text_input("VDB URL", "http://lb-o1jzbc6h-rmkf8txyz0c17dq5.clb.ap-shanghai.tencentclb.com:10000")
vdb_key = st.sidebar.text_input("VDB KEY", "R4fThwO4VJhgp9m70YvKDprZ8IrIuzBi1b43Gb9i", type="password")

st.markdown("---")
# ------------------------------------------------------------------------------
# 数据输入方式选择
# ------------------------------------------------------------------------------
input_mode = st.radio("请选择输入方式：", ("手动输入", "上传 CSV 文件"))

if input_mode == "手动输入":
    st.subheader("手动输入")
    queries_text = st.text_area("请输入查询（每行一个）：", height=150)
    docs_text = st.text_area("请输入内容片段（每行一个）：", height=150)
    queries = [line.strip() for line in queries_text.splitlines() if line.strip()]
    docs = [line.strip() for line in docs_text.splitlines() if line.strip()]
else:
    st.subheader("上传 CSV 文件")
    col1, col2 = st.columns(2)
    with col1:
        query_file = st.file_uploader("上传查询 CSV 文件（必须包含 'query' 列）", type=["csv"])
    with col2:
        doc_file = st.file_uploader("上传内容 CSV 文件（必须包含 'segment' 列）", type=["csv"])
    queries = []
    docs = []
    if query_file is not None:
        try:
            queries_df = pd.read_csv(query_file)
            if 'query' not in queries_df.columns:
                st.error("查询文件中必须包含 'query' 列")
            else:
                queries = queries_df['query'].dropna().astype(str).tolist()
        except Exception as e:
            st.error(f"读取查询文件出错: {e}")
    if doc_file is not None:
        try:
            docs_df = pd.read_csv(doc_file)
            if 'segment' not in docs_df.columns:
                st.error("内容文件中必须包含 'segment' 列")
            else:
                docs = docs_df['segment'].dropna().astype(str).tolist()
        except Exception as e:
            st.error(f"读取内容文件出错: {e}")

st.markdown("---")
# ------------------------------------------------------------------------------
# 设置混合检索的权重参数
# ------------------------------------------------------------------------------
st.subheader("设置检索权重")
dense_weight = st.slider("稠密向量权重（语义检索）", 0.0, 1.0, 0.9, 0.05)
sparse_weight = st.slider("稀疏向量权重（关键词检索）", 0.0, 1.0, 0.1, 0.05)

# ------------------------------------------------------------------------------
# 批量执行混合检索逻辑
# ------------------------------------------------------------------------------
if st.button("运行混合检索"):
    if not vdb_url or "YOUR CONNECTION URL" in vdb_url:
        st.error("请在侧边栏配置正确的 VDB URL！")
    elif not vdb_key or "YOUR CONNECTION KEY" in vdb_key:
        st.error("请在侧边栏配置正确的 VDB KEY！")
    elif not queries:
        st.error("请至少提供一个查询！")
    elif not docs:
        st.error("请至少提供一个内容片段！")
    else:
        st.info("正在调用腾讯云混合检索服务……")
        # 最终存放所有结果记录
        result_rows = []
        # 遍历每个查询，调用腾讯云混合检索接口
        for query in queries:
            search_result = tencent_cloud_hybrid_search(query, docs, dense_weight, sparse_weight, vdb_url, vdb_key)
            if search_result is None:
                st.error(f"查询 '{query}' 检索失败，跳过该查询。")
                continue
            # 期待返回结果中包含 'dense_scores', 'sparse_scores', 'hybrid_scores'
            dense_scores = search_result.get("dense_scores", [None]*len(docs))
            sparse_scores = search_result.get("sparse_scores", [None]*len(docs))
            hybrid_scores = search_result.get("hybrid_scores", [None]*len(docs))
            for i, doc in enumerate(docs):
                result_rows.append({
                    "query": query,
                    "segment": doc,
                    "dense_score": dense_scores[i] if i < len(dense_scores) else None,
                    "sparse_score": sparse_scores[i] if i < len(sparse_scores) else None,
                    "hybrid_score": hybrid_scores[i] if i < len(hybrid_scores) else None,
                })
        if not result_rows:
            st.error("未获取到任何结果。")
        else:
            result_df = pd.DataFrame(result_rows)
            st.success("混合检索完成！")
            st.subheader("检索结果")
            st.dataframe(result_df)
            
            # 提供下载 CSV 结果的按钮
            csv = result_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="导出结果为 CSV",
                data=csv,
                file_name='tencent_hybrid_search_results.csv',
                mime='text/csv'
            ) 
