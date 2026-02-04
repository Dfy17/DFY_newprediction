import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib  # 核心库：用于加载之前保存的模型文件 (.pkl)
from lime.lime_tabular import LimeTabularExplainer

# ==========================================
# 1. 页面配置与美化 (Page Config & Styling)
# ==========================================
st.set_page_config(
    page_title="DFY Prediction",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 注入 CSS 样式，保持专业的医学软件外观
st.markdown("""
<style>
    .main {
        background-color: #FAFAFA;
    }
    h1 {
        color: #2C3E50;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-weight: 600;
    }
    h3 {
        color: #34495E;
        border-bottom: 2px solid #EAEAEA;
        padding-bottom: 10px;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #4b6cb7 0%, #182848 100%);
        color: white;
        border: none;
        border-radius: 8px;
        height: 55px;
        font-size: 18px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    div[data-testid="stMetricValue"] {
        font-size: 26px;
        color: #2C3E50;
    }
</style>
""", unsafe_allow_html=True)

CATEGORY_FEATURE_DESC = {
    "Cholecystectomy": "Cholecystectomy (胆囊切除史)"
}
# ==========================================
# 2. 加载已保存的模型 (Model Loading)
# ==========================================
#@st.cache_resource
def load_saved_model():
    """
    从本地的 .pkl 文件加载所有模型组件。
    注意：服务器上必须有 '2.训练集构建模型/ann_model.pkl' 这个文件。
    """
    try:
        # 使用 joblib 加载模型字典
        model_package = joblib.load("ann_model_calculator.pkl")
        return model_package
    except FileNotFoundError:
        return None


# 执行加载
loaded_data = load_saved_model()

# 错误处理：如果没找到文件，停止运行并提示
if loaded_data is None:
    st.error("⚠️ Critical Error: Model file 'gist_model_v1.pkl' not found.")
    st.info("Please ensure you have run 'save_models.py' locally and uploaded the resulting .pkl file to the server.")
    st.stop()

# 解包模型组件 (从字典里取出)
#svm_model = loaded_data["svm"]
ann_model = loaded_data["ann"]
#meta_model = loaded_data["meta"]
scaler = loaded_data["scaler"]
feature_names = loaded_data["feature_names"]
X_train_data = loaded_data["X_train_data"]  # 用于 LIME 解释的背景数据




#scaler验证
# 打印关键信息，验证是否拟合
#print("✅ scaler 类型：", type(scaler))
#print("✅ scaler 是否有 mean_ 属性：", hasattr(scaler, "mean_"))
#print("✅ scaler 是否有 scale_ 属性：", hasattr(scaler, "scale_"))
#if hasattr(scaler, "mean_"):
    #print("✅ scaler.mean_（拟合后的均值）：", scaler.mean_)
#if hasattr(scaler, "scale_"):
    #print("✅ scaler.scale_（拟合后的标准差）：", scaler.scale_)

# 模拟一次 transform 操作，验证是否能正常运行
#test_input = np.array([[0, 25, 10, 5, 90, 1.2, 80]])  # 符合 7 个特征的测试数据
#test_input_df = pd.DataFrame(test_input, columns=feature_names)
#try:
    #test_scaled = scaler.transform(test_input_df)
    #print("✅ 模拟 transform 成功，缩放后结果：", test_scaled)
#except Exception as e:
    #print("❌ 模拟 transform 失败：", e)


#
# 验证 scaler 状态
#if hasattr(scaler, "mean_") and hasattr(scaler, "var_"):
    #print("✅ 验证通过：scaler 已拟合")
    #print(f"scaler 均值：{scaler.mean_}")
    #print(f"scaler 方差：{scaler.var_}")
#else:
    #print("❌ 验证失败：scaler 未拟合")
    # 此时必须重新运行训练代码，重新保存模型包
    # 重新训练的核心步骤：确保 train_data_scaler 有数据 → scaler_ann.fit(train_data_scaler) → 保存 model_package






# ==========================================
# 3. 定义预测管道 (Prediction Pipeline)
# ==========================================
def custom_pipeline_proba(X_input_df):
    """
    复现 Model 17 的预测逻辑：
    SVM(raw) + ANN(scaled) -> Meta Model
    """
    # 确保列名一致
    X_input_df.columns = feature_names

    # 1. SVM 预测 (使用原始数据)
    #prob_svm = svm_model.predict_proba(X_input_df)[:, 1]

    # 2. ANN 预测 (使用标准化数据)
    # 使用之前保存的 scaler 进行转换，不要重新 fit
    X_scaled = scaler.transform(X_input_df)
    #prob_ann = ann_model.predict_proba(X_scaled)[:, 1]
    # ANN 模型最终预测（返回类别概率分布）
    return ann_model.predict_proba(X_scaled)

    # 3. 堆叠 (Stacking)
    #stacked_features = np.column_stack((prob_svm, prob_ann))

    # 4. 元模型最终预测
    #return meta_model.predict_proba(stacked_features)


# 初始化 LIME 解释器
# 使用加载进来的 X_train_data 作为参考背景
lime_explainer = LimeTabularExplainer(
    training_data=X_train_data.values,
    feature_names=feature_names,
    class_names=["No Relapse", "Relapse"],
    mode="classification"
)

# ==========================================
# 4. 用户界面布局 (UI Layout)
# ==========================================

st.title("🔬 DFY  Predictive Model ")
st.markdown("""
This tool predicts the risk of relapse based on preoperative imaging and serological markers.
""")
st.caption("Powered by  ANN")
st.markdown("---")

with st.container():
    col1, col2 = st.columns([1, 1], gap="large")

    # --- 左侧特征 ---
    with col1:
        st.markdown("### 🖼️ col1")

        # 1. Cholecystectomy（分类特征：0=无，1=有）
        cholecystectomy = st.radio(
            CATEGORY_FEATURE_DESC["Cholecystectomy"],
            options=[0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes",
            horizontal=True
        )
        # 2. BMI
        # 身高输入
        height_cm = st.number_input(
            "Height (身高) [cm]",
            min_value=80.0, max_value=250.0, value=165.0, step=0.5,
            help="Normal adult range: 140-200 cm"
        )
        # 体重输入
        weight_kg = st.number_input(
            "Weight (体重) [kg]",
            min_value=30.0, max_value=200.0, value=60.0, step=0.5,
            help="Normal adult range: 40-150 kg"
        )
        # 实时自动计算BMI并展示
        height_m = height_cm / 100  # 转换为米
        bmi_calc = round(weight_kg / (height_m ** 2), 1)  # 保留1位小数
        st.success(f"✅ Auto-calculated BMI: **{bmi_calc}**")
        st.markdown('<p class="bmi-hint">BMI formula: weight(kg) / height(m)²</p>', unsafe_allow_html=True)


        # 3. CBD Diameter（胆总管直径）
        cbd_dia = st.number_input(
            "CBD Diameter [cm]",
            min_value=0.0, max_value=2.0, value=0.5, step=0.1,
            help="Common Bile Duct Diameter"
        )
        
    # --- 右侧：血清学指标 ---
    with col2:
        st.markdown("### 🩸 col2")
        #st.info("Continuous variables. Please enter the raw values from blood test.")

        # 4.Maximum CBDS Diameter（最大胆总管结石直径）
        max_cbds_dia = st.number_input(
            "Maximum CBDS Diameter [cm]",
            min_value=0.0, max_value=4.0, value=1.0, step=0.1,
            help="Maximum Common Bile Duct Stone Diameter"
        )

        # 5.CBD Angulation（胆总管成角）
        cbd_ang = st.number_input(
            "CBD Angulation [°]",
            min_value=80.0, max_value=180.0, value=90.0, step=0.1,
            help="Common Bile Duct Angulation (0-180°)"
        )
        # 6. QRLDKL（影像特征指标）
        qrldkl = st.number_input(
            "QRLDKL[boxes]",
            min_value=0.0, max_value=18.0, value=0.0, step=1.0,
            help="Imaging feature index"
        )
        #7.ALP（碱性磷酸酶，血清学指标）
        alp = st.number_input(
            "ALP [U/L]",
            min_value=40.0, max_value=580.0, value=100.0, step=1.0,
            help="Alkaline Phosphatase (normal: 40-150 U/L)"
        )

st.markdown("<br>", unsafe_allow_html=True)

# ==========================================
# 5. 预测逻辑与结果展示 (Prediction Logic)
# ==========================================
if st.button("CALCULATE RISK SCORE"):

    # 构建输入数据
    input_data = [
        cholecystectomy, bmi_calc, cbd_dia, max_cbds_dia,
        cbd_ang, qrldkl, alp
    ]
    input_df = pd.DataFrame([input_data], columns=feature_names)

    # 调用管道进行预测
    final_proba_dist = custom_pipeline_proba(input_df)[0]
    risk_score = final_proba_dist[1]
    risk_percentage = risk_score * 100

    # --- 结果展示区 ---
    st.markdown("---")
    st.markdown("### 📊 Prediction Results")

    r_col1, r_col2 = st.columns([1, 2])

    # 判定风险等级
    is_high_risk = risk_score > 0.5

    with r_col1:
        if is_high_risk:
            st.error("**RELAPSE RISK**")
            st.caption("Predicted as Relapse Category")
        else:
            st.success("**NO RELAPSE RISK**")
            st.caption("Predicted as No Relapse Category")

    with r_col2:
        st.metric(label="Predicted Probability (Relapse)", value=f"{risk_percentage:.1f}%")
        st.progress(int(risk_percentage))

    # --- 模型详情展示---
    with st.expander("Show Model Confidence Breakdown"):
        st.write("Detailed probability distribution from the ANN model:")
        ann_prob_no_relapse = final_proba_dist[0]
        ann_prob_relapse = final_proba_dist[1]
        c1, c2 = st.columns(2)
        c1.metric("No Relapse Probability", f"{ann_prob_no_relapse * 100:.1f}%")
        c2.metric("Relapse Probability", f"{ann_prob_relapse * 100:.1f}%")

    # --- LIME 可视化解释（修复特征数为7，适配输入）---
    st.markdown("#### 🔍 Feature Contribution Analysis (LIME)")
    st.caption("This chart shows how each feature pushed the prediction towards No Relapse (Left) or Relapse (Right).")


    # 定义 LIME 适配器
    def lime_predict_wrapper(input_array):
        df = pd.DataFrame(input_array, columns=feature_names)
        return custom_pipeline_proba(df)


    # 绘制 LIME 图
    with st.spinner("Analyzing feature importance..."):
        exp = lime_explainer.explain_instance(
            data_row=input_df.values[0],
            predict_fn=lime_predict_wrapper,
            num_features=6
        )
        fig = exp.as_pyplot_figure()
        # 调整尺寸适配网页
        fig.set_size_inches(10, 5)
        fig.patch.set_facecolor('#FAFAFA')
        plt.tight_layout()
        st.pyplot(fig)