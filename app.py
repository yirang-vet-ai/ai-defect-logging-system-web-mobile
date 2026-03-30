import os
import io
import json
import sqlite3
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# =========================
# 1. 기본 설정
# =========================
MODEL_PATH = "mobilenetv2_neu_defect_best.keras"
CLASS_INDEX_PATH = "class_indices.json"
DB_PATH = "defect_records.db"
UPLOAD_DIR = Path("uploads")
IMG_SIZE = 224

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# 2. 페이지 설정
# =========================
st.set_page_config(
    page_title="불량 즉시 기록 시스템",
    layout="wide"
)

# =========================
# 3. DB 함수
# =========================
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS defect_records (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        worker_name TEXT,
        line_name TEXT,
        pred_class TEXT,
        confidence REAL,
        image_path TEXT,
        memo TEXT
    )
    """)

    conn.commit()
    conn.close()

def insert_record(timestamp, worker_name, line_name, pred_class, confidence, image_path, memo):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
    INSERT INTO defect_records (
        timestamp, worker_name, line_name, pred_class, confidence, image_path, memo
    ) VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        timestamp,
        worker_name,
        line_name,
        pred_class,
        confidence,
        image_path,
        memo
    ))

    conn.commit()
    conn.close()

def load_all_records():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        "SELECT * FROM defect_records ORDER BY id DESC",
        conn
    )
    conn.close()
    return df

# =========================
# 4. 모델 및 클래스 로드
# =========================
@st.cache_resource
def load_defect_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"모델 파일이 없습니다: {MODEL_PATH}")
    model = load_model(MODEL_PATH)
    return model

@st.cache_data
def load_class_names():
    if not os.path.exists(CLASS_INDEX_PATH):
        raise FileNotFoundError(f"class_indices 파일이 없습니다: {CLASS_INDEX_PATH}")

    with open(CLASS_INDEX_PATH, "r", encoding="utf-8") as f:
        class_indices = json.load(f)

    idx_to_class = {v: k for k, v in class_indices.items()}
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
    return class_names

# =========================
# 5. 전처리 / 예측 함수
# =========================
def preprocess_pil_image(pil_img):
    pil_img = pil_img.convert("RGB")
    pil_img = pil_img.resize((IMG_SIZE, IMG_SIZE))
    x = np.array(pil_img).astype("float32")
    x = preprocess_input(x)
    x = np.expand_dims(x, axis=0)
    return x

def predict_image(model, class_names, pil_img):
    x = preprocess_pil_image(pil_img)
    pred = model.predict(x, verbose=0)[0]

    pred_idx = int(np.argmax(pred))
    pred_class = class_names[pred_idx]
    confidence = float(pred[pred_idx])

    prob_df = pd.DataFrame({
        "class": class_names,
        "probability": pred
    }).sort_values("probability", ascending=False).reset_index(drop=True)

    return pred_class, confidence, prob_df

def save_image_file(pil_img):
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = UPLOAD_DIR / f"defect_{timestamp_str}.jpg"
    pil_img.save(file_path)
    return str(file_path)

# =========================
# 6. 초기화
# =========================
init_db()
model = load_defect_model()
class_names = load_class_names()

# =========================
# 7. 사이드바 메뉴
# =========================
menu = st.sidebar.radio(
    "메뉴 선택",
    ["불량 등록", "기록 조회", "통계 대시보드"]
)

st.title("작업자 스마트폰 기반 불량 즉시 기록 시스템")

# =========================
# 8. 불량 등록 페이지
# =========================
if menu == "불량 등록":
    st.subheader("불량 등록")

    col1, col2 = st.columns([1, 1])

    with col1:
        worker_name = st.text_input("작업자 이름")
        line_name = st.text_input("라인 이름")
        memo = st.text_area("메모")

    with col2:
        st.write("스마트폰에서는 카메라 촬영 또는 파일 업로드를 사용할 수 있습니다.")
        camera_image = st.camera_input("카메라로 촬영")
        uploaded_file = st.file_uploader("또는 이미지 업로드", type=["jpg", "jpeg", "png", "bmp"])

    image_source = None

    if camera_image is not None:
        image_source = camera_image
    elif uploaded_file is not None:
        image_source = uploaded_file

    if image_source is not None:
        pil_img = Image.open(image_source).convert("RGB")
        st.image(pil_img, caption="입력 이미지", use_container_width=True)

        if st.button("예측 실행"):
            pred_class, confidence, prob_df = predict_image(model, class_names, pil_img)

            st.session_state["pred_class"] = pred_class
            st.session_state["confidence"] = confidence
            st.session_state["prob_df"] = prob_df
            st.session_state["pil_img"] = pil_img

        if "pred_class" in st.session_state:
            st.success(f"예측 결과: {st.session_state['pred_class']}")
            st.info(f"신뢰도: {st.session_state['confidence']:.4f}")

            st.write("클래스별 확률")
            st.dataframe(st.session_state["prob_df"], use_container_width=True)

            if st.button("결과 저장"):
                image_path = save_image_file(st.session_state["pil_img"])

                insert_record(
                    timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    worker_name=worker_name,
                    line_name=line_name,
                    pred_class=st.session_state["pred_class"],
                    confidence=float(st.session_state["confidence"]),
                    image_path=image_path,
                    memo=memo
                )

                st.success("DB 저장 완료")

# =========================
# 9. 기록 조회 페이지
# =========================
elif menu == "기록 조회":
    st.subheader("기록 조회")

    df = load_all_records()

    if df.empty:
        st.warning("저장된 기록이 없습니다.")
    else:
        col1, col2 = st.columns(2)

        with col1:
            class_filter = st.selectbox("불량 유형 필터", ["전체"] + class_names)

        with col2:
            worker_filter = st.text_input("작업자 이름 검색")

        filtered_df = df.copy()

        if class_filter != "전체":
            filtered_df = filtered_df[filtered_df["pred_class"] == class_filter]

        if worker_filter.strip():
            filtered_df = filtered_df[
                filtered_df["worker_name"].fillna("").str.contains(worker_filter.strip(), case=False)
            ]

        st.dataframe(filtered_df, use_container_width=True)

        csv_data = filtered_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            label="CSV 다운로드",
            data=csv_data,
            file_name="defect_records.csv",
            mime="text/csv"
        )

        st.write("최근 저장 이미지 보기")
        if len(filtered_df) > 0:
            selected_id = st.selectbox("기록 ID 선택", filtered_df["id"].tolist())
            row = filtered_df[filtered_df["id"] == selected_id].iloc[0]

            st.write(f"시간: {row['timestamp']}")
            st.write(f"작업자: {row['worker_name']}")
            st.write(f"라인: {row['line_name']}")
            st.write(f"예측 클래스: {row['pred_class']}")
            st.write(f"신뢰도: {row['confidence']:.4f}")
            st.write(f"메모: {row['memo']}")

            image_path = row["image_path"]
            if isinstance(image_path, str) and os.path.exists(image_path):
                st.image(image_path, caption=f"ID {selected_id} 이미지", use_container_width=True)
            else:
                st.warning("저장 이미지 파일을 찾을 수 없습니다.")

# =========================
# 10. 통계 대시보드
# =========================
elif menu == "통계 대시보드":
    st.subheader("통계 대시보드")

    df = load_all_records()

    if df.empty:
        st.warning("저장된 기록이 없습니다.")
    else:
        total_count = len(df)
        top_class = df["pred_class"].value_counts().idxmax()
        avg_conf = df["confidence"].mean()

        c1, c2, c3 = st.columns(3)
        c1.metric("총 등록 건수", total_count)
        c2.metric("가장 많은 불량", top_class)
        c3.metric("평균 신뢰도", f"{avg_conf:.4f}")

        st.write("불량 유형별 건수")
        defect_count = df["pred_class"].value_counts().reset_index()
        defect_count.columns = ["pred_class", "count"]
        st.bar_chart(defect_count.set_index("pred_class"))

        st.write("작업자별 등록 건수")
        worker_count = df["worker_name"].fillna("미입력").replace("", "미입력").value_counts().reset_index()
        worker_count.columns = ["worker_name", "count"]
        st.bar_chart(worker_count.set_index("worker_name"))

        st.write("라인별 등록 건수")
        line_count = df["line_name"].fillna("미입력").replace("", "미입력").value_counts().reset_index()
        line_count.columns = ["line_name", "count"]
        st.bar_chart(line_count.set_index("line_name"))