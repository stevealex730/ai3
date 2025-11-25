# streamlit_app.py
import os, re
from io import BytesIO
import numpy as np
import streamlit as st
from PIL import Image, ImageOps
from fastai.vision.all import *
import gdown

# ======================
# 페이지/스타일
# ======================
st.set_page_config(page_title="Fastai 이미지 분류기", page_icon="🤖", layout="wide")
st.markdown("""
<style>
h1 { color:#1E88E5; text-align:center; font-weight:800; letter-spacing:-0.5px; }
.prediction-box { background:#E3F2FD; border:2px solid #1E88E5; border-radius:12px; padding:22px; text-align:center; margin:16px 0; box-shadow:0 4px 10px rgba(0,0,0,.06);}
.prediction-box h2 { color:#0D47A1; margin:0; font-size:2.0rem; }
.prob-card { background:#fff; border-radius:10px; padding:12px 14px; margin:10px 0; box-shadow:0 2px 6px rgba(0,0,0,.06); }
.prob-bar-bg { background:#ECEFF1; border-radius:6px; width:100%; height:22px; overflow:hidden; }
.prob-bar-fg { background:#4CAF50; height:100%; border-radius:6px; transition:width .5s; }
.prob-bar-fg.highlight { background:#FF6F00; }
.info-grid { display:grid; grid-template-columns:repeat(12,1fr); gap:14px; }
.card { border:1px solid #e3e6ea; border-radius:12px; padding:14px; background:#fff; box-shadow:0 2px 6px rgba(0,0,0,.05); }
.card h4 { margin:0 0 10px; font-size:1.05rem; color:#0D47A1; }
.thumb { width:100%; height:auto; border-radius:10px; display:block; }
.thumb-wrap { position:relative; display:block; }
.play { position:absolute; top:50%; left:50%; transform:translate(-50%,-50%); width:60px; height:60px; border-radius:50%; background:rgba(0,0,0,.55); }
.play:after{ content:''; border-style:solid; border-width:12px 0 12px 20px; border-color:transparent transparent transparent #fff; position:absolute; top:50%; left:50%; transform:translate(-40%,-50%); }
.helper { color:#607D8B; font-size:.9rem; }
.stFileUploader, .stCameraInput { border:2px dashed #1E88E5; border-radius:12px; padding:16px; background:#f5fafe; }
</style>
""", unsafe_allow_html=True)

st.title("이미지 분류기 (Fastai) — 확률 막대 + 라벨별 고정 콘텐츠")

# ======================
# 세션 상태
# ======================
if "img_bytes" not in st.session_state:
    st.session_state.img_bytes = None
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None

# ======================
# 모델 로드
# ======================
FILE_ID = st.secrets.get("GDRIVE_FILE_ID", "1YuLCetTh_egOtS9mxzEzwGdazEYMn3Dm")
MODEL_PATH = st.secrets.get("MODEL_PATH", "model.pkl")

@st.cache_resource
def load_model_from_drive(file_id: str, output_path: str):
    if not os.path.exists(output_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output_path, quiet=False)
    return load_learner(output_path, cpu=True)

with st.spinner("🤖 모델 로드 중..."):
    learner = load_model_from_drive(FILE_ID, MODEL_PATH)
st.success("✅ 모델 로드 완료")

labels = [str(x) for x in learner.dls.vocab]
st.write(f"**분류 가능한 항목:** `{', '.join(labels)}`")
st.markdown("---")

# ======================
# 라벨 이름 매핑: 여기를 채우세요!
# 각 라벨당 최대 3개씩 표시됩니다.
# ======================
CONTENT_BY_LABEL: dict[str, dict[str, list[str]]] = {
    # 예)
    # "짬뽕": {
    #   "texts": ["짬뽕의 특징과 유래", "국물 맛 포인트", "지역별 스타일 차이"],
    #   "images": ["https://.../jjampong1.jpg", "https://.../jjampong2.jpg"],
    #   "videos": ["https://youtu.be/XXXXXXXXXXX"]
    # },
    labels[0] : ["중국식 냉면은 맛있어"], "image" : ["https://www.esquirekorea.co.kr/resources_old/online/org_online_image/eq/71c93efd-352d-4fb4-8a98-dd1b51475442.jpg"]},
    labels[1] : ["짜장면은 맛있어"], "image" : ["https://i.namu.wiki/i/j2AxLP9AtrcJebh4DVfGxowfXwI3a95dG_YZb_Ktczc6Ca7ACyd_NJL3YHQMw8SABGTQiJDwSpySOSSBLZVEZw.webp"]},
    labels[2] : ["짬뽕은 맛있어"], "image" : ["https://blog.kakaocdn.net/dna/YPxRW/btrzhpNljHH/AAAAAAAAAAAAAAAAAAAAAAhVpctCZeeRfUJSzJ9VBLKsQHsA38Gk5_KTV934P7vk/img.jpg?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1764514799&allow_ip=&allow_referer=&signature=KtGzPuSD0MLN59%2BpAsKcHnaNZ0U%3D"]},
    labels[3] : ["탕수육은 맛있어"], "image" : ["data:image/jpeg;base64,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"]},
}

# ======================
# 유틸
# ======================
def load_pil_from_bytes(b: bytes) -> Image.Image:
    pil = Image.open(BytesIO(b))
    pil = ImageOps.exif_transpose(pil)
    if pil.mode != "RGB": pil = pil.convert("RGB")
    return pil

def yt_id_from_url(url: str) -> str | None:
    if not url: return None
    pats = [r"(?:v=|/)([0-9A-Za-z_-]{11})(?:\?|&|/|$)", r"youtu\.be/([0-9A-Za-z_-]{11})"]
    for p in pats:
        m = re.search(p, url)
        if m: return m.group(1)
    return None

def yt_thumb(url: str) -> str | None:
    vid = yt_id_from_url(url)
    return f"https://img.youtube.com/vi/{vid}/hqdefault.jpg" if vid else None

def pick_top3(lst):
    return [x for x in lst if isinstance(x, str) and x.strip()][:3]

def get_content_for_label(label: str):
    """라벨명으로 콘텐츠 반환 (texts, images, videos). 없으면 빈 리스트."""
    cfg = CONTENT_BY_LABEL.get(label, {})
    return (
        pick_top3(cfg.get("texts", [])),
        pick_top3(cfg.get("images", [])),
        pick_top3(cfg.get("videos", [])),
    )

# ======================
# 입력(카메라/업로드)
# ======================
tab_cam, tab_file = st.tabs(["📷 카메라로 촬영", "📁 파일 업로드"])
new_bytes = None

with tab_cam:
    cam = st.camera_input("카메라 스냅샷", label_visibility="collapsed")
    if cam is not None:
        new_bytes = cam.getvalue()

with tab_file:
    f = st.file_uploader("이미지를 업로드하세요 (jpg, png, jpeg, webp, tiff)",
                         type=["jpg","png","jpeg","webp","tiff"])
    if f is not None:
        new_bytes = f.getvalue()

if new_bytes:
    st.session_state.img_bytes = new_bytes

# ======================
# 예측 & 레이아웃
# ======================
if st.session_state.img_bytes:
    top_l, top_r = st.columns([1, 1], vertical_alignment="center")

    pil_img = load_pil_from_bytes(st.session_state.img_bytes)
    with top_l:
        st.image(pil_img, caption="입력 이미지", use_container_width=True)

    with st.spinner("🧠 분석 중..."):
        pred, pred_idx, probs = learner.predict(PILImage.create(np.array(pil_img)))
        st.session_state.last_prediction = str(pred)

    with top_r:
        st.markdown(
            f"""
            <div class="prediction-box">
                <span style="font-size:1.0rem;color:#555;">예측 결과:</span>
                <h2>{st.session_state.last_prediction}</h2>
                <div class="helper">오른쪽 패널에서 예측 라벨의 콘텐츠가 표시됩니다.</div>
            </div>
            """, unsafe_allow_html=True
        )

    left, right = st.columns([1,1], vertical_alignment="top")

    # 왼쪽: 확률 막대
    with left:
        st.subheader("상세 예측 확률")
        prob_list = sorted(
            [(labels[i], float(probs[i])) for i in range(len(labels))],
            key=lambda x: x[1], reverse=True
        )
        for lbl, p in prob_list:
            pct = p * 100
            hi = "highlight" if lbl == st.session_state.last_prediction else ""
            st.markdown(
                f"""
                <div class="prob-card">
                  <div style="display:flex;justify-content:space-between;margin-bottom:6px;">
                    <strong>{lbl}</strong><span>{pct:.2f}%</span>
                  </div>
                  <div class="prob-bar-bg">
                    <div class="prob-bar-fg {hi}" style="width:{pct:.4f}%;"></div>
                  </div>
                </div>
                """, unsafe_allow_html=True
            )

    # 오른쪽: 정보 패널 (예측 라벨 기본, 다른 라벨로 바꿔보기 가능)
    with right:
        st.subheader("라벨별 고정 콘텐츠")
        default_idx = labels.index(st.session_state.last_prediction) if st.session_state.last_prediction in labels else 0
        info_label = st.selectbox("표시할 라벨 선택", options=labels, index=default_idx)

        texts, images, videos = get_content_for_label(info_label)

        if not any([texts, images, videos]):
            st.info(f"라벨 `{info_label}`에 대한 콘텐츠가 아직 없습니다. 코드의 CONTENT_BY_LABEL에 추가하세요.")
        else:
            # 텍스트
            if texts:
                st.markdown('<div class="info-grid">', unsafe_allow_html=True)
                for t in texts:
                    st.markdown(f"""
                    <div class="card" style="grid-column:span 12;">
                      <h4>텍스트</h4>
                      <div>{t}</div>
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            # 이미지(최대 3, 3열)
            if images:
                st.markdown('<div class="info-grid">', unsafe_allow_html=True)
                for url in images[:3]:
                    st.markdown(f"""
                    <div class="card" style="grid-column:span 4;">
                      <h4>이미지</h4>
                      <img src="{url}" class="thumb" />
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            # 동영상(유튜브 썸네일)
            if videos:
                st.markdown('<div class="info-grid">', unsafe_allow_html=True)
                for v in videos[:3]:
                    thumb = yt_thumb(v)
                    if thumb:
                        st.markdown(f"""
                        <div class="card" style="grid-column:span 6;">
                          <h4>동영상</h4>
                          <a href="{v}" target="_blank" class="thumb-wrap">
                            <img src="{thumb}" class="thumb"/>
                            <div class="play"></div>
                          </a>
                          <div class="helper">{v}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="card" style="grid-column:span 6;">
                          <h4>동영상</h4>
                          <a href="{v}" target="_blank">{v}</a>
                        </div>
                        """, unsafe_allow_html=True)
else:
    st.info("카메라로 촬영하거나 파일을 업로드하면 분석 결과와 라벨별 콘텐츠가 표시됩니다.")
