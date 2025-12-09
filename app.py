import streamlit as st
import torch
import gdown
import os
import re
import time
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

# ==========================================
# 0. 페이지 설정
# ==========================================
st.set_page_config(page_title="맞춤형 팀플 및 과제 부담 최소화 강의 분류 시스템", page_icon="🎓")

# ==========================================
# 1. 설정 및 모델 다운로드
# ==========================================
TEAMPLAY_FILE_ID = '1hcl250N4eFpdCpxIZlJNLQAv5FJpUvRX'  # model.pt ID
BURDEN_FILE_ID   = '1fCQ8Qr_GxJtcqAn7l91_Bf64GdDfRyyC'  # burden_model.pt ID 

MODEL_NAME = "monologg/distilkobert"
MAX_LEN = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_models():
    # 1. 파일 다운로드
    if not os.path.exists('model.pt'):
        gdown.download(f'https://drive.google.com/uc?id={TEAMPLAY_FILE_ID}', 'model.pt', quiet=True)
    if not os.path.exists('burden_model.pt'):
        gdown.download(f'https://drive.google.com/uc?id={BURDEN_FILE_ID}', 'burden_model.pt', quiet=True)

    # 2. 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    
    # 3. 팀플 모델 로드
    model_cls = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2, trust_remote_code=True)
    try:
        model_cls.load_state_dict(torch.load('model.pt', map_location=device, weights_only=False))
    except:
        model_cls.load_state_dict(torch.load('model.pt', map_location=device))
    model_cls.to(device).eval()

    # 4. 과제 부담 모델 로드
    model_reg = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=1, trust_remote_code=True)
    try:
        model_reg.load_state_dict(torch.load('burden_model.pt', map_location=device, weights_only=False))
    except:
        model_reg.load_state_dict(torch.load('burden_model.pt', map_location=device))
    model_reg.to(device).eval()

    return tokenizer, model_cls, model_reg

# ==========================================
# 2. 스마트 파싱 함수
# ==========================================
def parse_raw_text(raw_text):
    text = raw_text.replace("\n", " ")

    # (1) 과목명
    title_match = re.search(r"\(Course Title\)(.*?)이수구분", text)
    title = title_match.group(1).strip() if title_match else "과목명 미확인"

    # (2) 수업개요
    desc_match = re.search(r"\(Course Description.*?Objectives\)(.*?)교재", text)
    if not desc_match: 
        desc_match = re.search(r"\(Course Description.*?Objectives\)(.*?)참고문헌", text)
    desc = desc_match.group(1).strip() if desc_match else ""

    # (3) 수업방식
    method_match = re.search(r"\(Teaching Methods\)(.*?)학습평가방법", text)
    method = method_match.group(1).strip() if method_match else ""

    # (4) 기타안내
    note_match = re.search(r"\(Other Information.*?Notices\)(.*?)주차\(Week\)", text)
    note = note_match.group(1).strip() if note_match else ""

    if not title and not desc:
        return raw_text, "직접 입력 모드"

    full_text = f"과목명: {title} / 수업개요: {desc} / 수업방식: {method} / 기타사항: {note}"
    
    return full_text, title

# ==========================================
# 3. 추론 로직
# ==========================================
def analyze(text, tokenizer, model_cls, model_reg):
    inputs = tokenizer(text, return_tensors='pt', max_length=MAX_LEN, truncation=True, padding='max_length')
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    with torch.no_grad():
        out_cls = model_cls(input_ids=input_ids, attention_mask=attention_mask)
        probs = F.softmax(out_cls.logits, dim=1)
        prob_team = probs[0][1].item() * 100
        is_team = torch.argmax(out_cls.logits, dim=1).item()

        out_reg = model_reg(input_ids=input_ids, attention_mask=attention_mask)
        burden_score = out_reg.logits.item() * 100
        
        if is_team == 1:
            burden_score = burden_score * 1.5 + 20
        burden_score = min(max(burden_score, 0), 100)

    return is_team, prob_team, burden_score

# ==========================================
# 4. Streamlit UI 구성
# ==========================================

# 세션 상태 초기화
if 'analyzed' not in st.session_state:
    st.session_state['analyzed'] = False
if 'result' not in st.session_state:
    st.session_state['result'] = {}
if 'feedback_submitted' not in st.session_state:
    st.session_state['feedback_submitted'] = False
# 입력창 초기화를 위한 Key 관리
if 'input_key' not in st.session_state:
    st.session_state['input_key'] = 0

st.title("팀플 및 과제 부담 최소화 강의 분류 시스템")
st.markdown("""
**강의계획서 전체를 복사해서 붙여넣으세요!**
\n AI가 분석에 필요한 텍스트만 추출하여 분석하겠습니다. 
""")

with st.sidebar:
    st.header("About Project")
    st.write("### AI가 강의계획서를 통해 팀플 유무와, 과제 부담 지수를 예측하여 분류합니다.")
    st.write("### Student ID/Name: 202401394 박의진")
    st.write("### Model: Dual DistilKoBERT")
    st.write("### Credit : Gemini pro 3의 도움을 받아 제작하였습니다.")

# key 값을 변수로 설정하여, 리셋 버튼 클릭 시 강제로 새 입력창을 생성하게 함
raw_input = st.text_area(
    "강의계획서 전체 텍스트 붙여넣기:", 
    height=300, 
    placeholder="종합정보시스템 강의계획서 화면을 전체 복사(Ctrl+A, Ctrl+C)해서 여기에 붙여넣으세요.",
    key=f"syllabus_input_{st.session_state['input_key']}" 
)

if st.button("강의 분류 시작", type="primary"):
    if not raw_input:
        st.warning("내용을 입력해주세요!")
    else:
        # 진행바
        progress_text = "AI 분석 준비 중..."
        my_bar = st.progress(0, text=progress_text)

        time.sleep(0.3)
        my_bar.progress(30, text="📥 딥러닝 모델 로딩 중...")
        tokenizer, model_cls, model_reg = load_models()
        
        time.sleep(0.3)
        my_bar.progress(60, text="🔍 텍스트 핵심 정보 추출 중...")
        final_input, course_title = parse_raw_text(raw_input)
        
        time.sleep(0.3)
        my_bar.progress(90, text="🤖 팀플 위험도 및 과제 부담 예측 중...")
        is_team, prob_team, burden = analyze(final_input, tokenizer, model_cls, model_reg)
        
        my_bar.progress(100, text="✅ 분석 완료!")
        time.sleep(0.3)
        my_bar.empty()

        # 결과 저장 및 상태 업데이트
        st.session_state['analyzed'] = True
        st.session_state['feedback_submitted'] = False
        st.session_state['result'] = {
            'course_title': course_title,
            'is_team': is_team,
            'prob_team': prob_team,
            'burden': burden,
            'final_input': final_input
        }
        st.rerun()

# 결과 화면 표시
if st.session_state['analyzed']:
    res = st.session_state['result']
    
    st.divider()
    st.subheader(f"📂 분석 결과: {res['course_title']}")
    
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("팀플 위험도")
        if res['is_team'] == 1:
            st.error(f"🚨 팀플 있음 ({res['prob_team']:.1f}%)")
            st.write("조별 과제가 감지되었습니다. \n수강 시 유의하세요!")
        else:
            st.success(f"🍀 팀플 없음 ({res['prob_team']:.1f}%)")
            st.write("개인 과제 위주이거나 \n이론 수업일 확률이 높습니다.")

    with col2:
        st.subheader("과제 부담 지수")
        st.metric(label="Burden Score", value=f"{res['burden']:.1f}/100점")
        if res['burden'] > 60:
            st.write("🔥 **수강 주의** 🔥 (과제 많음)")
        elif res['burden'] > 30:
            st.write("**보통**")
        else:
            st.write("🍯 **꿀강 확정** 🍯 (과제 적음)")
    
    st.divider()
    
    # [수정됨] 하이라이팅 제거하고 텍스트만 표시
    with st.expander("🤖 AI가 추출한 핵심 내용 보기 (판단 근거)"):
        st.code(res['final_input'], language="text")
        st.caption("AI는 위 계획서에서 개요, 평가방식, 기타 등의 텍스트를 바탕으로 판단했습니다.")

    st.divider()
    st.markdown("##### 📢 분석 결과가 정확한가요?")
    
    # 피드백 영역
    if not st.session_state['feedback_submitted']:
        col_f1, col_f2, col_f3 = st.columns([1, 1, 3])
        
        if col_f1.button("👍 정확해요"):
            st.session_state['feedback_submitted'] = True
            st.rerun() 

        if col_f2.button("👎 틀렸어요"):
            st.session_state['feedback_submitted'] = True
            st.rerun() 
            
    else:
        st.success("✅ 피드백 감사합니다! 모델 개선에 활용하겠습니다. 🙇‍♂️")

    # 리셋 버튼
    st.divider()
    if st.button("🔄 다른 강의 분석하기"):
        st.session_state['analyzed'] = False
        st.session_state['result'] = {}
        st.session_state['feedback_submitted'] = False
        st.session_state['input_key'] += 1 
        st.rerun()