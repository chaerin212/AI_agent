import os
import sys
import importlib.util
import requests
import pandas as pd
import numpy as np
import cv2
import time
import json
import re
import base64
from io import BytesIO
from PIL import Image, ImageEnhance
from typing import Dict, List, Tuple

# =========================
# 공통 설정
# =========================
LUXIA_API_KEY = os.getenv("LUXIA_API_KEY", "")
MODEL_NAME = "gpt-4o"
LUXIA_ENDPOINT = f"https://bridge.luxiacloud.com/llm/openai/chat/completions/{MODEL_NAME}/create"
LUXIA_ENDPOINT_LUXIA = "https://bridge.luxiacloud.com/luxia/v1/chat"
MODEL_NAME_LUXIA = "luxia3-llm-32b-0731"

# =========================
# 1. transi_analysis_1140.py 내용 (묘사 생성)
# =========================
DESCRIPTION_PROMPT = """You are a detailed visual inspector for transistor components.

[REFERENCE - Normal Transistor Appearance]
A normal, properly mounted transistor should have:
- MOLDING: Rectangular black plastic body with smooth, intact surface

Now, describe what you see in the given transistor image, comparing it to this reference.

Focus on MOLDING (Black plastic body):
   - Edges: clean, chipped, intact?
   - Any visible damage, cracks, or irregularities?

IMPORTANT: 
- DO NOT make judgments
- ONLY describe what you actually see
- Compare observed features to the reference description above
- Be specific and detailed
- Use objective, factual language
- **IGNORE light reflections or glare** - they may make rectangular shapes appear cylindrical
- Focus on ACTUAL physical defects (cracks, chips, damage), NOT optical illusions from lighting

Respond in JSON format:
{
  "molding_description": "Detailed description of the molding/body"
}

Output ONLY valid JSON, no additional text.
"""

def preprocess_image(image: Image.Image, reduce_glare: bool = True) -> Image.Image:
    """이미지 전처리: 빛 반사 감소 및 대비 조정"""
    if not reduce_glare:
        return image
    
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(img_cv, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    img_cv = cv2.cvtColor(lab, cv2.COLOR_BGR2RGB)
    img_cv = cv2.bilateralFilter(img_cv, 9, 75, 75)
    img_pil = Image.fromarray(img_cv)
    enhancer = ImageEnhance.Contrast(img_pil)
    img_pil = enhancer.enhance(0.9)
    return img_pil

def image_url_to_base64(image_url: str, preprocess: bool = True) -> str:
    """이미지 URL을 base64로 변환"""
    response = requests.get(image_url, timeout=30)
    response.raise_for_status()
    img = Image.open(BytesIO(response.content))
    if preprocess:
        img = preprocess_image(img, reduce_glare=True)
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return f"data:image/png;base64,{img_base64}"

def describe_transistor(image_base64: str) -> Dict:
    """트랜지스터 이미지를 자세히 묘사 (몰딩만)"""
    headers = {
        "apikey": LUXIA_API_KEY,
        "Content-Type": "application/json"
    }
    payload = {
        "model": MODEL_NAME,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": DESCRIPTION_PROMPT},
                {"type": "image_url", "image_url": {"url": image_base64}}
            ]
        }],
        "temperature": 0.3,
        "stream": False
    }
    try:
        response = requests.post(LUXIA_ENDPOINT, json=payload, headers=headers, timeout=60)
        if response.status_code == 200:
            result = response.json()
            content = result["choices"][0]["message"]["content"].strip()
            try:
                clean_content = re.sub(r'```json\s*|\s*```', '', content)
                data = json.loads(clean_content)
                return {
                    'molding_description': data.get('molding_description', ''),
                    'status': 'success'
                }
            except json.JSONDecodeError:
                return {
                    'molding_description': content[:500],
                    'status': 'parse_error'
                }
        return {'molding_description': '', 'status': 'error'}
    except Exception as e:
        return {'molding_description': '', 'status': 'error'}

def describe_batch_transistor(csv_path: str, output_json: str, preprocess: bool = True):
    """배치로 트랜지스터 이미지들을 묘사"""
    df = pd.read_csv(csv_path)
    descriptions_json = {}
    
    print(f"🔍 트랜지스터 이미지 묘사 시작 (총 {len(df)}개)")
    print(f"🔧 전처리: {'활성화 (빛 반사 감소)' if preprocess else '비활성화'}")
    print("-" * 80)
    
    for idx, row in df.iterrows():
        img_id = row["id"]
        img_url = row["img_url"]
        print(f"\n[{idx+1}/{len(df)}] ID: {img_id}")
        
        try:
            img_base64 = image_url_to_base64(img_url, preprocess=preprocess)
            desc = describe_transistor(img_base64)
            descriptions_json[img_id] = {
                'img_url': img_url,
                'molding': desc['molding_description'],
                'status': desc['status']
            }
            print(f"  ✅ 묘사 완료")
            time.sleep(0.5)
        except Exception as e:
            print(f"  ❌ 오류: {str(e)}")
            descriptions_json[img_id] = {
                'img_url': img_url,
                'error': str(e),
                'status': 'error'
            }
    
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(descriptions_json, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 결과 저장: {output_json}")

# =========================
# 2. judge_molding.py 내용 (molding 판단)
# =========================
JUDGMENT_PROMPT = """You are an expert quality inspector analyzing transistor molding descriptions.

**MOLDING DESCRIPTION:**
{molding_text}

**YOUR TASK:**
Based ONLY on the molding description above, determine if there appears to be an abnormality/defect.

**CLASSIFICATION:**
- "abnormal": Clear signs of defects including:
  * Chips, cracks, damage, misalignment
  * NO MOLDING VISIBLE or molding is MISSING/ABSENT
- "normal": No issues mentioned, everything appears intact and proper
- "ambiguous": Unclear or insufficient information to make a determination

**IMPORTANT:**
- If the description indicates NO MOLDING or MISSING MOLDING, classify as "abnormal"
- Extract the EXACT sentence(s) from the description that support your judgment
- Use ONLY information provided in the description
- Be objective and evidence-based

Respond in JSON format:
{{
  "judgment": "abnormal" | "normal" | "ambiguous",
  "reason": "The exact sentence(s) from the description that support this judgment"
}}

Output ONLY valid JSON, no additional text.
"""

def judge_molding(molding_text: str) -> Dict:
    """Molding 설명을 분석해서 이상 여부 판단"""
    if not molding_text or molding_text.strip() == "":
        return {
            'judgment': 'abnormal',
            'reason': 'No molding description provided - molding is missing or not visible.',
            'status': 'no_data'
        }
    
    headers = {
        "apikey": LUXIA_API_KEY,
        "Content-Type": "application/json"
    }
    prompt = JUDGMENT_PROMPT.format(molding_text=molding_text)
    payload = {
        "model": MODEL_NAME_LUXIA,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_completion_tokens": 2048,
        "top_p": 1,
        "frequency_penalty": 0.1,
        "stream": False
    }
    
    try:
        response = requests.post(LUXIA_ENDPOINT_LUXIA, json=payload, headers=headers, timeout=60)
        if response.status_code == 200:
            result = response.json()
            if "choices" in result and len(result["choices"]) > 0:
                content = result["choices"][0]["message"]["content"].strip()
            else:
                return {'judgment': 'ambiguous', 'reason': 'Invalid API response format', 'status': 'error'}
            
            try:
                clean_content = re.sub(r'```json\s*|\s*```', '', content)
                data = json.loads(clean_content)
                return {
                    'judgment': data.get('judgment', 'ambiguous'),
                    'reason': data.get('reason', ''),
                    'status': 'success'
                }
            except json.JSONDecodeError:
                return {'judgment': 'ambiguous', 'reason': content[:200], 'status': 'parse_error'}
        return {'judgment': 'ambiguous', 'reason': f'API Error: {response.status_code}', 'status': 'error'}
    except Exception as e:
        return {'judgment': 'ambiguous', 'reason': f'Error: {str(e)}', 'status': 'error'}

def process_all_moldings(input_json_path: str) -> Dict[str, int]:
    """모든 molding 설명을 판단하고 label 딕셔너리 반환"""
    print(f"📖 입력 파일 읽는 중: {input_json_path}")
    with open(input_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    print(f"✅ 총 {len(data)}개 이미지 처리 시작")
    print("=" * 80)
    
    labels = {}  # {img_id: label (0 or 1)}
    
    for idx, (img_id, item) in enumerate(data.items()):
        print(f"\n[{idx+1}/{len(data)}] ID: {img_id}")
        molding_text = item.get("molding", "") or item.get("molding_description", "")
        
        if not molding_text:
            print("  ❌ Molding 설명 없음 → 비정상 (label=1)")
            labels[img_id] = 1
            continue
        
        judgment_result = judge_molding(molding_text)
        judgment = judgment_result['judgment']
        
        # abnormal=1, normal=0, ambiguous=1
        if judgment == 'abnormal':
            label = 1
            print(f"  ❌ 판단: 이상 있음 (label=1)")
        elif judgment == 'normal':
            label = 0
            print(f"  ✅ 판단: 정상 (label=0)")
        else:
            label = 1  # ambiguous도 안전하게 1
            print(f"  ⚪ 판단: 애매함 (label=1)")
        
        labels[img_id] = label
        time.sleep(1)
    
    return labels

# =========================
# 3. main.py 내용 (리드 삽입 판단) - 필요한 함수들만 import하거나 복사
# =========================
# main.py의 모든 함수를 여기에 포함해야 합니다.
# 파일이 크므로 주요 함수만 import하는 방식으로 진행하겠습니다.

# main.py에서 필요한 함수들을 import
# (실제로는 main.py의 모든 내용을 여기에 포함해야 하지만, 
#  파일이 너무 크므로 함수 호출 방식으로 진행)

# =========================
# 4. 통합 파이프라인
# =========================
def run_integrated_pipeline(csv_path: str, output_csv: str):
    """
    통합 파이프라인 실행
    
    Args:
        csv_path: 입력 CSV 파일 (id, img_url 컬럼 필요)
        output_csv: 최종 출력 CSV 파일 (id, label)
    """
    print("🚀" * 40)
    print("🤖 통합 파이프라인 시작")
    print("🚀" * 40 + "\n")
    
    # 임시 파일 경로
    temp_description_json = "./temp_descriptions.json"
    temp_lead_csv = "./temp_lead_results.csv"
    temp_lead_json = "./temp_lead_results.json"
    
    # Stage 1: Molding Description 생성
    print("=" * 80)
    print("🔍 Stage 1: Molding Description 생성 중...")
    print("=" * 80 + "\n")
    try:
        describe_batch_transistor(
            csv_path=csv_path,
            output_json=temp_description_json,
            preprocess=True
        )
        print("\n✅ Stage 1 완료!\n")
    except Exception as e:
        print(f"❌ Stage 1 실패: {str(e)}")
        raise
    
    # Stage 2: Molding 이상 판단 (judge_molding)
    print("=" * 80)
    print("⚖️  Stage 2: Molding 이상 판단 중...")
    print("=" * 80 + "\n")
    try:
        molding_labels = process_all_moldings(temp_description_json)
        print("\n✅ Stage 2 완료!\n")
    except Exception as e:
        print(f"❌ Stage 2 실패: {str(e)}")
        raise
    
    # Stage 3: 리드 삽입 판단 (main.py)
    print("=" * 80)
    print("🔌 Stage 3: 리드 삽입 판단 중...")
    print("=" * 80 + "\n")
    try:
        # main.py의 describe_batch 함수 호출
        # main.py를 동적으로 import
        main_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "main.py")
        if os.path.exists(main_path):
            spec = importlib.util.spec_from_file_location("main_module", main_path)
            main_module = importlib.util.module_from_spec(spec)
            sys.modules['main_module'] = main_module
            spec.loader.exec_module(main_module)
            main_module.describe_batch(
                csv_path=csv_path,
                output_csv=temp_lead_csv,
                output_json=temp_lead_json
            )
        else:
            # main.py가 같은 디렉토리에 없으면 직접 import 시도
            from main import describe_batch as describe_batch_lead
            describe_batch_lead(
                csv_path=csv_path,
                output_csv=temp_lead_csv,
                output_json=temp_lead_json
            )
        print("\n✅ Stage 3 완료!\n")
    except Exception as e:
        print(f"❌ Stage 3 실패: {str(e)}")
        import traceback
        traceback.print_exc()
        raise
    
    # Stage 4: 결과 병합
    print("=" * 80)
    print("🔀 Stage 4: 결과 병합 중...")
    print("=" * 80 + "\n")
    
    try:
        # main.py 결과 읽기 (CSV)
        lead_df = pd.read_csv(temp_lead_csv)
        lead_labels = {}  # {id: label}
        for _, row in lead_df.iterrows():
            lead_labels[row['id']] = row['label']
        
        # 최종 결과 생성: 하나라도 1이면 1, 둘 다 0이면 0
        df = pd.read_csv(csv_path)
        final_results = []
        
        for _, row in df.iterrows():
            img_id = row['id']
            molding_label = molding_labels.get(img_id, 1)  # 없으면 1 (안전하게)
            lead_label = lead_labels.get(img_id, 1)  # 없으면 1 (안전하게)
            
            # 하나라도 1이면 1, 둘 다 0이면 0
            final_label = 1 if (molding_label == 1 or lead_label == 1) else 0
            
            final_results.append({
                'id': img_id,
                'label': final_label
            })
            
            print(f"  ID: {img_id} - Molding: {molding_label}, Lead: {lead_label} → Final: {final_label}")
        
        # 최종 CSV 저장
        result_df = pd.DataFrame(final_results)
        result_df.to_csv(output_csv, index=False, encoding='utf-8-sig')
        
        # 통계
        total = len(final_results)
        label_0_count = sum(1 for r in final_results if r['label'] == 0)
        label_1_count = sum(1 for r in final_results if r['label'] == 1)
        
        print("\n" + "=" * 80)
        print("📊 최종 결과")
        print("=" * 80)
        print(f"✅ 총 처리: {total}개")
        print(f"   - Label 0 (정상): {label_0_count}개 ({label_0_count/total*100:.1f}%)")
        print(f"   - Label 1 (비정상): {label_1_count}개 ({label_1_count/total*100:.1f}%)")
        print(f"\n💾 최종 결과 저장: {output_csv}")
        
        print("\n" + "🎉" * 40)
        print("✅ 전체 파이프라인 완료!")
        print("🎉" * 40)
        
    except Exception as e:
        print(f"❌ Stage 4 실패: {str(e)}")
        raise

if __name__ == "__main__":
    run_integrated_pipeline(
        csv_path="./test.csv",
        output_csv="./final_results.csv"
    )
