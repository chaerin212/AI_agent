import os
import requests
import pandas as pd
import time
import json
import re
import base64
import cv2
import numpy as np
from io import BytesIO
from PIL import Image, ImageEnhance
from typing import Dict

LUXIA_API_KEY = os.getenv("LUXIA_API_KEY", "")

MODEL_NAME = "gpt-4o"
LUXIA_ENDPOINT = f"https://bridge.luxiacloud.com/llm/openai/chat/completions/{MODEL_NAME}/create"

# 묘사 전용 프롬프트
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
    """
    이미지 전처리: 빛 반사 감소 및 대비 조정
    
    Args:
        image: PIL Image
        reduce_glare: 빛 반사 감소 활성화 여부
    
    Returns:
        전처리된 PIL Image
    """
    if not reduce_glare:
        return image
    
    # PIL → OpenCV
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # 1. 히스토그램 평활화 (CLAHE) - 대비 개선
    lab = cv2.cvtColor(img_cv, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    
    lab = cv2.merge([l, a, b])
    img_cv = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    # 2. 약간의 블러로 하이라이트 완화
    img_cv = cv2.bilateralFilter(img_cv, 9, 75, 75)
    
    # OpenCV → PIL
    img_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
    
    # 3. 대비 약간 감소 (빛 반사 완화)
    enhancer = ImageEnhance.Contrast(img_pil)
    img_pil = enhancer.enhance(0.9)  # 대비 10% 감소
    
    return img_pil

def image_url_to_base64(image_url: str, preprocess: bool = True) -> str:
    """
    이미지 URL을 base64로 변환 (선택적 전처리 포함)
    
    Args:
        image_url: 이미지 URL
        preprocess: 전처리 활성화 여부
    """
    response = requests.get(image_url, timeout=30)
    response.raise_for_status()
    
    img = Image.open(BytesIO(response.content))
    
    # 전처리 적용
    if preprocess:
        img = preprocess_image(img, reduce_glare=True)
    
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    return f"data:image/png;base64,{img_base64}"

def describe_transistor(image_base64: str) -> Dict:
    """
    트랜지스터 이미지를 자세히 묘사 (몰딩만)
    
    Returns:
        {
            'molding_description': str,
            'overall_observation': str,
            'raw_response': str,
            'status': str
        }
    """
    headers = {
        "apikey": LUXIA_API_KEY,
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": DESCRIPTION_PROMPT},
                    {"type": "image_url", "image_url": {"url": image_base64}}
                ]
            }
        ],
        "temperature": 0.3,  # 낮은 값 = 일관되고 객관적인 묘사
        "stream": False
    }
    
    try:
        response = requests.post(LUXIA_ENDPOINT, json=payload, headers=headers, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            content = result["choices"][0]["message"]["content"].strip()
            
            # JSON 파싱
            try:
                # JSON 블록 제거
                clean_content = re.sub(r'```json\s*|\s*```', '', content)
                data = json.loads(clean_content)
                
                return {
                    'molding_description': data.get('molding_description', ''),
                    'status': 'success'
                }
            
            except json.JSONDecodeError as e:
                # JSON 파싱 실패 시
                print(f"    ⚠️  JSON 파싱 실패, 원본 응답 저장")
                return {
                    'molding_description': content[:500],
                    'status': 'parse_error'
                }
        
        else:
            return {
                'molding_description': '',
                'status': 'error'
            }
    
    except Exception as e:
        return {
            'molding_description': '',
            'status': 'error'
        }

def describe_batch(csv_path: str, output_csv: str, output_json: str, preprocess: bool = True):
    """
    배치로 트랜지스터 이미지들을 묘사
    
    Args:
        csv_path: 입력 CSV 경로 (id, img_url 컬럼 필요)
        output_csv: 결과 CSV 저장 경로
        output_json: 결과 JSON 저장 경로 (전체 묘사 데이터)
        preprocess: 이미지 전처리 활성화 (빛 반사 감소)
    """
    df = pd.read_csv(csv_path)
    results = []
    descriptions_json = {}
    
    print(f"🔍 트랜지스터 이미지 묘사 시작 (총 {len(df)}개)")
    print(f"🔧 전처리: {'활성화 (빛 반사 감소)' if preprocess else '비활성화'}")
    print("-" * 80)
    
    for idx, row in df.iterrows():
        img_id = row["id"]
        img_url = row["img_url"]
        
        print(f"\n[{idx+1}/{len(df)}] ID: {img_id}")
        
        try:
            # 이미지 로드 (전처리 옵션 포함)
            img_base64 = image_url_to_base64(img_url, preprocess=preprocess)
            
            # 묘사 생성
            desc = describe_transistor(img_base64)
            
            # CSV용 결과
            row_result = {
                'id': img_id,
                'img_url': img_url,
                'molding_description': desc['molding_description'],
                'status': desc['status']
            }
            results.append(row_result)
            
            # JSON용 상세 데이터
            descriptions_json[img_id] = {
                'img_url': img_url,
                'molding': desc['molding_description'],
                'status': desc['status']
            }
            
            # 출력
            print(f"  ✅ 묘사 완료")
            print(f"  📦 몰딩: {desc['molding_description'][:100]}...")
            
            # API Rate Limit 고려
            time.sleep(0.5)
            
        except Exception as e:
            print(f"  ❌ 오류: {str(e)}")
            results.append({
                'id': img_id,
                'img_url': img_url,
                'molding_description': '',
                'status': 'error'
            })
            
            descriptions_json[img_id] = {
                'img_url': img_url,
                'error': str(e),
                'status': 'error'
            }
    
    # CSV 저장
    result_df = pd.DataFrame(results)
    result_df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    
    # JSON 저장 (더 상세한 정보)
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(descriptions_json, f, ensure_ascii=False, indent=2)
    
    # 최종 통계
    print("\n" + "=" * 80)
    print("📊 묘사 완료")
    print("=" * 80)
    print(f"✅ 총 처리: {len(df)}개")
    print(f"✅ 성공: {len([r for r in results if r['status'] == 'success'])}개")
    print(f"⚠️  파싱 오류: {len([r for r in results if r['status'] == 'parse_error'])}개")
    print(f"❌ 실패: {len([r for r in results if r['status'] == 'error'])}개")
    print(f"\n💾 결과 저장:")
    print(f"   - CSV: {output_csv}")
    print(f"   - JSON: {output_json}")

if __name__ == "__main__":
    describe_batch(
        csv_path="./test.csv",
        output_csv="./transistor_descriptions_1140.csv",
        output_json="./transistor_descriptions_1140.json",
        preprocess=True  # 빛 반사 감소 전처리 활성화
    )