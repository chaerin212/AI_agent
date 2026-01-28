import json
import os
import requests
import time
from typing import Dict

LUXIA_API_KEY = os.getenv("LUXIA_API_KEY", "")

# 가장 좋은 모델: luxia3-llm-32b-0731 (flagship, agentic AI)
MODEL_NAME = "luxia3-llm-32b-0731"
LUXIA_ENDPOINT = "https://bridge.luxiacloud.com/luxia/v1/chat"

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
    """
    Molding 설명을 분석해서 이상 여부 판단
    
    Returns:
        {
            'judgment': 'abnormal' | 'normal' | 'ambiguous',
            'reason': str,
            'status': 'success' | 'error'
        }
    """
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
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.3,
        "max_completion_tokens": 2048,
        "top_p": 1,
        "frequency_penalty": 0.1,
        "stream": False
    }
    
    try:
        response = requests.post(LUXIA_ENDPOINT, json=payload, headers=headers, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            # Luxia API 응답 구조
            if "choices" in result and len(result["choices"]) > 0:
                content = result["choices"][0]["message"]["content"].strip()
            else:
                return {
                    'judgment': 'ambiguous',
                    'reason': 'Invalid API response format',
                    'status': 'error'
                }
            
            # JSON 파싱
            try:
                import re
                clean_content = re.sub(r'```json\s*|\s*```', '', content)
                data = json.loads(clean_content)
                
                return {
                    'judgment': data.get('judgment', 'ambiguous'),
                    'reason': data.get('reason', ''),
                    'status': 'success'
                }
            
            except json.JSONDecodeError:
                print(f"    ⚠️  JSON 파싱 실패")
                return {
                    'judgment': 'ambiguous',
                    'reason': content[:200],
                    'status': 'parse_error'
                }
        else:
            return {
                'judgment': 'ambiguous',
                'reason': f'API Error: {response.status_code}',
                'status': 'error'
            }
    
    except Exception as e:
        return {
            'judgment': 'ambiguous',
            'reason': f'Error: {str(e)}',
            'status': 'error'
        }

def process_all_moldings(input_json_path: str, output_json_path: str):
    """
    모든 molding 설명을 판단
    """
    print(f"📖 입력 파일 읽는 중: {input_json_path}")
    with open(input_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    print(f"✅ 총 {len(data)}개 이미지 처리 시작")
    print(f"🤖 모델: {MODEL_NAME}")
    print("=" * 80)
    
    results = {}
    
    # 통계
    stats = {
        'abnormal': 0,
        'normal': 0,
        'ambiguous': 0,
        'error': 0
    }
    
    for idx, (img_id, item) in enumerate(data.items()):
        print(f"\n[{idx+1}/{len(data)}] ID: {img_id}")
        
        # molding 설명 가져오기
        molding_text = item.get("molding", "") or item.get("molding_description", "")
        
        if not molding_text:
            print("  ❌ Molding 설명 없음 → 비정상 (molding missing)")
            results[img_id] = {
                'judgment': 'abnormal',
                'reason': 'No molding description available - molding is missing or not visible.',
                'molding_text': ''
            }
            stats['abnormal'] += 1
            continue
        
        print(f"  📝 Molding: {molding_text[:80]}...")
        print(f"  🔍 판단 중...")
        
        # 판단 수행
        judgment_result = judge_molding(molding_text)
        
        # 결과 저장
        results[img_id] = {
            'judgment': judgment_result['judgment'],
            'reason': judgment_result['reason'],
            'molding_text': molding_text,
            'status': judgment_result['status']
        }
        
        # 출력
        judgment = judgment_result['judgment']
        if judgment == 'abnormal':
            print(f"  ❌ 판단: 이상 있음 (abnormal)")
            stats['abnormal'] += 1
        elif judgment == 'normal':
            print(f"  ✅ 판단: 정상 (normal)")
            stats['normal'] += 1
        else:
            print(f"  ⚪ 판단: 애매함 (ambiguous)")
            stats['ambiguous'] += 1
        
        print(f"  📌 근거: {judgment_result['reason'][:100]}...")
        
        # Rate limit
        time.sleep(1)
    
    # 결과 저장
    print(f"\n💾 결과 저장 중: {output_json_path}")
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # 최종 통계
    print("\n" + "=" * 80)
    print("📊 판단 완료!")
    print("=" * 80)
    print(f"✅ 총 처리: {len(data)}개")
    print(f"❌ 이상 있음 (abnormal): {stats['abnormal']}개 ({stats['abnormal']/len(data)*100:.1f}%)")
    print(f"✅ 정상 (normal): {stats['normal']}개 ({stats['normal']/len(data)*100:.1f}%)")
    print(f"⚪ 애매함 (ambiguous): {stats['ambiguous']}개 ({stats['ambiguous']/len(data)*100:.1f}%)")
    print(f"\n💾 저장 완료: {output_json_path}")

if __name__ == "__main__":
    process_all_moldings(
        input_json_path="./transistor_descriptions_1140.json",
        output_json_path="./molding_judgments_1140.json"
    )
    print("\n✅ Done!")
