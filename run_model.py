import json
import pandas as pd
from typing import List

# transi_analysis_1140.py에서 필요한 함수들 import
from transi_analysis_1140 import describe_batch

# judge_molding.py에서 필요한 함수들 import
from judge_molding import process_all_moldings

def run_full_pipeline_simple(csv_path: str) -> List[int]:
    """
    간단한 파이프라인: 함수 직접 호출
    
    Args:
        csv_path: 입력 CSV 파일 (id, img_url 컬럼 필요)
    
    Returns:
        [0, 1, 0, 1, ...] 형태의 binary list
        0 = normal (정상)
        1 = abnormal (비정상)
    """
    print("🚀" * 40)
    print("🤖 Molding 분석 파이프라인 시작")
    print("🚀" * 40 + "\n")
    
    # 임시 파일 경로
    description_json = "./temp_descriptions.json"
    judgment_json = "./temp_judgments.json"
    
    # Stage 1: Molding Description 생성
    print("=" * 80)
    print("🔍 Stage 1: Molding Description 생성 중...")
    print("=" * 80 + "\n")
    
    try:
        describe_batch(
            csv_path=csv_path,
            output_csv="./temp_descriptions.csv",
            output_json=description_json,
            preprocess=True  # 빛 반사 감소
        )
        print("\n✅ Stage 1 완료!\n")
    except Exception as e:
        print(f"❌ Stage 1 실패: {str(e)}")
        return []
    
    # Stage 2: Molding 이상 판단
    print("=" * 80)
    print("⚖️  Stage 2: Molding 이상 판단 중...")
    print("=" * 80 + "\n")
    
    try:
        process_all_moldings(
            input_json_path=description_json,
            output_json_path=judgment_json
        )
        print("\n✅ Stage 2 완료!\n")
    except Exception as e:
        print(f"❌ Stage 2 실패: {str(e)}")
        return []
    
    # Stage 3: Binary List 생성
    print("=" * 80)
    print("📊 Stage 3: Binary List 생성 중...")
    print("=" * 80 + "\n")
    
    try:
        with open(judgment_json, "r", encoding="utf-8") as f:
            judgments = json.load(f)
        
        binary_list = []
        
        # 정렬된 순서로 처리
        for img_id in sorted(judgments.keys()):
            judgment = judgments[img_id].get('judgment', 'ambiguous')
            
            if judgment == 'normal':
                binary_list.append(0)
            elif judgment == 'abnormal':
                binary_list.append(1)
            else:  # ambiguous → 안전을 위해 1 (비정상)
                binary_list.append(1)
        
        # 통계
        normal_count = binary_list.count(0)
        abnormal_count = binary_list.count(1)
        
        print(f"✅ 총 {len(binary_list)}개 이미지 처리")
        print(f"   - 정상 (0): {normal_count}개 ({normal_count/len(binary_list)*100:.1f}%)")
        print(f"   - 비정상 (1): {abnormal_count}개 ({abnormal_count/len(binary_list)*100:.1f}%)")
        
        # 리스트 출력
        print("\n📋 최종 결과 (Binary List):")
        print("=" * 80)
        print(binary_list)
        print("=" * 80)
        
        # 파일 저장
        output_txt = "./molding_binary_list.txt"
        with open(output_txt, "w", encoding="utf-8") as f:
            f.write(str(binary_list))
        
        print(f"\n💾 리스트 저장: {output_txt}")
        
        # JSON 상세 저장 (이미지 ID와 함께)
        output_detail_json = "./molding_binary_detail.json"
        detail = {
            "binary_list": binary_list,
            "details": []
        }
        
        for idx, img_id in enumerate(sorted(judgments.keys())):
            detail["details"].append({
                "index": idx,
                "id": img_id,
                "binary": binary_list[idx],
                "judgment": judgments[img_id].get('judgment', 'unknown'),
                "reason": judgments[img_id].get('reason', '')
            })
        
        with open(output_detail_json, "w", encoding="utf-8") as f:
            json.dump(detail, f, ensure_ascii=False, indent=2)
        
        print(f"💾 상세 정보 저장: {output_detail_json}")
        
        # 완료
        print("\n" + "🎉" * 40)
        print("✅ 전체 파이프라인 완료!")
        print("🎉" * 40)
        
        return binary_list
        
    except Exception as e:
        print(f"❌ Stage 3 실패: {str(e)}")
        return []

if __name__ == "__main__":
    # 실행
    binary_result = run_full_pipeline_simple(
        csv_path="./test.csv"  # 입력 CSV 파일
    )
    
    if binary_result:
        print("\n" + "=" * 80)
        print("✅ 최종 Binary List:")
        print("=" * 80)
        print(binary_result)
        print("=" * 80)
        
        # Python 리스트로 출력 (복사하기 쉽게)
        print("\n📋 복사용:")
        print(binary_result)
