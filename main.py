import os
import requests
import pandas as pd
import numpy as np
import cv2
import time
import json
import re
import base64
from io import BytesIO
from PIL import Image
from typing import Dict, List, Tuple

LUXIA_API_KEY = os.getenv("LUXIA_API_KEY", "")

MODEL_NAME = "gpt-4o"
LUXIA_ENDPOINT = f"https://bridge.luxiacloud.com/llm/openai/chat/completions/{MODEL_NAME}/create"

# 리드 삽입 여부 판단 전용 프롬프트
LEAD_INSERTION_PROMPT = """You are inspecting cropped images showing a lead (metal wire) and a hole on a circuit board.

Your task: Determine if the lead tip is inserted into the hole or not.

[INSERTED - What it looks like]
- The lead tip clearly enters and overlaps the dark circular hole
- The tip appears to go INSIDE the hole, with the lead filling or partially filling the hole opening
- There is NO visible gap or space between the lead edges and the hole circumference
- The lead appears to penetrate INTO the hole, not just touch its edge
- The lead may extend through the hole (you might see it continuing below)
- The lead and hole appear as a continuous structure

[NOT_INSERTED - What it looks like]
- The lead tip is clearly ABOVE the board surface, floating above the hole
- There is a CLEAR VISIBLE GAP or space between the lead tip and the hole opening
- The lead is positioned next to or beside the hole without entering it
- The lead does NOT penetrate into the hole - it appears to hover or rest on the surface
- You can see the hole opening clearly without the lead blocking it
- **CRITICAL: The lead may TOUCH or BRUSH AGAINST the edge/rim of the hole, but if it does NOT actually ENTER and PENETRATE into the hole interior, it is NOT_INSERTED**
- **If the lead only touches the hole's edge/circumference without going INSIDE, it is NOT_INSERTED (even if there's no visible gap)**

[UNCERTAIN - When to use]
- ONLY use this if you truly cannot determine from the images
- If you can see a clear gap, it is NOT_INSERTED, not UNCERTAIN
- If the lead clearly enters the hole, it is INSERTED, not UNCERTAIN

CRITICAL RULES (Be balanced and precise):
1. GAP DETECTION is the MOST IMPORTANT indicator:
   - If you see a CLEAR, VISIBLE GAP or space between the lead tip and the hole opening → NOT_INSERTED
   - If there is NO visible gap and the lead tip overlaps/enters the hole → INSERTED
   - A gap means the lead is floating above the board, not inserted

2. Look at the RELATIONSHIP between the lead and the hole opening:
   - INSERTED: Lead tip clearly enters and fills the hole opening, NO gap visible, lead appears to penetrate INTO the hole, lead goes INSIDE the hole interior
   - NOT_INSERTED: Lead tip is clearly ABOVE the hole with a visible gap, lead hovers above the surface, does NOT penetrate
   - NOT_INSERTED (EDGE CASE): Lead touches or brushes against the hole's edge/rim but does NOT actually enter the hole interior - the lead is on the surface around the hole, not inside it

3. Depth perception:
   - INSERTED: Lead appears to go INTO the hole (you see depth, lead continues into the hole, no gap at the entry point)
   - NOT_INSERTED: Lead appears to be ON the surface (no depth, lead stops at the surface, gap visible)

4. Visual continuity:
   - INSERTED: Lead and hole appear as one continuous structure, lead fills the hole opening, no separation visible
   - NOT_INSERTED: Lead and hole appear as separate objects with clear separation/gap

5. DECISION CRITERIA (use these in order):
   a) If you see a CLEAR GAP → NOT_INSERTED (most reliable)
   b) If the lead only TOUCHES/BRUSHES the hole edge but does NOT enter the interior → NOT_INSERTED (even if no gap visible)
   c) If NO gap AND lead tip clearly PENETRATES and ENTERS the hole interior → INSERTED
   d) If uncertain about gap or penetration depth → Look for depth cues and visual continuity
   e) Only use UNCERTAIN if you truly cannot determine

6. EDGE CASE - "Touching but not inserted":
   - If the lead appears to touch the hole's edge/circumference but the hole interior remains dark/unobstructed → NOT_INSERTED
   - If you see a bright crescent or lighter area at the hole edge where the lead touches, but the lead does not go INTO the hole → NOT_INSERTED
   - The lead must actually PENETRATE INTO the hole interior, not just touch the rim

7. IMPORTANT: Do NOT default to INSERTED. Only say INSERTED if you can clearly see the lead ENTERING and PENETRATING INTO the hole interior with NO gap. Touching the edge is NOT enough.

Respond in JSON format:
{
  "insertion_status": "INSERTED" | "NOT_INSERTED" | "UNCERTAIN",
  "reason": "Detailed explanation: (1) What you see regarding gap/overlap, (2) Whether lead appears inside or above the hole, (3) Any depth cues you observe"
}

Output ONLY valid JSON, no additional text.
"""

# 전처리된 이미지를 저장할 폴더 생성
PROCESSED_DIR = "./test_img"
if not os.path.exists(PROCESSED_DIR):
    os.makedirs(PROCESSED_DIR)

def preprocess_crop_gentle(crop_bgr: np.ndarray) -> np.ndarray:
    """
    크롭 이미지에 완화된 전처리 적용 (세부 정보 보존)
    이진화 대신 CLAHE 사용하여 정보 손실 최소화
    
    Returns:
        전처리된 BGR 이미지 (3채널, 컬러 정보 유지)
    """
    # 1. 그레이스케일 변환
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    
    # 2. 가우시안 블러 (매우 약하게 - 노이즈만 제거)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # 3. CLAHE (적응형 히스토그램 균등화) - 이진화 대신 사용
    # 이진화는 정보 손실이 크므로, CLAHE로 대비만 개선하여 세부 정보 보존
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(blurred)
    
    # 4. 약한 샤프닝 (선명도 개선, 정보 손실 최소화)
    gaussian = cv2.GaussianBlur(enhanced, (0, 0), 1.0)
    sharpened = cv2.addWeighted(enhanced, 1.2, gaussian, -0.2, 0)
    
    # BGR로 변환 (3채널로, 컬러 정보 유지)
    result_bgr = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)
    
    return result_bgr

def preprocess_and_make_lead_crops(image_content: bytes, img_id: str):
    """
    1) 전체 이미지: 홀(검은 원)을 너무 밝히지 않는 '약한 보정'
    2) 리드 tip과 가장 가까운 홀을 찾아서 그 홀 중심으로 크롭 (각 리드별로 딱 하나의 홀만 포함)
    3) 홀 후보에 동그라미 그려서 시각화 저장
    Returns:
      full_rgb: PIL.Image
      crops_rgb: {"left": PIL, "center": PIL, "right": PIL}
      full_bgr: np.ndarray (CV용)
    """
    nparr = np.frombuffer(image_content, np.uint8)
    bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # --- (A) 홀 대비 유지: gamma를 과하게 낮추지 않기 (홀을 밝히면 오히려 헷갈림) ---
    gamma = 1.05
    lut = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)]).astype("uint8")
    bgr_gamma = cv2.LUT(bgr, lut)

    # --- (B) CLAHE는 약하게 (홀은 검게 남기고, 구리/리드 대비만 살리기) ---
    lab = cv2.cvtColor(bgr_gamma, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    bgr_enh = cv2.cvtColor(cv2.merge((l2, a, b)), cv2.COLOR_LAB2BGR)

    # --- (C) 샤프닝은 약하게 ---
    g = cv2.GaussianBlur(bgr_enh, (0, 0), 1.2)
    bgr_final = cv2.addWeighted(bgr_enh, 1.2, g, -0.2, 0)

    # ✅ 에지(Canny) 오버레이는 LLM 입력에 쓰지 않음 (오판 유발 가능)
    save_path = os.path.join(PROCESSED_DIR, f"proc_{img_id}.png")
    cv2.imwrite(save_path, bgr_final)

    h, w = bgr_final.shape[:2]
    gray = cv2.cvtColor(bgr_final, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.medianBlur(gray, 5)

    # --- 홀 후보 찾기 (전체 이미지에서) ---
    circles = cv2.HoughCircles(
        gray_blur,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(10, int(min(h, w) * 0.06)),
        param1=120,
        param2=22,
        minRadius=9,
        maxRadius=11,
    )

    # 홀 후보에 동그라미 그리기 (시각화용)
    debug_img = bgr_final.copy()
    all_holes = []
    if circles is not None:
        circles = np.round(circles[0, :]).astype(int)
        for (x, y, r) in circles:
            # 전체 이미지의 hole 모두 포함 (상단 포함)
            all_holes.append((x, y, r))
            # 동그라미 그리기 (초록색)
            cv2.circle(debug_img, (x, y), r, (0, 255, 0), 2)
            # 중심점 표시
            cv2.circle(debug_img, (x, y), 3, (0, 255, 0), -1)

    # 홀 후보 시각화 이미지 저장
    cv2.imwrite(os.path.join(PROCESSED_DIR, f"holes_debug_{img_id}.png"), debug_img)

    # --- 리드 tip 찾기 및 가장 가까운 홀 중심으로 크롭 ---
    y1 = int(h * 0.45)
    y2 = h
    roi = bgr_final[y1:y2, :]
    roi_gray = gray[y1:y2, :]

    crops = {}
    thirds = [(0, w//3), (w//3, 2*w//3), (2*w//3, w)]
    names = ["left", "center", "right"]

    for name, (x1, x2) in zip(names, thirds):
        sub_roi = roi[:, x1:x2]
        sub_gray = roi_gray[:, x1:x2]
        sh, sw = sub_roi.shape[:2]

        # 리드 tip 찾기: 밝은 영역에서 가장 아래쪽 점
        # 하단 70% 영역에서 밝은 픽셀 찾기
        tip_search_y0 = int(sh * 0.3)
        tip_search_region = sub_gray[tip_search_y0:, :]
        
        # 밝은 픽셀 (리드) 찾기
        bright_mask = (tip_search_region > 140) & (tip_search_region < 200)
        bright_y, bright_x = np.where(bright_mask)
        
        if len(bright_y) > 0:
            # 가장 아래쪽 밝은 픽셀을 tip으로
            tip_y_local = tip_search_y0 + int(np.max(bright_y))
            tip_x_local = int(np.mean(bright_x[bright_y == np.max(bright_y)]))
            # 전체 이미지 좌표로 변환
            tip_x_full = x1 + tip_x_local
            tip_y_full = y1 + tip_y_local
        else:
            # tip을 찾지 못한 경우, 하단 중앙을 tip으로 가정
            tip_x_full = x1 + sw // 2
            tip_y_full = y1 + int(sh * 0.9)

        # 이 리드 tip과 가장 가까운 홀 찾기
        best_hole = None
        min_dist = float('inf')
        for (hx, hy, hr) in all_holes:
            # 해당 리드 영역(x1~x2)에 있는 홀만 고려
            if x1 <= hx <= x2:
                dist = np.sqrt((tip_x_full - hx)**2 + (tip_y_full - hy)**2)
                if dist < min_dist:
                    min_dist = dist
                    best_hole = (hx, hy, hr)

        # 크롭: 홀 중심 기준으로 적절한 크기로
        if best_hole is not None:
            hx, hy, hr = best_hole
            # 홀 중심 기준으로 크롭 (홀 반지름의 3~4배 크기)
            crop_size = int(hr * 4)
            crop_x0 = max(0, hx - crop_size)
            crop_x1 = min(w, hx + crop_size)
            crop_y0 = max(y1, hy - crop_size)
            crop_y1 = min(h, hy + crop_size)
            
            crop_region = bgr_final[crop_y0:crop_y1, crop_x0:crop_x1]
        else:
            # 홀을 찾지 못한 경우, 기존 방식으로 fallback
            sub2 = sub_roi[int(sh*0.30):sh, :]
            crop_region = sub2

        # 확대 (LLM이 tip/홀 관계를 보기 쉬움)
        scale = 3.0
        # 빈 이미지 체크
        if crop_region.size == 0 or len(crop_region.shape) < 2 or crop_region.shape[0] == 0 or crop_region.shape[1] == 0:
            # 빈 이미지인 경우 기본 크기로 생성
            crop_region = np.ones((50, 50, 3), dtype=np.uint8) * 128  # 회색 배경
        
        new_w = max(1, int(crop_region.shape[1] * scale))
        new_h = max(1, int(crop_region.shape[0] * scale))
        up = cv2.resize(crop_region, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

        # 완화된 전처리 적용 (세부 정보 보존)
        preprocessed = preprocess_crop_gentle(up)
        
        # 원본과 전처리된 이미지 모두 저장 (디버깅용)
        debug_crop = up.copy()
        if best_hole is not None:
            # 크롭된 이미지에서 홀 중심 위치 계산
            hx_crop = (hx - crop_x0) * scale
            hy_crop = (hy - crop_y0) * scale
            hr_crop = hr * scale
            cv2.circle(debug_crop, (int(hx_crop), int(hy_crop)), int(hr_crop), (0, 255, 0), 2)
            cv2.circle(debug_crop, (int(hx_crop), int(hy_crop)), 3, (0, 255, 0), -1)
        cv2.imwrite(os.path.join(PROCESSED_DIR, f"crop_{img_id}_{name}_original.png"), debug_crop)
        cv2.imwrite(os.path.join(PROCESSED_DIR, f"crop_{img_id}_{name}_preprocessed.png"), preprocessed)

        # 원본과 전처리된 이미지를 모두 저장 (LLM에 둘 다 보내기 위해)
        crops[name] = {
            "original": Image.fromarray(cv2.cvtColor(up, cv2.COLOR_BGR2RGB)),
            "preprocessed": Image.fromarray(cv2.cvtColor(preprocessed, cv2.COLOR_BGR2RGB))
        }

    full_rgb = Image.fromarray(cv2.cvtColor(bgr_final, cv2.COLOR_BGR2RGB))
    return full_rgb, crops, bgr_final

def cv_check_not_inserted_right(bgr_final: np.ndarray) -> str:
    """
    오른쪽 리드가 '확실히' 홀에 안 들어간 케이스를 잡기 위한 보수적 체크.
    Return: "NOT_INSERTED" / "INSERTED_OR_UNKNOWN"
    """
    h, w = bgr_final.shape[:2]

    # 하단 ROI + 오른쪽 1/3만 보기
    y1 = int(h * 0.45)
    roi = bgr_final[y1:h, int(2*w/3):w]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)

    # 1) 홀(원) 후보 찾기 (HoughCircles)
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(15, roi.shape[1]//3),
        param1=120,
        param2=22,          # 낮추면 많이 잡힘 / 높이면 덜 잡힘
        minRadius=6,
        maxRadius=30
    )

    if circles is None:
        return "INSERTED_OR_UNKNOWN"

    circles = np.round(circles[0]).astype(int)

    # 2) 리드 tip 후보: 에지에서 가장 아래쪽(큰 contour) 포인트를 tip으로 근사
    edges = cv2.Canny(gray, 50, 150)
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return "INSERTED_OR_UNKNOWN"

    # 오른쪽 리드는 roi 중앙 근처 x에 있을 확률이 높으니, 그 근처 큰 contour 선호
    rh, rw = roi.shape[:2]
    target_x = int(rw * 0.55)

    best = None
    best_score = -1
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 20:
            continue
        xs = c[:, 0, 0]
        ys = c[:, 0, 1]
        # 아래쪽에 길게 존재하는 contour가 리드일 확률 높음
        score = area + (ys.max() * 0.5) - abs(int(xs.mean()) - target_x) * 2.0
        if score > best_score:
            best_score = score
            best = c

    if best is None:
        return "INSERTED_OR_UNKNOWN"

    # tip = contour의 최하단 점
    pts = best[:, 0, :]
    tip = pts[np.argmax(pts[:, 1])]
    tip_x, tip_y = int(tip[0]), int(tip[1])

    # 3) tip이 원(홀) 내부로 충분히 들어갔는지 검사
    #    + tip 주변 픽셀 밝기가 '홀(어두움)'인지도 같이 확인 (중요)
    inserted_votes = 0
    for (cx, cy, r) in circles:
        dist = np.sqrt((tip_x - cx) ** 2 + (tip_y - cy) ** 2)
        if dist <= r * 0.85:  # 원 내부 깊숙이
            # 주변 밝기 확인: 홀 내부는 더 어두움
            y0 = np.clip(tip_y, 0, rh - 1)
            x0 = np.clip(tip_x, 0, rw - 1)
            patch = gray[max(0, y0-2):min(rh, y0+3), max(0, x0-2):min(rw, x0+3)]
            if patch.size > 0 and patch.mean() < 90:  # 어두운 편이면 홀 내부 가능성↑
                inserted_votes += 1

    # inserted_votes가 0이면: 원 내부/어두움 근거가 없음 → NOT_INSERTED 쪽으로 강하게
    if inserted_votes == 0:
        return "NOT_INSERTED"

    return "INSERTED_OR_UNKNOWN"


def pil_to_b64(pil_img: Image.Image) -> str:
    buf = BytesIO()
    pil_img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

def extract_image_features(img: np.ndarray) -> np.ndarray:
    """
    이미지에서 특징 벡터 추출 (KNN용)
    
    특징:
    1. Histogram (RGB 각 채널)
    2. Edge density (Canny edge)
    3. Texture features (LBP-like)
    4. Hole-lead relationship features (중심부 밝기, 가장자리 밝기 등)
    
    Args:
        img: BGR 이미지 (numpy array)
    
    Returns:
        특징 벡터 (1D numpy array)
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    features = []
    
    # 1. Histogram features (RGB 각 채널)
    for channel in range(3):
        hist = cv2.calcHist([img], [channel], None, [32], [0, 256])
        features.extend(hist.flatten())
    
    # 2. Edge density
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / (h * w)
    features.append(edge_density)
    
    # 3. Texture features (Gradient magnitude)
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    features.append(np.mean(gradient_magnitude))
    features.append(np.std(gradient_magnitude))
    
    # 4. Hole-lead relationship features
    # 중심부와 가장자리 밝기 차이 (hole이 중심에 있다고 가정)
    center_region = gray[h//4:3*h//4, w//4:3*w//4]
    edge_region = np.concatenate([
        gray[0:h//4, :].flatten(),
        gray[3*h//4:h, :].flatten(),
        gray[:, 0:w//4].flatten(),
        gray[:, 3*w//4:w].flatten()
    ])
    
    if center_region.size > 0 and edge_region.size > 0:
        center_brightness = np.mean(center_region)
        edge_brightness = np.mean(edge_region)
        features.append(center_brightness)
        features.append(edge_brightness)
        features.append(center_brightness - edge_brightness)
    else:
        features.extend([0, 0, 0])
    
    # 5. Local binary pattern (간단한 버전)
    # 중심부의 밝기 분포
    center_hist = cv2.calcHist([center_region], [0], None, [16], [0, 256])
    features.extend(center_hist.flatten())
    
    return np.array(features, dtype=np.float32)

def analyze_with_knn_similarity(crop_bgr: np.ndarray) -> Dict:
    """
    Reference 이미지들과 KNN 유사도 측정으로 삽입 여부 판단
    
    Args:
        crop_bgr: 크롭된 BGR 이미지
    
    Returns:
        {
            "knn_status": "INSERTED" | "NOT_INSERTED" | "UNCERTAIN",
            "confidence": float,
            "details": {
                "inserted_similarities": List[float],
                "not_inserted_similarities": List[float],
                "avg_inserted_sim": float,
                "avg_not_inserted_sim": float
            }
        }
    """
    try:
        # Reference 이미지 로드
        ref_inserted_paths = [
            "./ref/INSERTED_01.png",
            "./ref/INSERTED_02.png"
        ]
        ref_not_inserted_paths = [
            "./ref/NOT_INSERTED_01.png",
            "./ref/NOT_INSERTED_02.png"
        ]
        
        # 입력 이미지 특징 추출
        input_features = extract_image_features(crop_bgr)
        
        # INSERTED reference 특징 추출
        inserted_features = []
        for ref_path in ref_inserted_paths:
            if os.path.exists(ref_path):
                ref_img = cv2.imread(ref_path)
                if ref_img is not None and ref_img.size > 0 and len(ref_img.shape) >= 2:
                    # 크기 정규화 (입력 이미지와 비슷한 크기로)
                    h, w = crop_bgr.shape[:2]
                    if h > 0 and w > 0 and ref_img.shape[0] > 0 and ref_img.shape[1] > 0:
                        ref_resized = cv2.resize(ref_img, (w, h))
                        ref_features = extract_image_features(ref_resized)
                        inserted_features.append(ref_features)
        
        # NOT_INSERTED reference 특징 추출
        not_inserted_features = []
        for ref_path in ref_not_inserted_paths:
            if os.path.exists(ref_path):
                ref_img = cv2.imread(ref_path)
                if ref_img is not None and ref_img.size > 0 and len(ref_img.shape) >= 2:
                    # 크기 정규화
                    h, w = crop_bgr.shape[:2]
                    if h > 0 and w > 0 and ref_img.shape[0] > 0 and ref_img.shape[1] > 0:
                        ref_resized = cv2.resize(ref_img, (w, h))
                        ref_features = extract_image_features(ref_resized)
                        not_inserted_features.append(ref_features)
        
        if len(inserted_features) == 0 or len(not_inserted_features) == 0:
            return {
                "knn_status": "UNCERTAIN",
                "confidence": 0.0,
                "details": {"error": "Reference images not found"}
            }
        
        # 특징 벡터 정규화 (스케일 차이 보정) - numpy만 사용
        all_features = np.vstack([input_features.reshape(1, -1)] + 
                                [f.reshape(1, -1) for f in inserted_features] +
                                [f.reshape(1, -1) for f in not_inserted_features])
        
        # StandardScaler와 동일한 작업 (mean=0, std=1로 정규화)
        mean = np.mean(all_features, axis=0)
        std = np.std(all_features, axis=0)
        std = np.where(std == 0, 1, std)  # std가 0인 경우 1로 설정 (division by zero 방지)
        all_features_scaled = (all_features - mean) / std
        
        input_scaled = all_features_scaled[0]
        inserted_scaled = all_features_scaled[1:1+len(inserted_features)]
        not_inserted_scaled = all_features_scaled[1+len(inserted_features):]
        
        # 유사도 계산 (Euclidean distance의 역수, 또는 cosine similarity)
        # Euclidean distance 사용
        inserted_distances = [np.linalg.norm(input_scaled - ref) for ref in inserted_scaled]
        not_inserted_distances = [np.linalg.norm(input_scaled - ref) for ref in not_inserted_scaled]
        
        # 거리를 유사도로 변환 (거리가 가까울수록 유사도 높음)
        # 거리 -> 유사도: 1 / (1 + distance)
        # numpy float32를 Python float로 변환하여 JSON 직렬화 문제 방지
        inserted_similarities = [float(1.0 / (1.0 + d)) for d in inserted_distances]
        not_inserted_similarities = [float(1.0 / (1.0 + d)) for d in not_inserted_distances]
        
        avg_inserted_sim = float(np.mean(inserted_similarities)) if inserted_similarities else 0.0
        avg_not_inserted_sim = float(np.mean(not_inserted_similarities)) if not_inserted_similarities else 0.0
        
        # 판단: INSERTED reference와 더 유사하면 INSERTED
        # 차이가 명확해야 confidence 높음
        sim_diff = float(avg_inserted_sim - avg_not_inserted_sim)
        
        if sim_diff > 0.1:  # INSERTED가 0.1 이상 더 유사하면
            knn_status = "INSERTED"
            confidence = float(min(0.95, 0.6 + abs(sim_diff) * 2.0))
        elif sim_diff < -0.1:  # NOT_INSERTED가 0.1 이상 더 유사하면
            knn_status = "NOT_INSERTED"
            confidence = float(min(0.95, 0.6 + abs(sim_diff) * 2.0))
        else:  # 차이가 작으면 불확실
            knn_status = "UNCERTAIN"
            confidence = 0.3
        
        return {
            "knn_status": knn_status,
            "confidence": confidence,
            "details": {
                "inserted_similarities": inserted_similarities,
                "not_inserted_similarities": not_inserted_similarities,
                "avg_inserted_sim": avg_inserted_sim,
                "avg_not_inserted_sim": avg_not_inserted_sim,
                "similarity_diff": sim_diff
            }
        }
    
    except Exception as e:
        return {
            "knn_status": "UNCERTAIN",
            "confidence": 0.0,
            "details": {"error": str(e)}
        }

def analyze_lead_insertion_cv_geometric(crop_bgr: np.ndarray) -> Dict:
    """
    전문적인 CV 기하학적 분석으로 리드 삽입 여부 판단
    
    방법론:
    1. Hole 중심 및 반지름 검출 (HoughCircles)
    2. Lead contour 추출 및 tip 위치 계산
    3. 기하학적 관계 분석 (거리, 겹침)
    4. Depth cue 분석 (그림자, 명암 변화)
    5. Contour overlap 분석
    
    Returns:
        {
            "cv_status": "INSERTED" | "NOT_INSERTED" | "UNCERTAIN",
            "confidence": float (0.0-1.0),
            "details": {
                "hole_center": (x, y),
                "hole_radius": float,
                "lead_tip": (x, y),
                "distance_from_center": float,
                "tip_inside_hole": bool,
                "overlap_ratio": float
            }
        }
    """
    try:
        h, w = crop_bgr.shape[:2]
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        
        # 1. Hole 검출 (HoughCircles)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=max(10, int(min(h, w) * 0.2)),
            param1=100,
            param2=20,
            minRadius=max(5, int(min(h, w) * 0.05)),
            maxRadius=int(min(h, w) * 0.3)
        )
        
        if circles is None:
            return {
                "cv_status": "UNCERTAIN",
                "confidence": 0.0,
                "details": {"error": "No hole detected"}
            }
        
        circles = np.round(circles[0, :]).astype(int)
        # 가장 큰 원을 hole로 선택 (일반적으로 가장 명확함)
        largest_circle = max(circles, key=lambda c: c[2])
        hx, hy, hr = largest_circle
        
        # 2. Lead contour 추출 (개선된 방법)
        # 리드는 밝은 영역이므로, 적응형 threshold 사용
        # 먼저 밝은 영역 찾기 (리드 후보)
        _, thresh_bright = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # OTSU는 어두운 부분을 흰색으로 만들므로, 밝은 부분(리드)을 찾으려면 반전
        thresh_bright = cv2.bitwise_not(thresh_bright)
        
        # 또는 직접 밝은 픽셀 찾기 (더 안정적)
        # 리드는 보통 중간~밝은 픽셀 (100-200 범위)
        bright_mask = (gray > 100) & (gray < 220)
        thresh_bright = bright_mask.astype(np.uint8) * 255
        
        # 모폴로지로 노이즈 제거
        kernel = np.ones((3, 3), np.uint8)
        cleaned = cv2.morphologyEx(thresh_bright, cv2.MORPH_OPEN, kernel, iterations=1)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Contour 찾기
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return {
                "cv_status": "UNCERTAIN",
                "confidence": 0.0,
                "details": {"error": "No lead contour detected"}
            }
        
        # Hole 근처에 있는 contour를 lead로 선택 (개선)
        # 1. Hole 중심에 가장 가까운 큰 contour 찾기
        best_contour = None
        best_score = -1
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 50:  # 너무 작은 contour는 제외
                continue
            # Contour의 중심점
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                # Hole 중심과의 거리
                dist_to_hole = np.sqrt((cx - hx)**2 + (cy - hy)**2)
                # Score: 면적이 크고, hole에 가까울수록 높은 점수
                score = area / (1 + dist_to_hole * 0.1)
                if score > best_score:
                    best_score = score
                    best_contour = contour
        
        if best_contour is None:
            # Fallback: 가장 큰 contour
            best_contour = max(contours, key=cv2.contourArea)
        
        lead_contour = best_contour
        lead_points = lead_contour.reshape(-1, 2)
        
        # Lead tip 찾기 (개선된 방법)
        # 방법 1: Hole 중심에 가장 가까운 lead 점
        tip_candidates = []
        for point in lead_points:
            dist_to_hole = np.sqrt((point[0] - hx)**2 + (point[1] - hy)**2)
            # Hole 반지름의 1.5배 이내에 있는 점들만 후보
            if dist_to_hole < hr * 1.5:
                tip_candidates.append((point, dist_to_hole))
        
        if tip_candidates:
            # Hole에 가장 가까운 점을 tip으로
            tip_point, _ = min(tip_candidates, key=lambda x: x[1])
            tip_x, tip_y = int(tip_point[0]), int(tip_point[1])
        else:
            # Fallback: 하단 20% 영역의 중심점
            bottom_threshold = np.percentile(lead_points[:, 1], 80)
            bottom_points = lead_points[lead_points[:, 1] > bottom_threshold]
            if len(bottom_points) > 0:
                tip_x = int(np.mean(bottom_points[:, 0]))
                tip_y = int(np.max(bottom_points[:, 1]))
            else:
                # 최하단 점
                tip_idx = np.argmax(lead_points[:, 1])
                tip_x, tip_y = int(lead_points[tip_idx, 0]), int(lead_points[tip_idx, 1])
        
        # 3. 기하학적 분석
        # Hole 중심과 lead tip의 거리
        dist_from_center = float(np.sqrt((tip_x - hx)**2 + (tip_y - hy)**2))
        
        # Tip이 hole 내부에 있는지 (더 관대하게 - 반지름의 100% 이내)
        tip_inside_hole = dist_from_center < hr * 1.0
        
        # 4. Contour overlap 분석
        # Hole 영역 마스크 생성
        hole_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(hole_mask, (hx, hy), int(hr * 0.9), 255, -1)
        
        # Lead contour 마스크 생성
        lead_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(lead_mask, [lead_contour], -1, 255, -1)
        
        # Overlap 계산
        overlap = cv2.bitwise_and(hole_mask, lead_mask)
        overlap_pixels = np.sum(overlap > 0)
        hole_pixels = np.sum(hole_mask > 0)
        overlap_ratio = float(overlap_pixels / hole_pixels if hole_pixels > 0 else 0.0)
        
        # 5. Gap Detection (리드와 hole 사이의 간격 검출)
        # 리드 tip 주변과 hole 가장자리 사이의 밝기 차이로 gap 판단
        gap_detected = False
        gap_score = 0.0
        
        if dist_from_center > hr * 0.5:  # Tip이 hole 중심에서 떨어져 있으면
            # Tip 주변 영역의 밝기
            tip_region_size = max(5, int(hr * 0.3))
            tip_roi = gray[max(0, tip_y-tip_region_size):min(h, tip_y+tip_region_size),
                          max(0, tip_x-tip_region_size):min(w, tip_x+tip_region_size)]
            
            # Hole 가장자리 영역의 밝기
            angle_to_tip = np.arctan2(tip_y - hy, tip_x - hx)
            edge_x = int(hx + hr * np.cos(angle_to_tip))
            edge_y = int(hy + hr * np.sin(angle_to_tip))
            edge_roi = gray[max(0, edge_y-tip_region_size):min(h, edge_y+tip_region_size),
                           max(0, edge_x-tip_region_size):min(w, edge_x+tip_region_size)]
            
            if tip_roi.size > 0 and edge_roi.size > 0:
                tip_brightness = float(np.mean(tip_roi))
                edge_brightness = float(np.mean(edge_roi))
                # Tip이 밝고 edge가 어두우면 gap이 있을 가능성
                if tip_brightness > edge_brightness + 20:  # 명확한 밝기 차이
                    gap_detected = True
                    gap_score = float((tip_brightness - edge_brightness) / 255.0)
        
        # 6. Depth cue 분석 (hole 내부의 밝기 변화)
        # Hole 내부 영역의 밝기 분석
        hole_roi = gray[max(0, hy-hr):min(h, hy+hr), max(0, hx-hr):min(w, hx+hr)]
        if hole_roi.size > 0:
            # Hole 중심부와 가장자리의 밝기 차이
            center_patch = hole_roi[hole_roi.shape[0]//3:2*hole_roi.shape[0]//3,
                                    hole_roi.shape[1]//3:2*hole_roi.shape[1]//3]
            edge_patch = hole_roi.copy()
            cv2.circle(edge_patch, (edge_patch.shape[1]//2, edge_patch.shape[0]//2),
                      min(edge_patch.shape)//3, 0, -1)  # 중심부 마스킹
            
            center_brightness = float(np.mean(center_patch)) if center_patch.size > 0 else 0.0
            edge_brightness = float(np.mean(edge_patch[edge_patch > 0])) if np.sum(edge_patch > 0) > 0 else 0.0
            
            # 리드가 들어가면 중심부가 더 밝아짐
            depth_cue = float(center_brightness - edge_brightness)
        else:
            depth_cue = 0.0
        
        # 7. 종합 판단 (엄격하고 균형잡힌 로직)
        votes_inserted = 0
        votes_not_inserted = 0
        
        # Gap Detection (가장 중요한 NOT_INSERTED 지표)
        if gap_detected and gap_score > 0.1:  # 명확한 gap이 감지되면
            votes_not_inserted += 5  # 매우 높은 가중치로 NOT_INSERTED
        elif gap_detected:
            votes_not_inserted += 2
        
        # Overlap 판단 (중요 지표)
        if overlap_ratio > 0.35:  # 35% 이상 겹치면 (더 엄격한 기준)
            votes_inserted += 3
        elif overlap_ratio > 0.25:  # 25-35% 사이면
            votes_inserted += 2
        elif overlap_ratio > 0.15:  # 15-25% 사이면
            votes_inserted += 1
        elif overlap_ratio < 0.1:  # 10% 미만이면
            votes_not_inserted += 2
        
        # 기하학적 판단 (거리 기반)
        if dist_from_center < hr * 0.8:  # 반지름의 80% 이내면 확실히 삽입
            votes_inserted += 3
        elif dist_from_center < hr * 1.0:  # 반지름의 100% 이내면
            votes_inserted += 2
        elif dist_from_center < hr * 1.2:  # 반지름의 120% 이내면
            votes_inserted += 1
        elif dist_from_center > hr * 1.5:  # 반지름의 150% 이상이면
            votes_not_inserted += 3
        
        # Tip inside 판단
        if tip_inside_hole:
            votes_inserted += 2
        else:
            votes_not_inserted += 1
        
        # Depth cue 판단 (보조 지표)
        if depth_cue > 15:  # 중심부가 더 밝으면 (리드가 밝은 색이고 삽입됨)
            votes_inserted += 1
        elif depth_cue < -20:  # 중심부가 훨씬 더 어두우면 (빈 구멍)
            votes_not_inserted += 1
        
        # 최종 판단 (엄격한 기준 - 여러 지표가 일치해야 INSERTED)
        vote_diff = votes_inserted - votes_not_inserted
        
        # INSERTED 판단: gap이 없고, overlap이 높고, 거리가 가까워야 함
        if gap_detected:
            # Gap이 있으면 NOT_INSERTED로 강하게 판단
            cv_status = "NOT_INSERTED"
            confidence = float(min(0.95, 0.7 + gap_score * 0.2))
        elif overlap_ratio > 0.3 and dist_from_center < hr * 1.0 and tip_inside_hole:
            # 모든 조건이 만족되면 INSERTED
            cv_status = "INSERTED"
            confidence = float(min(0.95, 0.7 + vote_diff * 0.1))
        elif vote_diff >= 3:  # INSERTED가 3점 이상 앞서면 (더 엄격)
            cv_status = "INSERTED"
            confidence = float(min(0.95, 0.6 + vote_diff * 0.1))
        elif vote_diff <= -3:  # NOT_INSERTED가 3점 이상 앞서면
            cv_status = "NOT_INSERTED"
            confidence = float(min(0.95, 0.6 + abs(vote_diff) * 0.1))
        else:
            cv_status = "UNCERTAIN"
            confidence = 0.3  # 불확실하면 낮은 confidence로 LLM에 위임
        
        return {
            "cv_status": cv_status,
            "confidence": float(confidence),
            "details": {
                "hole_center": (int(hx), int(hy)),
                "hole_radius": float(hr),
                "lead_tip": (int(tip_x), int(tip_y)),
                "distance_from_center": float(dist_from_center),
                "tip_inside_hole": bool(tip_inside_hole),
                "overlap_ratio": float(overlap_ratio),
                "depth_cue": float(depth_cue),
                "gap_detected": bool(gap_detected),
                "gap_score": float(gap_score),
                "votes": {"inserted": int(votes_inserted), "not_inserted": int(votes_not_inserted)}
            }
        }
    
    except Exception as e:
        return {
            "cv_status": "UNCERTAIN",
            "confidence": 0.0,
            "details": {"error": str(e)}
        }

def check_lead_insertion(crop_images: Dict[str, Image.Image], lead_name: str) -> Dict:
    """
    단일 크롭 이미지(원본 + 전처리)에서 리드 삽입 여부 판단
    CV 기하학적 분석 + LLM 조합 사용
    
    Args:
        crop_images: {"original": PIL.Image, "preprocessed": PIL.Image}
    
    Returns:
        {
            "insertion_status": "INSERTED" | "NOT_INSERTED" | "UNCERTAIN",
            "reason": str,
            "status": "success" | "error",
            "cv_analysis": Dict
        }
    """
    # 1. CV 기하학적 분석 먼저 수행
    original_np = np.array(crop_images["original"])
    original_bgr = cv2.cvtColor(original_np, cv2.COLOR_RGB2BGR)
    cv_result = analyze_lead_insertion_cv_geometric(original_bgr)
    
    # CV 결과 디버깅 출력
    print(f"    🔍 CV Analysis [{lead_name}]: {cv_result['cv_status']} (conf: {cv_result['confidence']:.2f})")
    if 'details' in cv_result and 'error' not in cv_result['details']:
        details = cv_result['details']
        print(f"       - Gap detected: {details.get('gap_detected', False)} (score: {details.get('gap_score', 0):.2f}), "
              f"Tip inside: {details.get('tip_inside_hole', False)}, "
              f"Overlap: {details.get('overlap_ratio', 0):.2f}, "
              f"Depth cue: {details.get('depth_cue', 0):.1f}, "
              f"Votes: I={details.get('votes', {}).get('inserted', 0)}, NI={details.get('votes', {}).get('not_inserted', 0)}")
    
    # CV 결과 분석
    details = cv_result.get('details', {})
    overlap_ratio = details.get('overlap_ratio', 0)
    dist_from_center = details.get('distance_from_center', 0)
    hole_radius = details.get('hole_radius', 0)
    gap_detected = details.get('gap_detected', False)
    gap_score = details.get('gap_score', 0.0)
    
    # Gap이 감지되면 NOT_INSERTED로 확실히 판단 (가장 중요한 지표)
    if gap_detected and gap_score > 0.1:
        distance_ratio = (dist_from_center / hole_radius) if hole_radius > 0 else 0.0
        return {
            "insertion_status": "NOT_INSERTED",
            "reason": f"CV gap detection (confidence: {cv_result['confidence']:.2f}). "
                     f"Gap detected with score: {gap_score:.2f}. "
                     f"Overlap: {overlap_ratio:.2f}, Distance ratio: {distance_ratio:.2f}",
            "status": "success",
            "cv_analysis": cv_result
        }
    
    # 애매한 케이스 감지: overlap이 작거나(0.1~0.3), gap이 명확하지 않거나, 거리가 애매한 경우
    # 이런 경우는 LLM으로 넘겨서 정확히 판단하도록 함
    distance_ratio = (dist_from_center / hole_radius) if hole_radius > 0 else 0.0
    is_ambiguous_case = (
        (0.1 <= overlap_ratio <= 0.3) or  # overlap이 작은 경우 (스치기만 한 가능성)
        (gap_score > 0 and gap_score <= 0.1) or  # gap이 애매한 경우
        (0.9 <= distance_ratio <= 1.2) or  # 거리가 hole edge 근처인 경우
        (cv_result["confidence"] < 0.7)  # CV confidence가 낮은 경우
    )
    
    # CV 결과가 확실하면 (confidence > 0.8이고 INSERTED인 경우만) 우선 사용
    # 단, gap이 없고, overlap이 높고(>0.3), 거리가 가까워야(<1.0*radius) 함
    # 애매한 케이스가 아니어야 함
    if cv_result["confidence"] > 0.8 and cv_result["cv_status"] == "INSERTED" and not is_ambiguous_case:
        if not gap_detected and overlap_ratio > 0.3 and (hole_radius == 0 or dist_from_center < hole_radius * 1.0):
            return {
                "insertion_status": cv_result["cv_status"],
                "reason": f"CV geometric analysis (confidence: {cv_result['confidence']:.2f}). "
                         f"No gap detected, Tip inside: {details.get('tip_inside_hole', False)}, "
                         f"Overlap ratio: {overlap_ratio:.2f}",
                "status": "success",
                "cv_analysis": cv_result
            }
    
    # 1.5. KNN 유사도 측정 (Reference 이미지 기반)
    knn_result = analyze_with_knn_similarity(original_bgr)
    print(f"    🔍 KNN Analysis [{lead_name}]: {knn_result['knn_status']} (conf: {knn_result['confidence']:.2f})")
    if 'details' in knn_result and 'error' not in knn_result['details']:
        knn_details = knn_result['details']
        print(f"       - Avg INSERTED sim: {knn_details.get('avg_inserted_sim', 0):.3f}, "
              f"Avg NOT_INSERTED sim: {knn_details.get('avg_not_inserted_sim', 0):.3f}, "
              f"Diff: {knn_details.get('similarity_diff', 0):.3f}")
    
    # KNN 결과가 높은 confidence면 우선 사용 (CV보다 우선순위 낮지만 LLM보다 빠름)
    # 단, 애매한 케이스가 아니어야 함
    if knn_result["confidence"] > 0.75 and not is_ambiguous_case:
        knn_status = knn_result["knn_status"]
        knn_details = knn_result.get('details', {})
        if knn_status != "UNCERTAIN":
            return {
                "insertion_status": knn_status,
                "reason": f"KNN similarity analysis (confidence: {knn_result['confidence']:.2f}). "
                         f"Avg INSERTED similarity: {knn_details.get('avg_inserted_sim', 0):.3f}, "
                         f"Avg NOT_INSERTED similarity: {knn_details.get('avg_not_inserted_sim', 0):.3f}",
                "status": "success",
                "cv_analysis": cv_result,
                "knn_analysis": knn_result
            }
    
    # 애매한 케이스 감지 시 LLM으로 넘기기
    if is_ambiguous_case:
        print(f"    ⚠️  Ambiguous case detected [{lead_name}]: Overlap={overlap_ratio:.2f}, Gap_score={gap_score:.2f}, Distance_ratio={distance_ratio:.2f} → Using LLM for precise judgment")
    
    # 2. CV와 KNN 결과가 불확실하면 LLM에 힌트 제공
    headers = {
        "apikey": LUXIA_API_KEY,
        "Content-Type": "application/json"
    }
    
    # Reference 이미지 로드 (INSERTED 2개, NOT_INSERTED 2개)
    reference_images = []
    try:
        # INSERTED 예시 1
        ref_inserted_path = "./ref/INSERTED_01.png"
        if os.path.exists(ref_inserted_path):
            ref_inserted = Image.open(ref_inserted_path)
            reference_images.append({
                "type": "image_url",
                "image_url": {"url": pil_to_b64(ref_inserted)}
            })
        
        # INSERTED 예시 2
        ref_inserted2_path = "./ref/INSERTED_02.png"
        if os.path.exists(ref_inserted2_path):
            ref_inserted2 = Image.open(ref_inserted2_path)
            reference_images.append({
                "type": "image_url",
                "image_url": {"url": pil_to_b64(ref_inserted2)}
            })
        
        # NOT_INSERTED 예시 1
        ref_not1_path = "./ref/NOT_INSERTED_01.png"
        if os.path.exists(ref_not1_path):
            ref_not1 = Image.open(ref_not1_path)
            reference_images.append({
                "type": "image_url",
                "image_url": {"url": pil_to_b64(ref_not1)}
            })
        
        # NOT_INSERTED 예시 2
        ref_not2_path = "./ref/NOT_INSERTED_02.png"
        if os.path.exists(ref_not2_path):
            ref_not2 = Image.open(ref_not2_path)
            reference_images.append({
                "type": "image_url",
                "image_url": {"url": pil_to_b64(ref_not2)}
            })
    except Exception as e:
        print(f"    ⚠️  Reference 이미지 로드 실패: {e}")
    
    # 원본과 전처리된 이미지를 모두 LLM에 보냄
    images_content = [
        {"type": "image_url", "image_url": {"url": pil_to_b64(crop_images["original"])}},
        {"type": "image_url", "image_url": {"url": pil_to_b64(crop_images["preprocessed"])}}
    ]
    
    # Reference 이미지 추가
    images_content.extend(reference_images)
    
    # CV 힌트 생성 (개선)
    cv_status = cv_result['cv_status']
    cv_conf = cv_result['confidence']
    details = cv_result.get('details', {})
    
    overlap_ratio = details.get('overlap_ratio', 0)
    dist_from_center = details.get('distance_from_center', 0)
    hole_radius = details.get('hole_radius', 0)
    
    gap_detected = details.get('gap_detected', False)
    gap_score = details.get('gap_score', 0.0)
    distance_ratio = (dist_from_center / hole_radius) if hole_radius > 0 else 0.0
    
    # KNN 힌트 추가
    knn_status = knn_result.get('knn_status', 'UNCERTAIN')
    knn_conf = knn_result.get('confidence', 0.0)
    knn_details = knn_result.get('details', {})
    knn_avg_inserted = knn_details.get('avg_inserted_sim', 0.0)
    knn_avg_not_inserted = knn_details.get('avg_not_inserted_sim', 0.0)
    knn_sim_diff = knn_details.get('similarity_diff', 0.0)
    
    cv_hint = f"""
CV_ANALYSIS_HINT (geometric analysis):
- CV Status: {cv_status} (confidence: {cv_conf:.2f})
- Gap detected: {gap_detected} (score: {gap_score:.2f}) - MOST IMPORTANT: If gap_detected=True, the lead is likely NOT_INSERTED
- Lead tip distance from hole center: {dist_from_center:.1f} pixels
- Hole radius: {hole_radius:.1f} pixels
- Distance ratio: {distance_ratio:.2f}
- Tip inside hole (geometric): {details.get('tip_inside_hole', False)}
- Contour overlap ratio: {overlap_ratio:.2f}
- Depth cue (brightness difference): {details.get('depth_cue', 0):.1f}
- Votes: INSERTED={details.get('votes', {}).get('inserted', 0)}, NOT_INSERTED={details.get('votes', {}).get('not_inserted', 0)}

KNN_SIMILARITY_HINT (reference image comparison):
- KNN Status: {knn_status} (confidence: {knn_conf:.2f})
- Average similarity to INSERTED references: {knn_avg_inserted:.3f}
- Average similarity to NOT_INSERTED references: {knn_avg_not_inserted:.3f}
- Similarity difference: {knn_sim_diff:.3f} (positive = more similar to INSERTED, negative = more similar to NOT_INSERTED)
- If similarity_diff > 0.1: More similar to INSERTED examples
- If similarity_diff < -0.1: More similar to NOT_INSERTED examples

CRITICAL INTERPRETATION (in priority order):
1. GAP DETECTION (MOST IMPORTANT):
   - If gap_detected=True: The lead is VERY LIKELY NOT_INSERTED (there is a visible gap)
   - If gap_detected=False: Check other indicators

2. KNN Similarity (Reference comparison):
   - If similarity_diff > 0.1: Input image is more similar to INSERTED reference images → Likely INSERTED
   - If similarity_diff < -0.1: Input image is more similar to NOT_INSERTED reference images → Likely NOT_INSERTED
   - Compare with reference images visually to confirm

3. Overlap and Distance:
   - If overlap > 0.3 AND distance ratio < 1.0 AND NO gap: Likely INSERTED
   - If overlap < 0.15 OR distance ratio > 1.5: Likely NOT_INSERTED
   - **CRITICAL - AMBIGUOUS CASE (overlap 0.1-0.3)**: If overlap is 0.1-0.3, the lead may only be TOUCHING/BRUSHING the hole edge without actually entering. Check carefully:
     * If the lead touches the edge but the hole interior remains dark/unobstructed → NOT_INSERTED
     * If you see a bright crescent at the edge where lead touches but lead doesn't go INSIDE → NOT_INSERTED
     * Only if the lead clearly PENETRATES INTO the hole interior → INSERTED

4. Depth cue:
   - Can be positive or negative depending on lead color, so use as supporting evidence only

5. **EDGE CASE - "Touching but not inserted" (This is why you're being called):**
   - If overlap_ratio is 0.1-0.3 AND distance_ratio is 0.9-1.2: The lead may be touching the hole edge but NOT inserted
   - Look carefully: Does the lead actually ENTER the hole interior, or does it only touch the rim?
   - If the hole interior remains dark/unobstructed where the lead touches → NOT_INSERTED
   - If you see a bright area at the edge (crescent shape) but the lead doesn't penetrate → NOT_INSERTED
   - The lead must go INSIDE the hole, not just touch the edge

IMPORTANT DECISION RULES:
- If gap_detected=True → NOT_INSERTED (trust this over other indicators)
- If NO gap AND overlap > 0.3 AND distance < 1.0*radius → INSERTED
- **If overlap is 0.1-0.3 (ambiguous): Check if lead only touches edge or actually enters interior → If only touches, NOT_INSERTED**
- If KNN shows high similarity to INSERTED references AND no gap visible AND overlap > 0.3 → INSERTED
- If KNN shows high similarity to NOT_INSERTED references → NOT_INSERTED
- If you see a CLEAR GAP in the images → NOT_INSERTED (trust your eyes over CV/KNN)
- If you see NO gap AND lead clearly enters hole → INSERTED
- **If lead only touches/brushes hole edge without entering interior → NOT_INSERTED (even if no gap)**
- Do NOT default to INSERTED - only say INSERTED if you can clearly see NO gap and lead ENTERING the hole interior
"""
    
    # Reference 이미지 설명 추가
    reference_text = ""
    if reference_images:
        reference_text = "\n\n=== REFERENCE IMAGES (Use these as examples) ===\n"
        reference_text += "After the two input images, you will see reference images:\n"
        reference_text += "- REFERENCE_INSERTED_1 and REFERENCE_INSERTED_2: These are CORRECT examples of INSERTED lead. Notice: NO visible gap, lead clearly enters and PENETRATES INTO the hole interior, seamless connection, lead goes INSIDE the hole.\n"
        reference_text += "- REFERENCE_NOT_INSERTED_1 and REFERENCE_NOT_INSERTED_2: These are CORRECT examples of NOT_INSERTED leads. Notice: CLEAR visible gap between lead tip and hole, lead floats above the board, OR lead only touches/brushes the hole edge without entering the interior.\n"
        reference_text += "\n**IMPORTANT for AMBIGUOUS CASES:**\n"
        reference_text += "If the input image shows the lead touching or brushing against the hole edge (like a bright crescent at the edge), but the hole interior remains dark/unobstructed, this is NOT_INSERTED - the lead must actually ENTER and PENETRATE INTO the hole interior, not just touch the rim.\n"
        reference_text += "Compare the input images carefully with these references, especially checking if the lead goes INSIDE the hole or only touches the edge.\n"
        reference_text += "===============================================\n"
    
    prompt_with_images = LEAD_INSERTION_PROMPT + "\n\nYou are provided with TWO images: the first is the original cropped image, and the second is a preprocessed version. Use BOTH images to make your judgment. The original image may show clearer details about whether the lead is inserted, while the preprocessed image may help with contrast." + reference_text + cv_hint
    
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_with_images},
                    *images_content
                ]
            }
        ],
        "temperature": 0.2,  # 낮은 값으로 일관성 확보
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
                
                insertion_status = data.get("insertion_status", "UNCERTAIN").upper()
                if insertion_status not in ["INSERTED", "NOT_INSERTED", "UNCERTAIN"]:
                    insertion_status = "UNCERTAIN"
                
                return {
                    "insertion_status": insertion_status,
                    "reason": data.get("reason", ""),
                    "status": "success",
                    "cv_analysis": cv_result,
                    "knn_analysis": knn_result
                }
            except json.JSONDecodeError:
                return {
                    "insertion_status": "UNCERTAIN",
                    "reason": f"JSON parse error: {content[:100]}",
                    "status": "parse_error",
                    "cv_analysis": cv_result,
                    "knn_analysis": knn_result
                }
        
        return {
            "insertion_status": "UNCERTAIN",
            "reason": f"API error: {response.status_code}",
            "status": "error",
            "cv_analysis": cv_result,
            "knn_analysis": knn_result
        }
    
    except Exception as e:
        return {
            "insertion_status": "UNCERTAIN",
            "reason": f"Exception: {str(e)}",
            "status": "error",
            "cv_analysis": cv_result,
            "knn_analysis": knn_result
        }

def process_image_and_get_crops(image_url: str, img_id: str):
    """
    이미지를 처리하고 크롭 이미지 생성 (원본 + 전처리)
    
    Returns:
        crops: {"left": {"original": PIL.Image, "preprocessed": PIL.Image}, ...}
    """
    response = requests.get(image_url, timeout=30)
    response.raise_for_status()

    _, crops, _ = preprocess_and_make_lead_crops(response.content, img_id)
    
    return crops

def check_all_leads_insertion(crops: Dict[str, Dict[str, Image.Image]]) -> Dict:
    """
    3개의 크롭 이미지(원본 + 전처리)에서 각 리드의 삽입 여부를 판단
    
    Args:
        crops: {"left": {"original": PIL, "preprocessed": PIL}, ...}
    
    Returns:
        {
            "left": {"insertion_status": "...", "reason": "..."},
            "center": {"insertion_status": "...", "reason": "..."},
            "right": {"insertion_status": "...", "reason": "..."},
            "overall": "POSITIVE" | "NEGATIVE",  # 하나라도 NOT_INSERTED면 NEGATIVE
            "status": "success" | "error"
        }
    """
    results = {}
    
    for lead_name in ["left", "center", "right"]:
        if lead_name in crops and isinstance(crops[lead_name], dict):
            result = check_lead_insertion(crops[lead_name], lead_name)
            results[lead_name] = result
            # API Rate Limit 고려
            time.sleep(0.3)
        else:
            results[lead_name] = {
                "insertion_status": "UNCERTAIN",
                "reason": "Crop image not found",
                "status": "error"
            }
    
    # 하나라도 NOT_INSERTED면 NEGATIVE
    overall = "POSITIVE"
    for lead_name, result in results.items():
        if result.get("insertion_status") == "NOT_INSERTED":
            overall = "NEGATIVE"
            break
    
    results["overall"] = overall
    results["status"] = "success"
    
    return results

def describe_batch(csv_path: str, output_csv: str, output_json: str):
    """
    배치로 트랜지스터 이미지들을 묘사
    
    Args:
        csv_path: 입력 CSV 경로 (id, img_url 컬럼 필요)
        output_csv: 결과 CSV 저장 경로
        output_json: 결과 JSON 저장 경로 (전체 묘사 데이터)
    """
    df = pd.read_csv(csv_path)
    results = []
    descriptions_json = {}
    
    print(f"🔍 트랜지스터 이미지 묘사 시작 (총 {len(df)}개)")
    print("-" * 80)
    
    for idx, row in df.iterrows():
        img_id = row["id"]
        img_url = row["img_url"]
        
        print(f"\n[{idx+1}/{len(df)}] ID: {img_id}")
        
        try:
            # 1. 이미지 처리 및 크롭 생성
            crops = process_image_and_get_crops(img_url, img_id)
            
            # 2. 각 크롭 이미지에서 리드 삽입 여부 판단
            insertion_results = check_all_leads_insertion(crops)
            
            # CSV용 결과
            row_result = {
                'id': img_id,
                'img_url': img_url,
                'left_insertion': insertion_results.get('left', {}).get('insertion_status', 'UNCERTAIN'),
                'center_insertion': insertion_results.get('center', {}).get('insertion_status', 'UNCERTAIN'),
                'right_insertion': insertion_results.get('right', {}).get('insertion_status', 'UNCERTAIN'),
                'overall': insertion_results.get('overall', 'UNCERTAIN'),
                'status': insertion_results.get('status', 'error')
            }
            results.append(row_result)
            
            # JSON용 상세 데이터
            descriptions_json[img_id] = {
                'img_url': img_url,
                'left': insertion_results.get('left', {}),
                'center': insertion_results.get('center', {}),
                'right': insertion_results.get('right', {}),
                'overall': insertion_results.get('overall', 'UNCERTAIN'),
                'status': insertion_results.get('status', 'error')
            }
            
            # 출력
            print(f"  ✅ 판단 완료")
            print(f"  📌 Left: {insertion_results.get('left', {}).get('insertion_status', 'UNCERTAIN')}")
            print(f"  📌 Center: {insertion_results.get('center', {}).get('insertion_status', 'UNCERTAIN')}")
            print(f"  📌 Right: {insertion_results.get('right', {}).get('insertion_status', 'UNCERTAIN')}")
            print(f"  🎯 Overall: {insertion_results.get('overall', 'UNCERTAIN')}")
            
        except Exception as e:
            print(f"  ❌ 오류: {str(e)}")
            results.append({
                'id': img_id,
                'img_url': img_url,
                'left_insertion': 'UNCERTAIN',
                'center_insertion': 'UNCERTAIN',
                'right_insertion': 'UNCERTAIN',
                'overall': 'UNCERTAIN',
                'status': 'error'
            })
            
            descriptions_json[img_id] = {
                'img_url': img_url,
                'error': str(e),
                'status': 'error'
            }
    
    # 결과 저장 경로 폴더 생성
    output_dir = os.path.dirname(output_csv)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # CSV 저장 (최종 형식: id, label)
    # label: POSITIVE -> 0, NEGATIVE -> 1
    csv_results = []
    for r in results:
        overall = r.get('overall', 'UNCERTAIN')
        if overall == 'POSITIVE':
            label = 0
        elif overall == 'NEGATIVE':
            label = 1
        else:
            # UNCERTAIN인 경우 기본값 (또는 에러 처리)
            label = 1  # 안전하게 NEGATIVE로 처리
        
        csv_results.append({
            'id': r.get('id'),
            'label': label
        })
    
    result_df = pd.DataFrame(csv_results)
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
      output_csv="./test_output/transistor_descriptions_1221.csv",
      output_json="./test_output/transistor_descriptions_1221.json"
      )