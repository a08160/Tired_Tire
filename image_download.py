import pandas as pd
import requests
import os

# CSV 파일 불러오기
csv_path = 'car_info.csv'  # 파일 경로를 필요에 따라 바꿔주세요
df = pd.read_csv(csv_path)

# 저장할 디렉토리 설정
save_dir = 'car_images'
os.makedirs(save_dir, exist_ok=True)

# 이미지 다운로드 및 저장
for idx, row in df.iterrows():
    car_name = row['차종']
    image_url = row['이미지']
    
    try:
        response = requests.get(image_url, timeout=5)
        response.raise_for_status()
        
        # 저장 경로 설정 (차종명을 파일명으로)
        file_path = os.path.join(save_dir, f"{car_name.replace('/', '_')}.jpg")
        
        # 이미지 저장
        with open(file_path, 'wb') as f:
            f.write(response.content)
        
        print(f"[{idx}] 저장 성공: {file_path}")
        
    except Exception as e:
        print(f"[{idx}] 저장 실패 ({car_name}): {e}")
