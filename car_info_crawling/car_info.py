from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By # By 사용
from selenium.webdriver.common.keys import Keys # Key 사용
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
import time
import os # 폴더 만들어주려고 추가함
import requests
import numpy as np
import pandas as pd

# 크롬 옵션 설정
chrome_option = Options()
chrome_option.add_argument("--start-maximized")  # 최대화된 창으로 열기
# chrome_option.add_argument("--headless") # GUI 창 안열기
chrome_option.add_argument("--incognito")  # 시크릿 창으로 열기

# 웹드라이버 설정
service = Service()
driver = webdriver.Chrome(service=service, options=chrome_option)
wait = WebDriverWait(driver, 10)

driver.get("https://search.naver.com/search.naver?where=nexearch&sm=top_hty&fbm=0&ie=utf8&query=%EC%9E%90%EB%8F%99%EC%B0%A8&ackey=jbbtcwu4")
driver.implicitly_wait(10)

driver.find_element(By.XPATH, '//*[@id="main_pack"]/div[5]/div[2]/div[1]/div/div[1]/div/div/div/ul/li[1]/div/a/span').click()
driver.implicitly_wait(10)

driver.find_element(By.XPATH, '//*[@id="main_pack"]/div[5]/div[2]/div[1]/div/div[2]/div/div/ul/li[3]/a/span').click()
driver.implicitly_wait(10)

# 2. 데이터프레임 초기화
df = pd.DataFrame(columns=["차종", "연비", "이미지"])

# 3. 차량 정보 크롤링 함수 정의
def extract_data():
    page_panels = driver.find_elements(By.CSS_SELECTOR, 'div.list_info._panel[style="display: flex;"]')
    for panel in page_panels:
        info_boxes = panel.find_elements(By.CLASS_NAME, "info_box")
        for box in info_boxes:
            # 차종
            try:
                car_name = box.find_element(By.CLASS_NAME, "_text").text.strip()
            except:
                car_name = ""

            # 연비
            try:
                sub_infos = box.find_elements(By.CLASS_NAME, "sub_info")
                if len(sub_infos) >= 2:
                    fuel_text = sub_infos[1].find_element(By.CLASS_NAME, "info_txt").text.strip()
                    fuel_text = fuel_text.replace("연비 ", "")
                else:
                    fuel_text = ""
            except:
                fuel_text = ""

            # 이미지
            try:
                img_tag = box.find_element(By.CSS_SELECTOR, "div.thumb_area img")
                img_url = img_tag.get_attribute("src")
            except:
                img_url = ""

            # 데이터 추가
            df.loc[len(df)] = [car_name, fuel_text, img_url]

# 4. 페이지 순회하며 크롤링
total_pages = 861
for page in range(total_pages):
    print(f"📄 페이지 {page+1} / {total_pages} 수집 중...")
    extract_data()

    # 다음 페이지 버튼 클릭
    try:
        next_button = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, 'div.cm_tab_info_box div.cm_paging_area.no_margin a.pg_next')))
        driver.execute_script("arguments[0].click();", next_button)
        time.sleep(2)  # 로딩 대기
    except Exception as e:
        print("🚫 다음 페이지 버튼 클릭 실패:", e)
        break

# 5. 결과 저장
df.to_csv("차량정보_전체.csv", index=False, encoding="utf-8-sig")
print("✅ 수집 완료 및 저장!")

# 6. 드라이버 종료
driver.quit()