import cv2

# 이미지 로드
image = cv2.imread("./defect_data/defective_train/Defective (300).jpg")  # 입력할 이미지 파일 이름
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 흑백 변환
inverted_image = cv2.bitwise_not(gray_image)  # 색상 반전

# 결과 출력
cv2.imshow("Gray Image", gray_image)
cv2.imshow("Inverted Image", inverted_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 저장 (선택)
cv2.imwrite("inverted_image.jpg", inverted_image)