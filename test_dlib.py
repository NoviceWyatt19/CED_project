import os
import urllib.request
import bz2

# 파일 다운로드 URL
url = 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2'
# 다운로드 받을 파일 이름
file_name = 'shape_predictor_68_face_landmarks.dat.bz2'
# 압축 해제 후 사용할 파일 이름
extracted_file_name = 'shape_predictor_68_face_landmarks.dat'

# 파일이 이미 존재하는지 확인
if not os.path.isfile(extracted_file_name):
    # 파일 다운로드
    print("Downloading shape predictor model...")
    urllib.request.urlretrieve(url, file_name)
    
    # 압축 해제
    print("Extracting the file...")
    with bz2.BZ2File(file_name) as fr, open(extracted_file_name, 'wb') as fw:
        fw.write(fr.read())

    # 압축 파일 삭제 (원하지 않으면 주석 처리 가능)
    os.remove(file_name)
    print("Download and extraction complete.")
else:
    print("Shape predictor model already exists.")

# dlib 불러오기 및 shape_predictor 로드
import dlib

print("Loading shape predictor...")
s = dlib.shape_predictor(extracted_file_name)
print("Shape predictor loaded successfully.")
