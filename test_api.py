import requests

def test_audio_processing(audio_file_path):
    url = "http://localhost:5000/process_audio"
    
    # 打开文件
    with open(audio_file_path, 'rb') as f:
        # 创建文件对象
        files = {'file': f}
        # 发送POST请求
        response = requests.post(url, files=files)
    
    # 打印响应
    print(response.json())

# 使用示例
test_audio_processing(r"C:\path\to\your\audio.wav") 