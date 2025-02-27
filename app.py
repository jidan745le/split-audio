from flask import Flask, request, jsonify
import os
from werkzeug.utils import secure_filename
from audio_processor import process_audio_file  # 我们将把主要处理逻辑移到这个文件中

app = Flask(__name__)

# 配置上传文件存储路径
UPLOAD_FOLDER = '/app/uploads'
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg', 'flac'}

# 确保上传目录存在
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'}), 200

@app.route('/process_audio', methods=['POST'])
def process_audio():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        try:
            # 处理音频文件
            results = process_audio_file(filepath)
            print(results)
            # 处理完成后删除文件
            os.remove(filepath)
            return jsonify(results)
        except Exception as e:
            # 确保清理临时文件
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000) 