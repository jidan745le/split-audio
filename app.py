from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO
import os
import traceback
import logging
from werkzeug.utils import secure_filename
from audio_processor import process_audio_file

# 配置详细日志
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
# 允许所有域名的跨域请求
CORS(app)
# 初始化SocketIO
socketio = SocketIO(app, cors_allowed_origins="*")

# 配置上传文件存储路径
UPLOAD_FOLDER = '/app/uploads'
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg', 'flac', 'MP3', 'WAV', 'OGG', 'FLAC'}  # 添加大写扩展名

# 确保上传目录存在
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    if '.' not in filename:
        logger.error(f"文件名 '{filename}' 没有扩展名")
        return False
    ext = filename.rsplit('.', 1)[1].lower()
    result = ext in {x.lower() for x in ALLOWED_EXTENSIONS}
    if not result:
        logger.error(f"扩展名 '{ext}' 不在允许列表中: {ALLOWED_EXTENSIONS}")
    return result

@app.route('/', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'}), 200

@app.route('/process_audio', methods=['POST'])
def process_audio():
    try:
        # 打印请求信息
        logger.info(f"收到请求: {request.method} {request.path}")
        logger.info(f"请求头: {dict(request.headers)}")
        logger.info(f"表单数据: {request.form}")
        logger.info(f"文件: {request.files}")
        user_id = request.form.get('userId')
        task_id = request.form.get('taskId')
        

        # 检查请求中是否有文件部分
        if 'file' not in request.files:
            logger.error("请求中没有'file'部分")
            return jsonify({'error': 'No file part in request'}), 400
        
        file = request.files['file']
        logger.info(f"获取到文件: {file}")
        logger.info(f"文件名: {file.filename}")
        
        # 检查文件名是否为空
        if file.filename == '':
            logger.error("文件名为空")
            return jsonify({'error': 'No selected file (empty filename)'}), 400
        
        # 检查文件类型
        logger.info(f"检查文件类型: {file.filename}")
        if '.' in file.filename:
            ext = file.filename.rsplit('.', 1)[1].lower()
            logger.info(f"文件扩展名: {ext}")
            logger.info(f"允许的扩展名: {ALLOWED_EXTENSIONS}")
        else:
            logger.error("文件名没有扩展名")
            return jsonify({'error': 'File has no extension'}), 400
        
        # 检查文件类型是否允许
        if not allowed_file(file.filename):
            logger.error(f"文件类型不允许: {file.filename}")
            return jsonify({'error': f'File type not allowed: {file.filename}'}), 400
        
        # 保存文件
        try:
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            logger.info(f"保存文件到: {filepath}")
            file.save(filepath)
            logger.info(f"文件保存成功: {filepath}")
        except Exception as e:
            logger.error(f"保存文件失败: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({'error': f'Failed to save file: {str(e)}'}), 500
        
        try:
            # 处理音频文件
            logger.info(f"开始处理音频文件: {filepath}")
            results = process_audio_file(filepath, socketio=socketio,user_id=user_id,task_id=task_id)
            logger.info("音频处理完成")
            
            # 删除临时文件
            if os.path.exists(filepath):
                os.remove(filepath)
                logger.info(f"临时文件已删除: {filepath}")
            
            return jsonify(results)
        except Exception as e:
            logger.error(f"处理音频文件失败: {str(e)}")
            logger.error(traceback.format_exc())
            
            # 确保清理临时文件
            if os.path.exists(filepath):
                os.remove(filepath)
                logger.info(f"临时文件已删除: {filepath}")
            
            return jsonify({'error': f'Failed to process audio: {str(e)}'}), 500
    
    except Exception as e:
        logger.error(f"处理请求时发生未捕获的异常: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500

# WebSocket事件处理
@socketio.on('connect')
def handle_connect():
    logger.info('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    logger.info('Client disconnected')

if __name__ == '__main__':
    logger.info("启动服务器...")
    socketio.run(app, host='0.0.0.0', port=8080, allow_unsafe_werkzeug=True) 