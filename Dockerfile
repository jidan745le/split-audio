# 使用CPU版本的Python基础镜像
FROM python:3.9-slim

# 设置工作目录
WORKDIR /app

# 设置环境变量
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV DEBIAN_FRONTEND=noninteractive

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# 复制requirements并安装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 创建上传目录
RUN mkdir -p /app/uploads

# 复制应用文件
COPY app.py .
COPY audio_processor.py .

# 设置环境变量
COPY .env .

# 暴露端口
EXPOSE 5000

# 启动应用
CMD ["python", "app.py"] 