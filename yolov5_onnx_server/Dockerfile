# 使用官方的 Python 基础映像，选择较小的版本
FROM python:3.9-slim-buster

# 设置容器中的工作目录
WORKDIR /app

# 复制依赖文件到镜像中的 /app 目录
COPY . /app

# 安装依赖项
RUN pip install --no-cache-dir -r requirements.txt

# Expose the required port
EXPOSE 8000

# 清理 pip 更新的内容
RUN pip uninstall --yes pip

# 启动服务器
CMD ["uvicorn", "main:app", "--host", "0.0.0.0"]