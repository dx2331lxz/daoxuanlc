#!/bin/bash

# 读取配置文件
source ./deploy.conf

# 检查配置是否完整
if [ -z "$HOST" ] || [ -z "$USER" ] || [ -z "$PROJECT_PATH" ]; then
    echo "错误：请在deploy.conf中填写完整的服务器配置信息"
    exit 1
fi

# 使用rsync推送文件
rsync -avz --exclude-from='.gitignore' ./ "$USER@$HOST:$PROJECT_PATH"

echo "文件推送完成！"