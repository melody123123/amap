# 🗺️ 智能路线规划助手

基于高德地图 MCP 服务 + 通义千问 AI Agent 的智能路线规划系统，支持多途经点优化、兴趣点搜索和对话式交互。

## ✨ 功能特性

- 🚗 **多出行方式支持**：驾车、步行、公共交通
- 📍 **智能途经点规划**：自动优化途经点顺序，生成最佳路线
- 🔍 **兴趣点搜索**：对话式搜索沿途兴趣点（餐厅、加油站、休息区等）
- 💬 **AI 对话交互**：自然语言描述需求，AI 自动理解并搜索
- 🗺️ **高德地图集成**：一键打开高德地图导航
- 📊 **路线详情展示**：距离、时间、详细步骤等信息

## 📋 环境要求

- Python 3.11+
- 高德地图 API Key
- 通义千问 API Key（阿里云 DashScope）

## 🚀 快速开始

### 1. 克隆项目

```bash
git clone <repository-url>
cd amap
```


### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置 API 密钥

编辑 `config.json` 文件，填入你的 API 密钥：

```json
{
  "openai": {
    "model": "qwen-plus",
    "temperature": 0.3,
    "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "api_key": "你的通义千问API密钥"
  },
  "agent": {
    "type": "OPENAI_FUNCTIONS",
    "verbose": true
  },
  "amap": {
    "api_key": "你的高德地图API密钥"
  }
}
```

**获取 API 密钥：**
- **通义千问 API Key**：访问 [阿里云 DashScope](https://dashscope.console.aliyun.com/) 获取
- **高德地图 API Key**：访问 [高德开放平台](https://lbs.amap.com/) 获取

### 5. 启动后端服务器

**方式一：直接运行（开发模式，支持热重载）**
```bash
python api_server.py
```

**方式二：使用启动脚本（生产模式，后台运行）**

**Linux/Mac:**
```bash
chmod +x start_server.sh
./start_server.sh
```

**Windows:**
```powershell
# 使用 PowerShell 运行
python api_server.py prod
```

后端服务器将在 `http://localhost:8000` 启动。

### 6. 打开前端页面

**方式一：直接打开 HTML 文件**
- 双击 `index.html` 文件，在浏览器中打开

**方式二：使用 HTTP 服务器（推荐）**
```bash
# Python 3
python -m http.server 8080

# 然后在浏览器访问 http://localhost:8080/index.html
```

## 🌐 外部网络访问配置

**默认情况下，前端配置为本地访问模式**（`http://localhost:8000`）。如果需要从外部网络访问，请按以下步骤修改：

### 1. 修改前端 API 地址

编辑 `index.html` 文件，找到 `getApiBaseUrl()` 函数（约第 891 行），修改为你的服务器地址：

```javascript
function getApiBaseUrl() {
  // 将 localhost 替换为你的服务器 IP 或域名
  // 例如：'http://192.168.1.100:8000' 或 'http://your-domain.com:8000'
  return 'http://你的服务器IP:8000';
  
  // 如果使用 Nginx 代理，可以使用相对路径：
  // return '';
}
```

### 2. 修改后端监听地址（可选）

如果需要从外部访问，确保后端服务器监听所有网络接口。编辑 `api_server.py`，检查启动配置：

```python
# 确保 host 设置为 "0.0.0.0"（默认已配置）
uvicorn.run("api_server:app", host="0.0.0.0", port=8000, ...)
```

### 3. 防火墙配置

确保防火墙允许 8000 端口的访问：

**Linux:**
```bash
# Ubuntu/Debian
sudo ufw allow 8000

# CentOS/RHEL
sudo firewall-cmd --add-port=8000/tcp --permanent
sudo firewall-cmd --reload
```

**Windows:**
- 打开 Windows Defender 防火墙
- 添加入站规则，允许 8000 端口

### 4. 使用 Nginx 反向代理（生产环境推荐）

如果需要使用 Nginx 作为反向代理，配置示例：

```nginx
server {
    listen 80;
    server_name your-domain.com;

    # 前端静态文件
    location / {
        root /path/to/amap;
        index index.html;
        try_files $uri $uri/ /index.html;
    }

    # 后端 API 代理
    location /api {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
}
```

然后修改前端 `getApiBaseUrl()` 返回空字符串（使用相对路径）。

## 📁 项目结构

```
amap/
├── api_server.py          # FastAPI 后端服务器
├── amap_mcp_agent.py     # MCP Agent 实现
├── amap_tools.py         # 高德地图工具函数
├── config.json           # 配置文件（API 密钥）
├── index.html            # 前端页面
├── requirements.txt      # Python 依赖
├── start_server.sh       # 启动脚本（Linux/Mac）
├── stop_server.sh        # 停止脚本（Linux/Mac）
├── route_cache.json      # 路线缓存文件（自动生成）
└── README.md             # 本文件
```

## 🔧 常用命令

### 查看服务器日志

**Linux/Mac:**
```bash
tail -f logs/api_server.log
```

**Windows:**
```powershell
Get-Content logs\api_server.log -Wait
```

### 停止服务器

**Linux/Mac:**
```bash
./stop_server.sh
```

**Windows:**
```powershell
# 查找进程
netstat -ano | findstr :8000
# 结束进程（替换 PID 为实际进程ID）
taskkill /PID <PID> /F
```

## ❓ 常见问题

### 1. CORS 跨域错误

后端已配置允许所有来源的 CORS，如果仍遇到问题，检查：
- 后端服务器是否正常运行
- 前端 API 地址配置是否正确
- 浏览器控制台是否有具体错误信息

### 2. API 调用失败

- 检查 `config.json` 中的 API 密钥是否正确
- 确认 API 密钥有足够的配额
- 查看后端日志了解详细错误信息

### 3. 前端无法连接后端

- 确认后端服务器已启动（访问 `http://localhost:8000/docs` 查看 API 文档）
- 检查前端 `getApiBaseUrl()` 配置的地址是否正确
- 如果从外部访问，确保防火墙已开放端口

### 4. 路线规划超时

- 路线规划可能需要 30-90 秒，请耐心等待
- 如果长时间无响应，检查网络连接和 API 服务状态
- 查看后端日志了解具体错误

## 📝 开发说明

### 开发模式

开发模式下，修改代码后服务器会自动重启：

```bash
python api_server.py
```

### 生产模式

生产模式使用多进程，性能更好：

```bash
python api_server.py prod
```

或使用启动脚本：

```bash
./start_server.sh
```

## 📄 许可证

本项目仅供学习和研究使用。

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

---

**提示**：首次使用前，请确保已正确配置 API 密钥，否则功能将无法正常使用。

