import os
import json
import re
import time
import logging
from collections import Counter
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any

import dashscope
import httpx
from qwen_agent.agents import Assistant

# 配置日志输出
# 使用 force=True 避免多次调用 basicConfig 导致重复日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    force=True  # Python 3.8+ 支持，避免重复配置
)
logger = logging.getLogger(__name__)

# 降低 qwen_agent 相关库的日志级别，减少重复输出
# qwen_agent 库内部可能有多个 logger，导致重复输出
logging.getLogger('qwen_agent').setLevel(logging.WARNING)
logging.getLogger('qwen_agent_logger').setLevel(logging.WARNING)
logging.getLogger('base').setLevel(logging.WARNING)  # qwen_agent 内部可能使用的 logger
logging.getLogger('httpx').setLevel(logging.WARNING)  # 减少 HTTP 请求日志的冗余输出


def load_config() -> dict:
    """从 config.json 中读取配置（如果存在）。"""
    cfg_path = Path(__file__).with_name("config.json")
    if not cfg_path.exists():
        return {}
    try:
        with cfg_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return {}


def build_route_agent(transport_mode: str = "driving") -> Assistant:
    """
    使用百炼 MCP SDK（qwen-agent）构建一个调用 amap-maps MCP 服务的本地路线规划智能体。
    - LLM：使用百炼的通义千问（如 qwen-max / qwen-plus 等），模型名从 config.json 读取；
    - API Key：优先使用 config.json 中的 openai.api_key，其次使用环境变量 DASHSCOPE_API_KEY；
    - 工具：远端 amap-maps MCP 服务（文档示例：https://help.aliyun.com/zh/model-studio/mcp-external-calls）。
    - transport_mode: 出行方式，可选值："walking"（步行）、"driving"（驾车）、"transit"（公共交通）
    """
    cfg = load_config()
    openai_cfg = cfg.get("openai", {})
    
    # 模型名称：优先取 config.json.openai.model，否则默认 qwen-max
    model_name = openai_cfg.get("model", "qwen-max")
    
    # API Key：优先取 config.json.openai.api_key，否则退回到环境变量 DASHSCOPE_API_KEY
    api_key = openai_cfg.get("api_key") or os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "未找到 DashScope API Key。\n"
            "请在 config.json 的 openai.api_key 中配置，或在环境变量中设置 DASHSCOPE_API_KEY。"
        )
    
    # 配置 DashScope SDK 使用同一个 Key，避免模型调用报错
    dashscope.api_key = api_key
    os.environ["DASHSCOPE_API_KEY"] = api_key
    
    # LLM 配置（可根据需要改成 qwen-plus / qwen-turbo 等，对应配置到 config.json 即可）
    llm_cfg = {"model": model_name}
    
    # 根据出行方式选择对应的路线规划工具和描述
    if transport_mode == "walking":
        route_tool = "maps_direction_walking（步行路线规划）"
        route_description = "步行路线"
        route_notes = "步行路线，请考虑人行道、过街天桥、地下通道等步行设施。"
    elif transport_mode == "transit":
        route_tool = "maps_direction_transit（公共交通路线规划，如果可用）或其他合适的工具"
        route_description = "公共交通路线"
        route_notes = "公共交通路线，请考虑地铁、公交、轻轨等公共交通工具，提供换乘方案。"
    else:  # 默认为驾车
        route_tool = "maps_direction_driving（驾车路线规划）"
        route_description = "驾车路线"
        route_notes = "驾车路线，请考虑高速公路、主要道路和重要路口。"
    
    system = (
        "你是一个路线规划智能体。\n"
        "你将调用名为 amap-maps 的 MCP 服务来完成以下任务：\n"
        "1. 根据用户的自然语言描述，先确认出发地和目的地所在的城市/行政区；\n"
        "2. 必要时调用地理编码/逆地理/IP 定位等能力，避免地点歧义；\n"
        f"3. 使用{route_tool}生成合理的{route_description}；\n"
        "4. 最终用简体中文给出：\n"
        "   - 大致路线概览（尽量包含沿途经过的主要城市、城区、道路、高速和重要节点）；\n"
        "   - 预估里程和时间（如有）；\n"
        f"   - {route_notes}\n"
        "如有信息不确定，请向用户追问澄清，不要自行臆测。"
    )

    # MCP 工具配置：amap-maps
    # 参考官方文档示例：https://help.aliyun.com/zh/model-studio/mcp-external-calls
    tools = [
        {
            "mcpServers": {
                "amap-maps": {
                    "url": "https://dashscope.aliyuncs.com/api/v1/mcps/amap-maps/sse",
                    "headers": {
                        "Authorization": f"Bearer {api_key}",
                    },
                }
            }
        }
    ]

    bot = Assistant(
        llm=llm_cfg,
        name="路线规划智能体",
        description="基于高德 Amap Maps MCP 的路线规划",
        system_message=system,
        function_list=tools,
    )
    return bot


def extract_waypoint_names_from_json(waypoint_json_text: str) -> List[Dict[str, str]]:
    """
    从 JSON 文本中提取转弯点信息（名称、坐标等）。
    
    返回格式：[{"name": "节点名", "type": "节点类型", "lon": 经度, "lat": 纬度}, ...]
    """
    waypoints = []
    
    # 尝试解析 JSON
    try:
        # 清理可能的 markdown 代码块标记
        cleaned = waypoint_json_text.strip()
        if cleaned.startswith("```"):
            # 移除 markdown 代码块标记
            lines = cleaned.split("\n")
            cleaned = "\n".join(lines[1:-1]) if len(lines) > 2 else cleaned
        elif cleaned.startswith("```json"):
            lines = cleaned.split("\n")
            cleaned = "\n".join(lines[1:-1]) if len(lines) > 2 else cleaned
        
        data = json.loads(cleaned)
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    # 优先使用坐标，如果没有坐标则使用名称
                    if "lon" in item and "lat" in item:
                        lon = item.get("lon")
                        lat = item.get("lat")
                        name = item.get("name", f"{lon},{lat}")
                    elif "name" in item:
                        name = item.get("name", "")
                        lon = None
                        lat = None
                    else:
                        continue
                    
                    waypoints.append({
                        "name": name,
                        "type": item.get("type", "turn"),
                        "note": item.get("note", ""),
                        "lon": lon,
                        "lat": lat
                    })
    except json.JSONDecodeError:
        # 如果 JSON 解析失败，尝试用正则表达式提取
        # 匹配 "name": "xxx" 的模式
        name_pattern = r'"name"\s*:\s*"([^"]+)"'
        type_pattern = r'"type"\s*:\s*"([^"]+)"'
        lon_pattern = r'"lon"\s*:\s*([\d.]+)'
        lat_pattern = r'"lat"\s*:\s*([\d.]+)'
        
        names = re.findall(name_pattern, waypoint_json_text)
        types = re.findall(type_pattern, waypoint_json_text)
        lons = re.findall(lon_pattern, waypoint_json_text)
        lats = re.findall(lat_pattern, waypoint_json_text)
        
        for i, name in enumerate(names):
            lon = float(lons[i]) if i < len(lons) else None
            lat = float(lats[i]) if i < len(lats) else None
            waypoints.append({
                "name": name,
                "type": types[i] if i < len(types) else "turn",
                "note": "",
                "lon": lon,
                "lat": lat
            })
    
    return waypoints


def calculate_midpoint(lon1: float, lat1: float, lon2: float, lat2: float) -> Tuple[float, float]:
    """
    计算两个经纬度坐标之间的中点。
    
    返回：(中点经度, 中点纬度)
    """
    mid_lon = (lon1 + lon2) / 2.0
    mid_lat = (lat1 + lat2) / 2.0
    return (mid_lon, mid_lat)


def run_agent_with_retry(agent: Assistant, messages: List[Dict], max_retries: int = 2, retry_delay: float = 3.0, step_name: str = "") -> List:
    """
    带重试机制的 Agent 运行函数，处理网络连接错误，并打印详细的调用过程。
    
    Args:
        agent: Assistant 实例
        messages: 消息列表
        max_retries: 最大重试次数
        retry_delay: 重试延迟（秒）
        step_name: 步骤名称（用于日志输出）
    
    Returns:
        响应列表，如果所有重试都失败则返回空列表
    """
    if step_name:
        print(f"\n[智能体调用] {step_name}")
        print("-" * 50)
        print(f"[输入消息] {messages[-1].get('content', '')[:100]}...")
    
    for attempt in range(max_retries + 1):
        try:
            if attempt > 0:
                print(f"[重试] 第 {attempt} 次重试...")
            
            responses = []
            print("[开始执行] 智能体正在处理...")
            
            # 添加超时保护：如果生成器长时间无响应或整体耗时过长，抛出超时异常
            # 不再使用“按流式 chunk 次数计数”的 max_iterations，避免把一次完整回复拆成很多迭代。
            if "路线摘要" in step_name or "提取关键节点" in step_name:
                # 简单的摘要和提取任务，整体期望在 60 秒内完成
                max_total_seconds = 60
            else:
                # 复杂的对话任务，允许更长的总时长
                max_total_seconds = 120
            
            start_time = time.time()
            last_response_time = start_time
            max_idle_seconds = 30  # 如果30秒没有新响应，认为卡死

            # 对于“提取关键节点”这类严格要求输出 JSON 数组的任务，
            # 增加一个“JSON 解析成功即提前结束”的早停机制，减少无意义重复输出。
            enable_json_early_stop = "提取关键节点" in step_name
            full_text_for_json = ""  # 累积当前 assistant 的完整文本，用于尝试 json.loads
            
            try:
                for idx, resp in enumerate(agent.run(messages)):
                    current_time = time.time()
                    elapsed_time = current_time - start_time
                    idle_time = current_time - last_response_time
                    
                    # 检查总超时时间
                    if elapsed_time > max_total_seconds:
                        # 不再抛出异常中断整个请求，只打印日志并结束当前智能体调用，返回已获取的部分结果
                        print(
                            f"[警告] 智能体执行超过总超时时间 ({max_total_seconds} 秒)，自动结束本次生成（返回已完成部分）"
                        )
                        break
                    
                    # 检查空闲时间（长时间无新响应）
                    if idle_time > max_idle_seconds:
                        print(
                            f"[警告] 智能体超过 {max_idle_seconds} 秒无新响应，自动结束本次生成（返回已完成部分）"
                        )
                        break
                    
                    # 更新最后响应时间
                    last_response_time = current_time
                    
                    # 每隔固定时间打印一次进度（仅基于耗时，不再关心流式块数量）
                    if int(elapsed_time) % 20 == 0:
                        print(f"[进度] 已运行 {elapsed_time:.1f} 秒...")
                    
                    responses.append(resp)
                    
                    # 提取工具调用和回复内容（仅用于日志与 JSON 早停，不再做“死循环”强制中断）
                    current_response_text = None
                    
                    if isinstance(resp, list):
                        for item in resp:
                            if isinstance(item, dict):
                                role = item.get("role", "")
                                content = item.get("content", "")
                                function_call = item.get("function_call")
                                
                                if function_call:
                                    # 这里只做日志展示，避免死循环检测误伤正常长对话
                                    pass
                                elif role == "assistant" and content:
                                    # 1) 保存前 200 字符用于重复检测
                                    current_response_text = content[:200]

                                    # 2) 若是“提取关键节点”步骤，累积完整文本并尝试解析为 JSON 数组，成功则立刻早停
                                    if enable_json_early_stop:
                                        full_text_for_json += content
                                        text_stripped = full_text_for_json.strip()
                                        # 只有在看起来像完整 JSON 数组时才尝试解析，避免对半截文本频繁 json.loads
                                        if text_stripped.startswith("[") and text_stripped.endswith("]"):
                                            try:
                                                parsed = json.loads(text_stripped)
                                                if isinstance(parsed, list):
                                                    print("[完成] 检测到完整 JSON 数组，提前结束（提取关键节点）")
                                                    # 提前结束当前智能体运行循环
                                                    raise StopIteration
                                            except Exception:
                                                # 解析失败说明还不是完整 JSON，继续累积
                                                pass
                    elif isinstance(resp, dict):
                        if "function_call" in resp:
                            # 这里只做日志展示
                            pass
                    
                    # 打印中间响应（工具调用等）
                    if isinstance(resp, list):
                        for item in resp:
                            if isinstance(item, dict):
                                role = item.get("role", "")
                                content = item.get("content", "")
                                function_call = item.get("function_call")
                                
                                if function_call:
                                    func_name = function_call.get("name", "unknown")
                                    func_args = function_call.get("arguments", {})
                                    print(f"[工具调用] {func_name}")
                                    if func_args:
                                        print(f"  └─ 参数: {json.dumps(func_args, ensure_ascii=False, indent=2)[:200]}...")
                                elif role == "assistant" and content:
                                    # 只打印前200个字符，避免输出过长
                                    preview = content[:200].replace("\n", " ")
                                    print(f"[智能体回复] {preview}...")
                    elif isinstance(resp, dict):
                        if "function_call" in resp:
                            func_name = resp["function_call"].get("name", "unknown")
                            print(f"[工具调用] {func_name}")
            except (httpx.HTTPStatusError, httpx.RequestError, httpx.TimeoutException) as e:
                # 捕获生成器内部可能抛出的HTTP错误
                print(f"[错误] 生成器内部HTTP错误: {type(e).__name__}: {e}")
                raise  # 重新抛出，让外层异常处理处理
            except Exception as e:
                # 捕获其他异常，包括 SSE 错误（MCP 服务连接问题）
                error_str = str(e).lower()
                error_type = type(e).__name__
                
                # 检查是否是 SSE 相关错误（MCP 服务连接问题）
                if "sse" in error_str or "event-stream" in error_str or "content-type" in error_str:
                    print(f"[错误] MCP SSE 连接错误: {error_type}: {e}")
                    print(f"[错误] MCP 服务可能暂时不可用或返回了错误的响应头")
                    # 将 SSE 错误转换为可重试的错误
                    raise httpx.HTTPStatusError(
                        "MCP SSE connection error",
                        request=None,
                        response=None
                    ) from e
                else:
                    # 其他未知错误，重新抛出
                    print(f"[错误] 生成器内部未知错误: {error_type}: {e}")
                    raise
            
            if responses:
                print(f"[完成] 共收到 {len(responses)} 个响应")
            
            return responses
        except (httpx.RemoteProtocolError, httpx.ConnectError, httpx.TimeoutException) as e:
            if attempt < max_retries:
                print(f"[错误] 网络连接错误: {type(e).__name__}")
                print(f"[重试] {retry_delay} 秒后重试 ({attempt + 1}/{max_retries})...")
                time.sleep(retry_delay)
                continue
            else:
                print(f"[失败] 网络连接失败（已重试 {max_retries} 次）: {e}")
                logger.error(f"网络连接失败: {e}", exc_info=True)
                return []
        except httpx.HTTPStatusError as e:
            # 处理 HTTP 状态码错误（如 503 Service Unavailable）
            status_code = e.response.status_code if e.response else 0
            if status_code in [503, 502, 504]:  # 服务不可用、网关错误、网关超时
                if attempt < max_retries:
                    wait_time = retry_delay * (attempt + 1)  # 递增延迟
                    print(f"[错误] 服务暂时不可用 (HTTP {status_code}): {e}")
                    print(f"[重试] {wait_time} 秒后重试 ({attempt + 1}/{max_retries})...")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"[失败] 服务不可用（已重试 {max_retries} 次）: HTTP {status_code}")
                    logger.error(f"服务不可用: HTTP {status_code} - {e}", exc_info=True)
                    return []
            else:
                # 其他 HTTP 错误（如 401, 403, 500）不重试
                print(f"[错误] HTTP 错误 (HTTP {status_code}): {e}")
                logger.error(f"HTTP 错误: HTTP {status_code} - {e}", exc_info=True)
                return []
        except (TimeoutError, KeyboardInterrupt) as e:
            # 超时或中断错误，不重试
            print(f"[错误] 超时或中断: {type(e).__name__}: {e}")
            logger.error(f"超时或中断: {e}", exc_info=True)
            return []
        except Exception as e:
            # 其他类型的错误，检查是否是HTTP相关错误或SSE错误
            error_str = str(e).lower()
            error_type = type(e).__name__
            
            # 检查是否是 SSE 错误（MCP 服务连接问题）
            if "sse" in error_str or "event-stream" in error_str or "content-type" in error_str or "SSEError" in error_type:
                print(f"[错误] MCP SSE 连接错误: {error_type}: {e}")
                print(f"[错误] MCP 服务可能暂时不可用或返回了错误的响应头（Content-Type 缺失）")
                # SSE 错误通常表示服务不可用，应该重试
                if attempt < max_retries:
                    wait_time = retry_delay * (attempt + 1)
                    print(f"[重试] {wait_time} 秒后重试 ({attempt + 1}/{max_retries})...")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"[失败] MCP SSE 连接失败（已重试 {max_retries} 次）: {error_type}: {e}")
                    logger.error(f"MCP SSE 连接失败: {e}", exc_info=True)
                    return []
            # 检查是否是服务不可用错误
            elif "503" in error_str or "service unavailable" in error_str or "502" in error_str or "504" in error_str:
                # 可能是包装在其他异常中的HTTP错误
                if attempt < max_retries:
                    wait_time = retry_delay * (attempt + 1)
                    print(f"[错误] 检测到服务不可用错误: {error_type}: {e}")
                    print(f"[重试] {wait_time} 秒后重试 ({attempt + 1}/{max_retries})...")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"[失败] 服务不可用（已重试 {max_retries} 次）: {error_type}: {e}")
                    logger.error(f"服务不可用: {e}", exc_info=True)
                    return []
            else:
                # 其他错误直接返回，不重试
                print(f"[错误] 发生异常: {error_type}: {e}")
                logger.error(f"执行错误: {e}", exc_info=True)
                return []
    
    return []


def get_waypoint_coordinates_from_name(poi_bot: Assistant, waypoint_name: str) -> Optional[Tuple[float, float]]:
    """
    通过 MCP 工具获取途经点的经纬度坐标。
    
    返回：(经度, 纬度) 或 None（如果获取失败）
    """
    geo_prompt = (
        f"请调用地理编码工具（maps_geo）获取地点「{waypoint_name}」的经纬度坐标。\n"
        f"只返回坐标，格式为：经度,纬度（例如：120.123456,30.123456）"
    )
    
    geo_messages = [{"role": "user", "content": geo_prompt}]
    geo_responses = run_agent_with_retry(
        poi_bot, 
        geo_messages, 
        max_retries=2, 
        step_name=f"获取坐标: {waypoint_name}"
    )
    
    if geo_responses:
        last = geo_responses[-1]
        content = ""
        if isinstance(last, list):
            for item in last:
                if isinstance(item, dict) and item.get("role") == "assistant" and "content" in item:
                    content = item["content"]
        elif isinstance(last, dict) and "content" in last:
            content = last["content"]
        
        # 尝试从返回内容中提取坐标（格式：lon,lat）
        coord_pattern = r'(\d+\.?\d*)\s*[,，]\s*(\d+\.?\d*)'
        match = re.search(coord_pattern, content)
        if match:
            try:
                lon = float(match.group(1))
                lat = float(match.group(2))
                return (lon, lat)
            except ValueError:
                pass
    
    return None


def build_chat_poi_agent() -> Assistant:
    """
    构建对话式兴趣点助手，支持多轮对话和需求提炼。
    能够理解用户的自然语言表达，提炼出兴趣点类型。
    注意：这个助手不直接调用搜索工具，只负责对话和关键词提取。
    """
    cfg = load_config()
    openai_cfg = cfg.get("openai", {})
    
    model_name = openai_cfg.get("model", "qwen-max")
    api_key = openai_cfg.get("api_key") or os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "未找到 DashScope API Key。\n"
            "请在 config.json 的 openai.api_key 中配置，或在环境变量中设置 DASHSCOPE_API_KEY。"
        )
    
    dashscope.api_key = api_key
    os.environ["DASHSCOPE_API_KEY"] = api_key
    
    llm_cfg = {"model": model_name}
    
    system = (
        "你是一个智能的旅行助手，专门帮助用户规划路线和找到感兴趣的地点。\n\n"
        "你的任务：\n"
        "1. 与用户进行自然对话，理解他们的需求\n"
        "2. 识别用户的意图：搜索沿途兴趣点 OR 添加途经点到路线\n"
        "3. 根据用户需求，提取关键词或地点名称\n\n"
        "用户意图识别：\n"
        "【搜索沿途兴趣点】- 用户想在路线附近找某类地方（如餐厅、加油站）\n"
        "  示例：'找个火锅店'、'附近有加油站吗'、'我想喝咖啡'\n"
        "  输出格式：「关键词：XXX」\n"
        "\n"
        "【添加途经点】- 用户想在路线中增加一个必经的地点\n"
        "  示例：'我要去天安门'、'加一个故宫'、'途经上海外滩'\n"
        "  输出格式：「途经点：XXX」\n"
        "\n"
        "关键区别：\n"
        "- 搜索：用户想看附近有什么（不确定具体去哪），使用「关键词：」\n"
        "- 途经点：用户明确要去某个地方，使用「途经点：」\n"
        "\n"
        "搜索示例：\n"
        "- '找个火锅店' → 「关键词：火锅」\n"
        "- '附近有加油站吗' → 「关键词：加油站」\n"
        "- '我想喝咖啡' → 「关键词：咖啡厅」\n"
        "\n"
        "途经点示例：\n"
        "- '我要去天安门' → 「途经点：天安门」\n"
        "- '加一个故宫' → 「途经点：故宫」\n"
        "- '途经上海外滩' → 「途经点：上海外滩」\n"
        "- '经过杭州西湖' → 「途经点：杭州西湖」\n"
        "\n"
        "回复规则：\n"
        "- 如果用户需求不明确，询问更多信息\n"
        "- 识别出意图后，立即输出对应格式\n"
        "- 一次只处理一个需求\n"
        "- 不要重复问候或介绍自己\n"
        "- 保持回复简洁友好"
    )
    
    # 不使用任何工具，纯对话助手
    agent = Assistant(
        llm=llm_cfg,
        system_message=system,
        function_list=[],  # 空列表，不需要工具
    )
    return agent


def build_poi_search_agent() -> Assistant:
    """
    构建一个用于在途经点附近搜索 POI 的智能体。
    使用 amap-maps MCP 的周边搜索功能。
    """
    cfg = load_config()
    openai_cfg = cfg.get("openai", {})
    
    model_name = openai_cfg.get("model", "qwen-max")
    api_key = openai_cfg.get("api_key") or os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "未找到 DashScope API Key。\n"
            "请在 config.json 的 openai.api_key 中配置，或在环境变量中设置 DASHSCOPE_API_KEY。"
        )
    
    dashscope.api_key = api_key
    os.environ["DASHSCOPE_API_KEY"] = api_key
    
    llm_cfg = {"model": model_name}
    
    system = (
        "你是一个 POI 搜索助手，使用 amap-maps MCP 服务在指定地点附近搜索相关目标。\n"
        "你的任务是：\n"
        "1. 对于给定的节点名称，先调用地理编码工具（maps_geo）获取该节点的经纬度坐标；\n"
        "2. 然后调用周边搜索工具（maps_around_search），在节点附近搜索用户指定的关键词；\n"
        "3. 重要：只使用 maps_geo 和 maps_around_search 这两个工具，不要调用 maps_search_detail 工具获取详细信息；\n"
        "4. 对于每个节点，只返回搜索到的最相关结果的基本信息，并且必须包含 POI 的 id：\n"
        "   - 建议输出 JSON 数组，每个元素包含 name、distance、id 三个字段；\n"
        "   - 或者在每一行文本中显式包含 id，例如：\"有家小院川渝市井火锅（距离约60米，id=B0FFHVFVO0）\"；\n"
        "   - 每个节点最多返回 3-5 个结果，用简洁的列表或 JSON 格式；\n"
        "   - 直接从 maps_around_search 的返回结果中提取名称、距离和 poi.id，不要调用其他工具获取详细信息；\n"
        "5. 如果某个节点无法找到坐标或搜索无结果，请明确说明；\n"
        "6. 最终用简体中文简洁输出，不要包含过多细节。"
    )
    
    # MCP 工具配置：amap-maps（包含地理编码和周边搜索）
    tools = [
        {
            "mcpServers": {
                "amap-maps": {
                    "url": "https://dashscope.aliyuncs.com/api/v1/mcps/amap-maps/sse",
                    "headers": {
                        "Authorization": f"Bearer {api_key}",
                    },
                }
            }
        }
    ]
    
    bot = Assistant(
        llm=llm_cfg,
        name="POI 搜索智能体",
        description="在途经点附近搜索相关 POI",
        system_message=system,
        function_list=tools,
    )
    return bot


def build_poi_detail_agent() -> Assistant:
    """
    构建一个用于根据 POI id 查询兴趣点详细信息的智能体。
    主要使用 amap-maps 的 maps_search_detail 能力。
    """
    cfg = load_config()
    openai_cfg = cfg.get("openai", {})

    model_name = openai_cfg.get("model", "qwen-max")
    api_key = openai_cfg.get("api_key") or os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "未找到 DashScope API Key。\n"
            "请在 config.json 的 openai.api_key 中配置，或在环境变量中设置 DASHSCOPE_API_KEY。"
        )

    dashscope.api_key = api_key
    os.environ["DASHSCOPE_API_KEY"] = api_key

    llm_cfg = {"model": model_name}

    system = (
        "你是一个兴趣点详情查询助手，使用 amap-maps MCP 服务的 maps_search_detail 能力。\n"
        "你的任务是：\n"
        "1. 根据给定的 POI id（例如 B0FFHVFVO0），调用 maps_search_detail 工具获取该兴趣点的详细信息；\n"
        "2. 尽量返回结构化的 JSON，包含名称、地址、类型、电话、评分、营业时间、坐标等字段；\n"
        "3. 如果工具返回的是原始 JSON 字符串，请直接输出该 JSON 字符串，不要丢失字段；\n"
        "4. 如果查询失败或找不到该 id，请明确说明原因。"
    )

    tools = [
        {
            "mcpServers": {
                "amap-maps": {
                    "url": "https://dashscope.aliyuncs.com/api/v1/mcps/amap-maps/sse",
                    "headers": {
                        "Authorization": f"Bearer {api_key}",
                    },
                }
            }
        }
    ]

    bot = Assistant(
        llm=llm_cfg,
        name="POI 详情查询智能体",
        description="根据 POI id 查询兴趣点详细信息",
        system_message=system,
        function_list=tools,
    )
    return bot


def build_keyword_extractor() -> Assistant:
    """
    构建一个用于提取搜索关键词的智能体。
    使用大模型从用户输入中提取搜索关键词。
    """
    cfg = load_config()
    openai_cfg = cfg.get("openai", {})
    
    model_name = openai_cfg.get("model", "qwen-max")
    api_key = openai_cfg.get("api_key") or os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "未找到 DashScope API Key。\n"
            "请在 config.json 的 openai.api_key 中配置，或在环境变量中设置 DASHSCOPE_API_KEY。"
        )
    
    dashscope.api_key = api_key
    os.environ["DASHSCOPE_API_KEY"] = api_key
    
    llm_cfg = {"model": model_name}
    
    system = (
        "你是一个关键词提取助手。\n"
        "你的任务是从用户的自然语言输入中提取搜索关键词。\n"
        "例如：\n"
        "- \"想吃火锅\" -> \"火锅\"\n"
        "- \"想找加油站\" -> \"加油站\"\n"
        "- \"需要服务区\" -> \"服务区\"\n"
        "- \"附近有餐厅吗\" -> \"餐厅\"\n"
        "请只返回提取出的关键词，不要返回其他内容。如果无法提取出有意义的关键词，返回空字符串。"
    )
    
    bot = Assistant(
        llm=llm_cfg,
        name="关键词提取智能体",
        description="从用户输入中提取搜索关键词",
        system_message=system,
    )
    return bot


def extract_search_keywords_from_user_input(user_input: str) -> Optional[str]:
    """
    使用大模型从用户输入中提取搜索关键词（如"想吃火锅" -> "火锅"）。
    如果用户输入中包含明显的需求关键词，返回该关键词；否则返回 None。
    """
    if not user_input or len(user_input.strip()) < 2:
        return None
    
    try:
        # 创建关键词提取智能体
        keyword_bot = build_keyword_extractor()
        
        extract_messages = [
            {
                "role": "user",
                "content": f"请从以下用户输入中提取搜索关键词：\n{user_input}"
            }
        ]
        
        # 调用大模型提取关键词
        responses = run_agent_with_retry(
            keyword_bot,
            extract_messages,
            max_retries=1,
            step_name="提取搜索关键词"
        )
        
        if responses:
            last = responses[-1]
            content = ""
            if isinstance(last, list):
                for item in last:
                    if isinstance(item, dict) and item.get("role") == "assistant" and "content" in item:
                        content = item["content"]
            elif isinstance(last, dict) and "content" in last:
                content = last["content"]
            
            # 清理提取的关键词
            keyword = content.strip()
            # 移除可能的引号、标点等
            keyword = re.sub(r'^["\'「」『』]|["\'「」『』]$', '', keyword)
            keyword = keyword.strip()
            
            # 验证关键词是否有效
            if keyword and len(keyword) > 0 and keyword not in ["的", "什么", "哪里", "没有", "无法"]:
                return keyword
        
        return None
    except Exception as e:
        logger.warning(f"关键词提取失败，使用原始输入: {e}")
        # 如果提取失败，返回原始输入
        return user_input.strip() if user_input.strip() else None


def build_poi_intent_extractor() -> Assistant:
    """
    构建一个用于识别用户 POI 意图的智能体。
    输出严格 JSON，包含关键词、类别等关键信息，供后续检索使用。
    """
    cfg = load_config()
    openai_cfg = cfg.get("openai", {})

    model_name = openai_cfg.get("model", "qwen-max")
    api_key = openai_cfg.get("api_key") or os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "未找到 DashScope API Key。\n"
            "请在 config.json 的 openai.api_key 中配置，或在环境变量中设置 DASHSCOPE_API_KEY。"
        )

    dashscope.api_key = api_key
    os.environ["DASHSCOPE_API_KEY"] = api_key

    llm_cfg = {"model": model_name}

    system = (
        "你是一个 POI 意图识别助手，只输出 JSON。\n"
        "输入：用户的自然语言需求（例如“沿途找家川菜馆，最好24小时”）。\n"
        "输出：严格的 JSON 对象，字段：\n"
        "- keyword: 必填，主要检索关键词（如“川菜馆”“加油站”）。\n"
        "- categories: 可选，细分类型数组（如[\"川菜\",\"火锅\"]）。\n"
        "- cuisine: 可选，菜系或口味关键词数组。\n"
        "- service: 可选，服务诉求（如[\"24小时\",\"停车\"]）。\n"
        "- note: 可选，其他备注。\n"
        "禁止输出除 JSON 外的任何文字。"
    )

    bot = Assistant(
        llm=llm_cfg,
        name="POI 意图识别助手",
        description="提取兴趣点检索关键信息",
        system_message=system,
    )
    return bot


def extract_poi_intent_from_user_input(user_input: str) -> Optional[Dict[str, Any]]:
    """
    使用大模型识别用户 POI 意图，提取关键词与类别等信息。
    返回 dict: {"keyword": str, "categories": [...], "cuisine": [...], "service": [...], "note": "..."}
    """
    if not user_input or len(user_input.strip()) < 2:
        return None
    try:
        bot = build_poi_intent_extractor()
        messages = [
            {
                "role": "user",
                "content": f"请从以下用户输入中提取 POI 检索信息，并仅输出 JSON：\n{user_input}",
            }
        ]
        responses = run_agent_with_retry(
            bot,
            messages,
            max_retries=1,
            step_name="提取 POI 意图",
        )
        if not responses:
            return None
        last = responses[-1]
        content = ""
        if isinstance(last, list):
            for item in last:
                if isinstance(item, dict) and item.get("role") == "assistant" and "content" in item:
                    content = item["content"]
        elif isinstance(last, dict) and "content" in last:
            content = last["content"]
        if not content:
            return None

        cleaned = content.strip()
        # 去除可能的代码块包装
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            if len(lines) > 2:
                cleaned = "\n".join(lines[1:-1])

        obj = json.loads(cleaned)
        if not isinstance(obj, dict):
            return None

        # 规范字段
        intent: Dict[str, Any] = {}
        keyword = obj.get("keyword")
        if isinstance(keyword, str) and keyword.strip():
            intent["keyword"] = keyword.strip()
        for key in ["categories", "cuisine", "service"]:
            val = obj.get(key)
            if isinstance(val, list):
                intent[key] = [str(v).strip() for v in val if str(v).strip()]
        note = obj.get("note")
        if isinstance(note, str) and note.strip():
            intent["note"] = note.strip()

        if intent:
            return intent
        return None
    except Exception as e:
        logger.warning(f"POI 意图提取失败: {e}")
        return None


def build_waypoint_extractor() -> Assistant:
    """
    构建一个"解析高德路线JSON并抽取关键节点"的智能体：
    - 输入：高德路线规划原始 JSON（含 route.paths[].steps[].polyline / road / navi 等字段），以及可选的出行方式提示；
    - 输出：严格的 JSON 数组，顺序与行驶顺序一致，包含 name/lon/lat/type/note 等字段；
    - 只做结构化提取，不调用任何外部工具。
    - 适用于驾车和步行路线。
    """
    cfg = load_config()
    openai_cfg = cfg.get("openai", {})
    
    model_name = openai_cfg.get("model", "qwen-max")
    api_key = openai_cfg.get("api_key") or os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "未找到 DashScope API Key。\n"
            "请在 config.json 的 openai.api_key 中配置，或在环境变量中设置 DASHSCOPE_API_KEY。"
        )

    dashscope.api_key = api_key
    os.environ["DASHSCOPE_API_KEY"] = api_key

    llm_cfg = {"model": model_name}

    system = (
        "你是路线节点抽取助手。输入是高德路线规划 API 的原始 JSON（route.paths[].steps[]）。请：\n"
        "1) 不调用任何工具；只阅读给定 JSON。\n"
        "2) 按行驶顺序抽取关键节点：起点、每个 step 的起点/终点、转向点、汇入/驶出高速等；\n"
        "3) 尽量使用 step.navi 或 step.road 作为名称；若缺失，用坐标字符串作为占位；\n"
        "4) 输出严格的 JSON 数组，每个元素包含：name、lon、lat、type（如 segment_start/segment_end/turn/intersection/merge/exit）、note（可空）；\n"
        "5) 只输出 JSON 数组，不要多余文字。"
    )

    bot = Assistant(
        llm=llm_cfg,
        name="路线节点抽取助手",
        description="解析高德路线JSON并输出关键节点",
        system_message=system,
    )
    return bot


def build_transit_waypoint_extractor() -> Assistant:
    """
    构建一个专门用于公共交通路线的节点抽取智能体：
    - 输入：高德公交路线规划原始 JSON（route.transits[].segments[]）
    - 输出：严格的 JSON 数组，按出行顺序包含关键节点
    - 提取规则：
      * 步行部分：提取转弯点作为中间节点
      * 公交部分：不获取中间节点，只保留出发站和到达站
      * 地铁部分：将经停点作为中间节点
    """
    cfg = load_config()
    openai_cfg = cfg.get("openai", {})
    
    model_name = openai_cfg.get("model", "qwen-max")
    api_key = openai_cfg.get("api_key") or os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "未找到 DashScope API Key。\n"
            "请在 config.json 的 openai.api_key 中配置，或在环境变量中设置 DASHSCOPE_API_KEY。"
        )

    dashscope.api_key = api_key
    os.environ["DASHSCOPE_API_KEY"] = api_key

    llm_cfg = {"model": model_name}

    system = (
        "你是公共交通路线节点抽取助手。输入是高德公交路线规划 API 的原始 JSON（route.transits[].segments[]）。\n"
        "请严格按照以下规则提取节点：\n"
        "\n"
        "1) 不调用任何工具；只阅读给定 JSON。\n"
        "\n"
        "2) 按出行顺序遍历 transits[0].segments[] 数组，对每个 segment 根据其类型处理：\n"
        "   - 步行段（walking）：提取起点、所有转向点（turn）、终点，使用 segment.walking.steps[] 中的信息\n"
        "   - 公交段（bus）：提取出发站（segment.bus.departure_stop）、所有途经站（segment.bus.via_stops[]）、到达站（segment.bus.arrival_stop）\n"
        "   - 地铁段（subway）：提取出发站、所有经停站（segment.subway.stops[] 或 via_stops[]）、到达站\n"
        "   - 其他类型（如出租车等）：提取起点和终点\n"
        "\n"
        "3) 节点类型（type）说明：\n"
        "   - segment_start：路段起点（起点、公交/地铁出发站）\n"
        "   - segment_end：路段终点（终点、公交/地铁到达站）\n"
        "   - turn：转向点（仅用于步行段）\n"
        "   - intersection：交叉路口（仅用于步行段）\n"
        "   - via_stop：途经站（仅用于地铁段）\n"
        "\n"
        "4) 节点信息：\n"
        "   - name：站点名称或道路名称\n"
        "   - lon、lat：经纬度坐标（必须提供）\n"
        "   - type：节点类型\n"
        "   - note：备注信息（如\"公交出发站\"、\"公交到达站\"、\"地铁经停站\"、\"步行XX米左转\"等）\n"
        "\n"
        "5) 输出严格的 JSON 数组，按实际出行顺序排列，不要多余文字。\n"
        "\n"
        "重要：公交段（bus）必须提取所有途经站（via_stops），这样可以获得更多中间节点，用于沿途兴趣点搜索！"
    )

    bot = Assistant(
        llm=llm_cfg,
        name="公交路线节点抽取助手",
        description="解析高德公交路线JSON并输出关键节点（步行段提取转弯点，公交段只保留起终点，地铁段提取经停站）",
        system_message=system,
    )
    return bot


def build_route_summary_agent() -> Assistant:
    """构建一个用于将高德路线 JSON 概述成中文摘要的智能体。"""
    cfg = load_config()
    openai_cfg = cfg.get("openai", {})

    model_name = openai_cfg.get("model", "qwen-max")
    api_key = openai_cfg.get("api_key") or os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "未找到 DashScope API Key。\n"
            "请在 config.json 的 openai.api_key 中配置，或在环境变量中设置 DASHSCOPE_API_KEY。"
        )

    dashscope.api_key = api_key
    os.environ["DASHSCOPE_API_KEY"] = api_key

    llm_cfg = {"model": model_name}

    system = (
        "你是路线摘要助手。输入是高德路线规划 API 的原始 JSON（route.paths[].steps[]）。\n"
        "请用简体中文输出精简摘要，包含：\n"
        "1) 总距离与预计时间；\n"
        "2) 主要经过的道路/高速或关键区域；\n"
        "3) 如有多条 path，只基于首条 path。\n"
        "保持简洁，不要输出原始 JSON。"
    )

    bot = Assistant(
        llm=llm_cfg,
        name="路线摘要助手",
        description="将高德路线JSON转成中文概要",
        system_message=system,
    )
    return bot




