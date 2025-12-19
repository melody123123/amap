import io
import json
import os
import sys
import time
import re
import math
import contextlib
from urllib.parse import quote, unquote, parse_qs, urlparse, urlencode
from pathlib import Path
from typing import List, Dict, Optional, Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from amap_mcp_agent import (
    build_waypoint_extractor,
    build_transit_waypoint_extractor,
    build_poi_search_agent,
    build_poi_detail_agent,
    build_chat_poi_agent,
    extract_waypoint_names_from_json,
    extract_search_keywords_from_user_input,
    run_agent_with_retry,
    build_keyword_extractor,
    build_route_summary_agent,
)
from amap_tools import AMapClient, extract_lonlat_from_geocode


app = FastAPI(title="Amap MCP Route Agent API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    # 允许任意前端来源访问（包括 file:// 打开的本地页面，origin 为 null）
    allow_origins=["*"],
    allow_credentials=False,  # 当 allow_origins 为 "*" 时，不能同时为 True
    allow_methods=["*"],
    allow_headers=["*"],
)


class RouteRequest(BaseModel):
    start: str
    end: str
    poi: Optional[str] = None
    transport_mode: Optional[str] = "driving"  # 出行方式：walking（步行）、driving（驾车）、transit（公共交通）
    # 可选：前端上传的用户真实 IP，用于辅助地理编码（提高地址歧义场景下的准确率）
    user_ip: Optional[str] = None


class RouteResponse(BaseModel):
    summary: str
    waypoints: Optional[List]
    # 优化：pois 改为标准 JSON 结构，避免前端再去解析自然语言文本
    # 结构为 [{"name": "...", "distance": "...", "id": "..."}, ...]
    pois: Optional[List[Dict[str, str]]] = None
    pois_ids: Optional[List[str]] = None
    logs: str
    # 公交路线详细信息（仅当 transport_mode == "transit" 时返回）
    transit_info: Optional[Dict[str, Any]] = None


class PoiSearchRequest(BaseModel):
    start: str
    end: str
    poi: str
    transport_mode: Optional[str] = "driving"  # 出行方式：walking（步行）、driving（驾车）、transit（公共交通）


class PoiSearchResponse(BaseModel):
    pois: Optional[List[Dict[str, str]]] = None
    pois_ids: Optional[List[str]] = None
    logs: str


class ChatPoiRequest(BaseModel):
    start: str
    end: str
    message: str
    transport_mode: Optional[str] = "driving"
    conversation_history: Optional[List[Dict[str, str]]] = []  # 对话历史
    session_id: Optional[str] = None  # 会话ID（用于途经点管理）


class ChatPoiResponse(BaseModel):
    reply: str  # AI 回复
    pois: Optional[List[Dict[str, str]]] = None  # 搜索到的兴趣点
    pois_ids: Optional[List[str]] = None
    extracted_keywords: Optional[str] = None  # 提取的关键词
    conversation_history: List[Dict[str, str]]  # 更新后的对话历史
    action: Optional[str] = None  # 动作类型：'search'（搜索POI）、'add_waypoint'（添加途经点）
    waypoint_added: Optional[str] = None  # 如果添加了途经点，显示途经点名称
    logs: str


class PoiDetailRequest(BaseModel):
    poi_id: str
    start: Optional[str] = None
    end: Optional[str] = None
    transport_mode: Optional[str] = "driving"  # 出行方式：walking（步行）、driving（驾车）、transit（公共交通）


class PoiDetailResponse(BaseModel):
    detail: str
    extra_distance_km: Optional[float] = None
    via_route_summary: Optional[str] = None  # 起点-POI-终点 路线摘要
    via_route: Optional[Dict[str, Any]] = None  # 可选，返回路线 JSON（可能较大）
    via_nav_url: Optional[str] = None  # 高德一键导航URI，含途经点（公交模式下为None）
    # 公交模式专用字段
    alight_stop_name: Optional[str] = None  # 下车站点名称（仅公交模式）
    walk_nav_url: Optional[str] = None  # 从下车站点到POI的步行导航链接（仅公交模式）
    show_add_to_route: bool = True  # 是否显示"加入路线"按钮（公交模式为False）
    logs: str


# ===== 途经点管理相关模型 =====
class Waypoint(BaseModel):
    """途经点数据模型"""
    name: str  # 途经点名称/地址
    location: Optional[str] = None  # 坐标 'lon,lat'
    poi_id: Optional[str] = None  # POI ID（如果是从POI搜索添加的）
    order: int = 0  # 顺序（用于排序）


class AddWaypointRequest(BaseModel):
    """添加途经点请求"""
    session_id: str  # 会话ID
    waypoint_name: str  # 途经点名称/地址
    poi_id: Optional[str] = None  # 可选的POI ID
    start: Optional[str] = None  # 起点（用于坐标解析）
    end: Optional[str] = None  # 终点（用于坐标解析）
    transport_mode: Optional[str] = None  # 出行方式，用于判断是否支持多途经点（优先使用前端传入）


class AddWaypointResponse(BaseModel):
    """添加途经点响应"""
    success: bool
    message: str
    waypoints: List[Waypoint]  # 当前所有途经点
    logs: str


class RemoveWaypointRequest(BaseModel):
    """删除途经点请求"""
    session_id: str
    waypoint_index: int  # 要删除的途经点索引


class RemoveWaypointResponse(BaseModel):
    """删除途经点响应"""
    success: bool
    message: str
    waypoints: List[Waypoint]


class ListWaypointsRequest(BaseModel):
    """列出途经点请求"""
    session_id: str


class ListWaypointsResponse(BaseModel):
    """列出途经点响应"""
    waypoints: List[Waypoint]
    start: Optional[str] = None
    end: Optional[str] = None


class OptimizeRouteRequest(BaseModel):
    """优化路线请求（重排序途经点）"""
    session_id: str
    start: str
    end: str
    transport_mode: Optional[str] = "driving"


class OptimizeRouteResponse(BaseModel):
    """优化路线响应"""
    success: bool
    message: str
    optimized_waypoints: List[Waypoint]  # 优化后的途经点顺序
    route_summary: Optional[str] = None  # 路线摘要
    total_distance_km: Optional[float] = None  # 总距离
    total_duration_min: Optional[float] = None  # 总时间
    nav_url: Optional[str] = None  # 导航链接
    logs: str


# ===== 简单的本地缓存：长期保存同一 A-B 路径的路线与拐角点 =====
CACHE_VERSION = 2  # 用于简单的缓存兼容检查
_CACHE_FILE = Path(__file__).with_name("route_cache.json")
_EXPECTED_CACHE_KEYS = {"route_text", "waypoints", "distance_m", "start_loc", "end_loc", "cache_version"}

try:
    if _CACHE_FILE.exists():
        raw_cache = json.loads(_CACHE_FILE.read_text(encoding="utf-8"))
        _ROUTE_CACHE: Dict[str, Dict[str, object]] = {}
        invalid_count = 0
        for k, v in raw_cache.items():
            if isinstance(v, dict) and v.get("cache_version") == CACHE_VERSION:
                _ROUTE_CACHE[k] = v
            else:
                invalid_count += 1
        # 如果缓存大部分不兼容，直接清空
        if invalid_count and invalid_count >= 3:
            _ROUTE_CACHE = {}
    else:
        _ROUTE_CACHE = {}
except Exception:
    # 若缓存文件损坏，则从空缓存开始
    _ROUTE_CACHE = {}


# ===== 途经点会话管理：内存存储（可扩展为Redis等） =====
_WAYPOINT_SESSIONS: Dict[str, Dict[str, Any]] = {}
# 结构：{session_id: {"start": str, "end": str, "waypoints": [Waypoint], "created_at": timestamp}}


def _get_or_create_session(session_id: str) -> Dict[str, Any]:
    """获取或创建会话"""
    if session_id not in _WAYPOINT_SESSIONS:
        _WAYPOINT_SESSIONS[session_id] = {
            "start": None,
            "end": None,
            "waypoints": [],
            "created_at": time.time(),
        }
    return _WAYPOINT_SESSIONS[session_id]


def _cleanup_old_sessions(max_age_seconds: int = 3600 * 24) -> None:
    """清理超过24小时的旧会话"""
    current_time = time.time()
    expired_sessions = [
        sid for sid, sess in _WAYPOINT_SESSIONS.items()
        if current_time - sess.get("created_at", 0) > max_age_seconds
    ]
    for sid in expired_sessions:
        del _WAYPOINT_SESSIONS[sid]
    if expired_sessions:
        print(f"[会话清理] 清理了 {len(expired_sessions)} 个过期会话")


def _make_route_key(start: str, end: str, transport_mode: str = "driving") -> str:
    """根据起终点和出行方式生成缓存 key（去掉首尾空格，统一大小写）。"""
    return f"{start.strip().lower()}||{end.strip().lower()}||{transport_mode}"


def _save_route_cache() -> None:
    """将内存中的缓存写回到本地 JSON 文件。"""
    try:
        _CACHE_FILE.write_text(
            json.dumps(_ROUTE_CACHE, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception:
        # 缓存写入失败不影响主流程
        pass


def _extract_assistant_content(responses: List) -> str:
    """从 qwen-agent 的响应中提取最后一条 assistant 文本内容。"""
    if not responses:
        return ""
    last = responses[-1]
    content = ""
    if isinstance(last, list):
        for item in last:
            if isinstance(item, dict) and item.get("role") == "assistant" and "content" in item:
                content = item["content"]
    elif isinstance(last, dict) and "content" in last:
        content = last["content"]
    return content or ""


def _calculate_straight_distance_km(loc1: str, loc2: str) -> float:
    """
    计算两点之间的直线距离（公里），使用 Haversine 公式。
    
    Args:
        loc1: 坐标字符串 'lon,lat'
        loc2: 坐标字符串 'lon,lat'
    
    Returns:
        直线距离（公里）
    """
    try:
        lon1, lat1 = map(float, loc1.split(','))
        lon2, lat2 = map(float, loc2.split(','))
        
        # Haversine 公式计算地球表面两点间距离
        R = 6371.0  # 地球半径（公里）
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
        c = 2 * math.asin(math.sqrt(a))
        distance_km = R * c
        
        return distance_km
    except (ValueError, IndexError) as e:
        print(f"[距离预检查] 坐标解析失败：{e}")
        return float('inf')  # 解析失败时返回无穷大，让后续检查失败


def _check_route_distance_limit(start_loc: str, end_loc: str, transport_mode: str) -> Optional[str]:
    """
    预检查路线距离是否超出API限制。
    
    Args:
        start_loc: 起点坐标 'lon,lat'
        end_loc: 终点坐标 'lon,lat'
        transport_mode: 出行方式
    
    Returns:
        如果超出限制，返回错误提示信息；否则返回 None
    """
    try:
        distance_km = _calculate_straight_distance_km(start_loc, end_loc)
        
        # 根据出行方式设置不同的距离限制
        if transport_mode == "transit":
            # 公交路线规划限制：通常50-100公里
            limit_km = 80
            if distance_km > limit_km:
                return f"起点和终点直线距离约 {distance_km:.1f} 公里，超过公交路线规划限制（{limit_km} 公里），无法计算绕路距离"
        elif transport_mode == "walking":
            # 步行路线规划限制：通常10-20公里
            limit_km = 15
            if distance_km > limit_km:
                return f"起点和终点直线距离约 {distance_km:.1f} 公里，超过步行路线规划限制（{limit_km} 公里），无法计算绕路距离"
        # 驾车路线规划通常没有严格限制，但超过500公里可能也有问题
        elif transport_mode == "driving":
            limit_km = 500
            if distance_km > limit_km:
                return f"起点和终点直线距离约 {distance_km:.1f} 公里，距离过远，可能无法计算精确的绕路距离"
        
        print(f"[距离预检查] 起点到终点直线距离：{distance_km:.1f} 公里，在限制范围内")
        return None
    except Exception as e:
        print(f"[距离预检查] 距离检查失败：{e}，继续尝试API调用")
        return None  # 检查失败时，不阻止API调用


def _extract_poi_location(detail_text: str) -> Optional[str]:
    """从 POI 详情文本中尝试解析出 location 字段（'lon,lat'）。"""
    print(f"[POI坐标提取] 开始提取POI坐标，文本长度：{len(detail_text)}")
    cleaned = detail_text.strip()
    # 去掉可能的 ``` 或 ```json 代码块包装
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        if len(lines) > 2:
            cleaned = "\n".join(lines[1:-1])
            print(f"[POI坐标提取] 已移除代码块标记")
    
    try:
        obj = json.loads(cleaned)
        print(f"[POI坐标提取] JSON解析成功，类型：{type(obj)}")
    except Exception as e:
        print(f"[POI坐标提取] JSON解析失败：{e}")
        return None

    if isinstance(obj, dict):
        loc = obj.get("location")
        print(f"[POI坐标提取] 从根对象获取location：{loc}")
        if isinstance(loc, str) and "," in loc:
            print(f"[POI坐标提取] 成功提取坐标：{loc}")
            return loc
        # 有些返回结构里可能是 {"pois":[{...}]}
        pois = obj.get("pois") or []
        print(f"[POI坐标提取] 检查pois数组，长度：{len(pois)}")
        if isinstance(pois, list) and pois:
            first = pois[0]
            if isinstance(first, dict):
                loc = first.get("location")
                print(f"[POI坐标提取] 从pois[0]获取location：{loc}")
                if isinstance(loc, str) and "," in loc:
                    print(f"[POI坐标提取] 成功提取坐标：{loc}")
                    return loc
    
    print(f"[POI坐标提取] 未能提取到有效坐标")
    return None


def _safe_route_distance_m(client: AMapClient, origin: str, destination: str, waypoints: Optional[str] = None, transport_mode: str = "driving", strategy: int = 4, max_retries: int = 3) -> float:
    """
    调用高德路线规划 API，返回两点间最短路径距离（米）。
    支持途经点，使用途经点API可以计算完整的优化路径，而不是分段距离的简单相加。
    支持不同的出行方式：驾车、步行、公共交通。
    带重试机制，处理并发限制错误。
    
    Args:
        client: AMapClient 实例
        origin: 起点坐标 'lon,lat'
        destination: 终点坐标 'lon,lat'
        waypoints: 途经点坐标 'lon,lat'，如果为None则不使用途经点
        transport_mode: 出行方式，'driving'（驾车）、'walking'（步行）、'transit'（公共交通）
        strategy: 路线规划策略（仅驾车时有效），默认4（躲避拥堵），与高德地图应用保持一致
        max_retries: 最大重试次数
    """
    for attempt in range(max_retries):
        try:
            # 根据出行方式选择对应的API
            if transport_mode == "walking":
                data = client.walking_route(origin, destination, waypoints=waypoints)
            elif transport_mode == "transit":
                # 公交路线规划不支持途经点
                if waypoints:
                    print(f"[绕路距离计算] 警告：公交路线规划不支持途经点，将忽略途经点")
                data = client.transit_route(origin, destination)
            else:  # 默认为驾车
                data = client.driving_route(origin, destination, waypoints=waypoints, strategy=strategy)
            route = data.get("route") or {}
            # 公交路线返回 transits，驾车/步行返回 paths
            if transport_mode == "transit":
                transits = route.get("transits") or []
                if not transits:
                    raise ValueError("no route transits")
                distance_str = transits[0].get("distance")
                if not distance_str:
                    raise ValueError("no distance field in transit")
                return float(distance_str)
            else:
                paths = route.get("paths") or []
                if not paths:
                    raise ValueError("no route paths")
                distance_str = paths[0].get("distance")
                if not distance_str:
                    raise ValueError("no distance field")
                return float(distance_str)
        except (RuntimeError, ValueError) as e:
            error_msg = str(e)
            # 检查是否是并发限制错误
            if "并发" in error_msg or "concurrent" in error_msg.lower() or "over limit" in error_msg.lower() or "请求超限" in error_msg:
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 0.5  # 递增延迟：0.5s, 1s, 1.5s
                    print(f"[绕路距离计算] 遇到并发限制，等待 {wait_time} 秒后重试 ({attempt + 1}/{max_retries})...")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"[绕路距离计算] 并发限制，已重试 {max_retries} 次，放弃")
                    raise
            else:
                # 其他错误直接抛出
                raise
    raise ValueError("Failed to get route distance after retries")


def _compute_extra_distance_km(
    start: Optional[str],
    end: Optional[str],
    poi_location: Optional[str],
    transport_mode: str = "driving",
    cached_base_m: Optional[float] = None,
    cached_start_loc: Optional[str] = None,
    cached_end_loc: Optional[str] = None,
    client: Optional[AMapClient] = None,
) -> Optional[float]:
    """
    计算：从 start 直接到 end 与 从 start 途经 poi 再到 end 两种路线的里程差（公里）。
    start/end 为地址字符串，poi_location 为 'lon,lat'。
    transport_mode: 出行方式，'driving'（驾车）、'walking'（步行）、'transit'（公共交通）
    """
    print(f"[绕路距离计算] 输入参数检查：start={start}, end={end}, poi_location={poi_location}")
    
    # 验证输入参数
    if not start or not end or not poi_location or "," not in poi_location:
        print(f"[绕路距离计算] 参数验证失败：start={bool(start)}, end={bool(end)}, poi_location={bool(poi_location)}, poi_location格式={'有效' if poi_location and ',' in poi_location else '无效'}")
        return None
    
    # 验证地址字符串是否有效（去除空白后不为空）
    start = start.strip()
    end = end.strip()
    if not start or not end:
        print(f"[绕路距离计算] 地址字符串验证失败：起点='{start}', 终点='{end}'")
        return None
    
    print(f"[绕路距离计算] 参数验证通过，开始计算...")

    try:
        print(f"[绕路距离计算] 检查 AMAP_API_KEY 配置...")
        # 如果外部传入 client，则重用；否则自行创建
        if client is None:
            # 检查是否有高德 Web API Key（用于直接调用高德服务）
            # 优先从 config.json 读取，其次从环境变量读取
            amap_api_key = None
            config_path = Path(__file__).with_name("config.json")
            if config_path.exists():
                try:
                    with config_path.open("r", encoding="utf-8") as f:
                        config = json.load(f)
                        amap_cfg = config.get("amap", {})
                        if amap_cfg and amap_cfg.get("api_key"):
                            amap_api_key = amap_cfg["api_key"]
                except (json.JSONDecodeError, KeyError):
                    pass
            
            if not amap_api_key:
                amap_api_key = os.getenv("AMAP_API_KEY", "")
            
            if not amap_api_key:
                print(f"[绕路距离计算] 未配置 AMAP_API_KEY，跳过绕路距离计算")
                print(f"[绕路距离计算] 提示：如需计算绕路距离，请在 config.json 的 amap.api_key 中配置，或在 .env 文件中设置 AMAP_API_KEY")
                return None
            
            print(f"[绕路距离计算] AMAP_API_KEY 已配置，创建 AMapClient 实例...")
            client = AMapClient()
            print(f"[绕路距离计算] AMapClient 创建成功")

        # 起终点坐标：优先使用缓存，否则地理编码
        if cached_start_loc and "," in cached_start_loc:
            start_loc = cached_start_loc
            print(f"[绕路距离计算] 使用缓存起点坐标：{start_loc}")
        else:
            try:
                print(f"[绕路距离计算] 开始地理编码起点：{start}")
                start_geo = client.geocode(start)
                start_loc = extract_lonlat_from_geocode(start_geo)
                print(f"[绕路距离计算] 起点地理编码成功：{start_loc}")
            except (ValueError, RuntimeError, KeyError) as e:
                print(f"[绕路距离计算] 起点地理编码失败：{start}，错误：{e}")
                return None
        
        if cached_end_loc and "," in cached_end_loc:
            end_loc = cached_end_loc
            print(f"[绕路距离计算] 使用缓存终点坐标：{end_loc}")
        else:
            try:
                print(f"[绕路距离计算] 开始地理编码终点：{end}")
                end_geo = client.geocode(end)
                end_loc = extract_lonlat_from_geocode(end_geo)
                print(f"[绕路距离计算] 终点地理编码成功：{end_loc}")
            except (ValueError, RuntimeError, KeyError) as e:
                print(f"[绕路距离计算] 终点地理编码失败：{end}，错误：{e}")
                return None

        # 验证坐标格式
        if not start_loc or "," not in start_loc or not end_loc or "," not in end_loc:
            print(f"[绕路距离计算] 坐标格式验证失败：start={start_loc}, end={end_loc}")
            return None

        # 距离预检查：在调用API之前检查距离是否超出限制
        print(f"[绕路距离计算] 进行距离预检查...")
        distance_check_result = _check_route_distance_limit(start_loc, end_loc, transport_mode)
        if distance_check_result:
            print(f"[绕路距离计算] {distance_check_result}")
            return None  # 距离超出限制，直接返回，不调用API

        print(f"[绕路距离计算] 开始计算路线距离（出行方式：{transport_mode}）...")
        if cached_base_m is not None:
            base_m = cached_base_m
            print(f"[绕路距离计算] 使用缓存的基础距离：{base_m} 米")
        else:
            print(f"[绕路距离计算] 计算起点到终点距离（直接路线）...")
            try:
                base_m = _safe_route_distance_m(client, start_loc, end_loc, waypoints=None, transport_mode=transport_mode)
                print(f"[绕路距离计算] 起点到终点距离：{base_m} 米")
            except ValueError as e:
                if transport_mode == "transit" and "no route transits" in str(e):
                    print(f"[绕路距离计算] 起点到终点之间无公交路线，无法计算绕路距离")
                    return None
                else:
                    raise
        
        # 添加延迟，避免并发请求过快
        time.sleep(0.3)  # 延迟300毫秒

        # 使用途经点API计算完整的"起点→POI→终点"路径距离
        # 这样计算的是优化后的完整路径，而不是两段距离的简单相加
        # 注意：公交路线规划不支持途经点，如果选择公交，将使用分段计算
        print(f"[绕路距离计算] 计算途经POI的完整路径距离...")
        if transport_mode == "transit":
            # 公交路线绕路逻辑：从中转站步行到兴趣点
            print(f"[绕路距离计算] 公交路线绕路逻辑：从中转站步行到兴趣点...")
            
            # 步骤1：获取起点到终点的公交路线，找到公交站
            print(f"[绕路距离计算] 获取起点到终点的公交路线...")
            try:
                transit_route_data = client.transit_route(start_loc, end_loc)
                route = transit_route_data.get("route") or {}
                transits = route.get("transits") or []
                if not transits:
                    print(f"[绕路距离计算] 无法获取公交路线，降级为步行计算...")
                    # 降级为步行计算前，先检查分段距离
                    start_to_poi_check = _check_route_distance_limit(start_loc, poi_location, "walking")
                    poi_to_end_check = _check_route_distance_limit(poi_location, end_loc, "walking")
                    if start_to_poi_check or poi_to_end_check:
                        print(f"[绕路距离计算] 分段距离超出步行限制：{start_to_poi_check or poi_to_end_check}")
                        return None
                    # 降级为步行计算
                    via_total_m = _safe_route_distance_m(client, start_loc, poi_location, waypoints=None, transport_mode="walking")
                    time.sleep(0.3)
                    via_total_m += _safe_route_distance_m(client, poi_location, end_loc, waypoints=None, transport_mode="walking")
                else:
                    # 步骤2：从公交路线中提取所有公交站（包括出发站、途经站、到达站）
                    bus_stops = []
                    transit = transits[0]
                    segments = transit.get("segments") or []
                    for segment in segments:
                        # 提取公交段的站点
                        if "bus" in segment:
                            bus = segment["bus"]
                            # 出发站
                            departure = bus.get("departure_stop", {})
                            if departure.get("location"):
                                bus_stops.append(departure.get("location"))
                            # 途经站（如果有）
                            via_stops = bus.get("via_stops") or []
                            for via_stop in via_stops:
                                if isinstance(via_stop, dict) and via_stop.get("location"):
                                    bus_stops.append(via_stop.get("location"))
                            # 到达站
                            arrival = bus.get("arrival_stop", {})
                            if arrival.get("location"):
                                bus_stops.append(arrival.get("location"))
                        # 提取地铁段的站点
                        elif "subway" in segment:
                            subway = segment["subway"]
                            # 出发站
                            departure = subway.get("departure_stop", {})
                            if departure.get("location"):
                                bus_stops.append(departure.get("location"))
                            # 经停站（如果有）
                            via_stops = subway.get("via_stops") or subway.get("stops") or []
                            for via_stop in via_stops:
                                if isinstance(via_stop, dict) and via_stop.get("location"):
                                    bus_stops.append(via_stop.get("location"))
                            # 到达站
                            arrival = subway.get("arrival_stop", {})
                            if arrival.get("location"):
                                bus_stops.append(arrival.get("location"))
                    
                    if not bus_stops:
                        print(f"[绕路距离计算] 公交路线中未找到公交站，降级为步行计算...")
                        # 降级为步行计算前，先检查分段距离
                        start_to_poi_check = _check_route_distance_limit(start_loc, poi_location, "walking")
                        poi_to_end_check = _check_route_distance_limit(poi_location, end_loc, "walking")
                        if start_to_poi_check or poi_to_end_check:
                            print(f"[绕路距离计算] 分段距离超出步行限制：{start_to_poi_check or poi_to_end_check}")
                            return None
                        via_total_m = _safe_route_distance_m(client, start_loc, poi_location, waypoints=None, transport_mode="walking")
                        time.sleep(0.3)
                        via_total_m += _safe_route_distance_m(client, poi_location, end_loc, waypoints=None, transport_mode="walking")
                    else:
                        # 步骤3：找到距离POI最近的公交站
                        print(f"[绕路距离计算] 找到 {len(bus_stops)} 个公交站，计算距离POI最近的站点...")
                        min_distance = float('inf')
                        nearest_stop = None
                        for stop_loc in bus_stops:
                            try:
                                # 计算POI到公交站的步行距离
                                dist = _safe_route_distance_m(client, poi_location, stop_loc, waypoints=None, transport_mode="walking")
                                if dist < min_distance:
                                    min_distance = dist
                                    nearest_stop = stop_loc
                            except Exception as e:
                                print(f"[绕路距离计算] 计算POI到公交站 {stop_loc} 距离失败：{e}")
                                continue
                            time.sleep(0.1)  # 避免请求过快
                        
                        if not nearest_stop:
                            print(f"[绕路距离计算] 无法找到最近的公交站，降级为步行计算...")
                            # 降级为步行计算前，先检查分段距离
                            start_to_poi_check = _check_route_distance_limit(start_loc, poi_location, "walking")
                            poi_to_end_check = _check_route_distance_limit(poi_location, end_loc, "walking")
                            if start_to_poi_check or poi_to_end_check:
                                print(f"[绕路距离计算] 分段距离超出步行限制：{start_to_poi_check or poi_to_end_check}")
                                return None
                            via_total_m = _safe_route_distance_m(client, start_loc, poi_location, waypoints=None, transport_mode="walking")
                            time.sleep(0.3)
                            via_total_m += _safe_route_distance_m(client, poi_location, end_loc, waypoints=None, transport_mode="walking")
                        else:
                            print(f"[绕路距离计算] 最近的公交站：{nearest_stop}，距离POI {min_distance:.0f} 米")
                            
                            # 步骤4：计算绕路距离
                            # 基础路线：起点 -> 最近公交站 -> 终点（公交）
                            # 绕路路线：起点 -> 最近公交站（公交）+ 最近公交站 -> POI（步行）+ POI -> 最近公交站（步行）+ 最近公交站 -> 终点（公交）
                            
                            # 计算起点到最近公交站的公交距离
                            print(f"[绕路距离计算] 计算起点到最近公交站的公交距离...")
                            try:
                                start_to_stop_m = _safe_route_distance_m(client, start_loc, nearest_stop, waypoints=None, transport_mode="transit")
                                print(f"[绕路距离计算] 起点到最近公交站（公交）：{start_to_stop_m} 米")
                            except ValueError as e:
                                if "no route transits" in str(e):
                                    print(f"[绕路距离计算] 起点到最近公交站无公交路线，使用步行...")
                                    start_to_stop_m = _safe_route_distance_m(client, start_loc, nearest_stop, waypoints=None, transport_mode="walking")
                                else:
                                    raise
                            
                            time.sleep(0.3)
                            
                            # 计算最近公交站到终点的公交距离
                            print(f"[绕路距离计算] 计算最近公交站到终点的公交距离...")
                            try:
                                stop_to_end_m = _safe_route_distance_m(client, nearest_stop, end_loc, waypoints=None, transport_mode="transit")
                                print(f"[绕路距离计算] 最近公交站到终点（公交）：{stop_to_end_m} 米")
                            except ValueError as e:
                                if "no route transits" in str(e):
                                    print(f"[绕路距离计算] 最近公交站到终点无公交路线，使用步行...")
                                    stop_to_end_m = _safe_route_distance_m(client, nearest_stop, end_loc, waypoints=None, transport_mode="walking")
                                else:
                                    raise
                            
                            # 绕路总距离 = 起点->公交站（公交）+ 公交站->POI（步行）+ POI->公交站（步行）+ 公交站->终点（公交）
                            # 注意：POI->公交站的距离等于公交站->POI的距离（往返）
                            via_total_m = start_to_stop_m + min_distance * 2 + stop_to_end_m
                            print(f"[绕路距离计算] 绕路总距离：起点->公交站({start_to_stop_m}m) + 公交站->POI往返({min_distance*2}m) + 公交站->终点({stop_to_end_m}m) = {via_total_m}m")
            except Exception as e:
                print(f"[绕路距离计算] 获取公交路线失败：{e}，降级为步行计算...")
                import traceback
                print(f"[绕路距离计算] 异常堆栈：{traceback.format_exc()}")
                # 降级为步行计算前，先检查分段距离
                start_to_poi_check = _check_route_distance_limit(start_loc, poi_location, "walking")
                poi_to_end_check = _check_route_distance_limit(poi_location, end_loc, "walking")
                if start_to_poi_check or poi_to_end_check:
                    print(f"[绕路距离计算] 分段距离超出步行限制：{start_to_poi_check or poi_to_end_check}")
                    return None
                # 降级为步行计算
                via_total_m = _safe_route_distance_m(client, start_loc, poi_location, waypoints=None, transport_mode="walking")
                time.sleep(0.3)
                via_total_m += _safe_route_distance_m(client, poi_location, end_loc, waypoints=None, transport_mode="walking")
        else:
            # 驾车和步行支持途经点：一次调用含途经点的完整路线
            print(f"[绕路距离计算] 使用途经点API：起点={start_loc}, 途经点={poi_location}, 终点={end_loc}")
            via_total_m = _safe_route_distance_m(client, start_loc, end_loc, waypoints=poi_location, transport_mode=transport_mode)
        print(f"[绕路距离计算] 途经POI的完整路径距离：{via_total_m} 米")

        extra_m = max(via_total_m - base_m, 0.0)
        extra_km = extra_m / 1000.0
        result = round(extra_km, 1)
        print(f"[绕路距离计算] 计算完成：绕路距离 = {result} 公里")
        print(f"[绕路距离计算] 说明：直接路线 {base_m/1000:.1f} 公里，途经POI路线 {via_total_m/1000:.1f} 公里")
        return result
    except Exception as e:
        print(f"[绕路距离计算] 发生异常：{type(e).__name__}: {e}")
        import traceback
        print(f"[绕路距离计算] 异常堆栈：{traceback.format_exc()}")
        return None


@app.post("/api/route", response_model=RouteResponse)
async def route_handler(req: RouteRequest, request: Request):
    """对外提供的路线规划 + 拐角点 + 兴趣点 一体化接口。"""
    log_buffer = io.StringIO()
    with contextlib.redirect_stdout(log_buffer):
        transport_mode = req.transport_mode or "driving"
        summary_bot = build_route_summary_agent()
        # 根据出行方式选择不同的节点提取器
        if transport_mode == "transit":
            waypoint_bot = build_transit_waypoint_extractor()
        else:
            waypoint_bot = build_waypoint_extractor()
        try:
            poi_bot = build_poi_search_agent()
        except Exception as e:  # MCP 服务不可用时，优雅降级，跳过兴趣点搜索
            print(f"[降级] 初始化 POI 搜索智能体失败，将跳过兴趣点搜索：{e}")
            poi_bot = None

        # 初始化高德客户端（直接用 Web API，跳过 MCP）
        try:
            amap_client = AMapClient()
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"高德 API Key 配置有误或不可用：{e}")

        # ===== 利用用户 IP + 大模型 辅助判断城市/行政区，提高地理编码准确率 =====
        ip_city_or_adcode: Optional[str] = None
        ip_city_human_readable: Optional[str] = None
        try:
            # 1. 优先使用前端显式上传的 user_ip
            candidate_ip = (req.user_ip or "").strip()
            
            # 2. 如果未上传，则尝试从常见代理头或连接信息中获取
            if not candidate_ip:
                xff = request.headers.get("x-forwarded-for") or request.headers.get("X-Forwarded-For")
                if xff:
                    # 可能是 "client, proxy1, proxy2" 形式，取第一个
                    candidate_ip = xff.split(",")[0].strip()
            if not candidate_ip and request.client:
                candidate_ip = request.client.host or ""
            
            # 3. 过滤常见内网 / 本地地址，避免误用服务器内网 IP 做定位
            def _is_private_ip(ip: str) -> bool:
                return (
                    ip.startswith("10.") or
                    ip.startswith("192.168.") or
                    ip.startswith("127.") or
                    ip.startswith("::1") or
                    ip.startswith("172.16.") or
                    ip.startswith("172.17.") or
                    ip.startswith("172.18.") or
                    ip.startswith("172.19.") or
                    ip.startswith("172.2")  # 粗略过滤 172.20.x.x ~ 172.29.x.x
                )
            
            if candidate_ip and not _is_private_ip(candidate_ip):
                try:
                    ip_info = amap_client.ip_location(candidate_ip)
                    # AMap IP API 返回的结构一般包含 province / city / adcode 等字段
                    ip_city_or_adcode = (
                        (ip_info.get("adcode") or "").strip()
                        or (ip_info.get("city") or "").strip()
                    ) or None
                    # 记录一个更可读的城市名称，用于给大模型参考，例如“西安市”
                    ip_city_human_readable = (
                        ip_info.get("city") or ip_info.get("province") or ""
                    ).strip() or None
                    if ip_city_or_adcode:
                        print(f"[IP定位] 使用 IP={candidate_ip} 辅助地理编码，城市/行政区：{ip_city_or_adcode}（{ip_city_human_readable or '未知'}）")
                except Exception as e:
                    # IP 定位失败不影响主流程，只打印日志
                    print(f"[IP定位] 通过 IP={candidate_ip} 获取行政区失败：{e}")
        except Exception as e:
            # 防御性兜底，确保任何异常都不会中断主流程
            print(f"[IP定位] 处理用户 IP 时发生异常：{e}")

        # ===== 使用大模型辅助判断：IP 城市与起点/终点是否一致 =====
        # 只在我们拿到了 IP 城市的情况下，才调用一次摘要助手做轻量判断。
        if ip_city_or_adcode and ip_city_human_readable:
            try:
                # 这里复用路线摘要助手的 LLM 能力，做一个极简的结构化判断：
                # 输入：起点、终点、IP 城市名称，让模型判断：
                #  - start/end 文本中是否明显包含某个城市名；
                #  - 若有，则返回最可能的城市名；
                #  - 同时返回一个 is_ip_city_reasonable 标志，表示是否建议继续使用 IP 城市做地理编码。
                check_prompt = (
                    "你是一个地理位置判断助手，请根据用户输入的起点/终点文本和 IP 定位的城市，判断 IP 城市是否适合作为地理编码的 city 参数。\n"
                    "请严格输出 JSON，对象字段为：\n"
                    "  - detected_start_city: 从起点文本中识别出的城市/区县名称（如无法确定请用 null）\n"
                    "  - detected_end_city: 从终点文本中识别出的城市/区县名称（如无法确定请用 null）\n"
                    "  - is_ip_city_reasonable: 布尔值，若 IP 城市和起点/终点显著冲突（例如 IP=西安，但起点/终点明确在杭州），请返回 false，否则 true。\n"
                    "  - preferred_city_hint: 若 is_ip_city_reasonable=false，建议用于地理编码的城市名称（如“杭州市”）；否则可为 null。\n"
                    "注意：\n"
                    "  - 不要调用任何外部工具，仅根据文本判断。\n"
                    "  - 只输出 JSON，不要输出多余文字。\n"
                )
                check_messages = [
                    {"role": "system", "content": check_prompt},
                    {
                        "role": "user",
                        "content": (
                            f"起点：{req.start}\n"
                            f"终点：{req.end}\n"
                            f"IP 城市：{ip_city_human_readable}\n"
                        ),
                    },
                ]
                check_responses = run_agent_with_retry(
                    summary_bot,
                    check_messages,
                    max_retries=1,
                    step_name="IP 城市合理性判断",
                )
                check_text = _extract_assistant_content(check_responses)
                try:
                    check_data = json.loads(check_text)
                except Exception:
                    check_data = {}
                
                if isinstance(check_data, dict):
                    is_ip_city_reasonable = bool(check_data.get("is_ip_city_reasonable", True))
                    preferred_city_hint = check_data.get("preferred_city_hint")
                    print(f"[IP+LLM] 判断结果：is_ip_city_reasonable={is_ip_city_reasonable}, preferred_city_hint={preferred_city_hint}")
                    # 如果模型认为 IP 城市不合理，则后续地理编码阶段不再优先使用 IP 城市，而是交给通用方案；
                    # 若模型建议了 preferred_city_hint，可在后续按需接入（当前仅记录日志，不强制替换）。
                    if not is_ip_city_reasonable:
                        print("[IP+LLM] 模型认为 IP 城市可能与起点/终点不符，将在地理编码阶段弱化 IP 方案的优先级")
                        # 简单做法：清空 ip_city_or_adcode，使后续仅走通用地理编码逻辑
                        ip_city_or_adcode = None
            except Exception as e:
                print(f"[IP+LLM] IP 城市合理性判断失败，忽略本次判断：{e}")

        # ===== 步骤1：路线规划 =====
        # 根据出行方式生成不同的用户文本
        if transport_mode == "walking":
            route_type = "步行路线"
        elif transport_mode == "transit":
            route_type = "公共交通路线"
        else:
            route_type = "驾车路线"
        user_text = f"从 {req.start} 到 {req.end} 的{route_type}。"
        # if req.poi:
        #     user_text += f" 用户沿途希望方便「{req.poi}」。"

        # 规范化 key，用于命中缓存（包含出行方式）
        transport_mode = req.transport_mode or "driving"
        route_key = _make_route_key(req.start, req.end, transport_mode)
        
        # 初始化 transit_info（公交路线信息）
        transit_info: Optional[Dict[str, Any]] = None

        if route_key in _ROUTE_CACHE:
            print(f"[缓存] 命中起点-终点-出行方式缓存：{req.start} -> {req.end} ({transport_mode})")
            cached = _ROUTE_CACHE[route_key]
            route_text = cached.get("route_text", "")
            waypoints_parsed = cached.get("waypoints") or None
            cached_distance_m = cached.get("distance_m")
            cached_start_loc = cached.get("start_loc")
            cached_end_loc = cached.get("end_loc")
            # 尝试从缓存中读取公交路线信息（如果存在）
            transit_info = cached.get("transit_info")
        else:
            print(f"[缓存] 未命中，直接调用高德 Web API：{req.start} -> {req.end} ({transport_mode})")
            
            # 地理编码：地址 -> 坐标
            # 优先尝试“IP 城市 + 通用”双方案，对比直线距离，避免 IP 城市把终点拉到完全错误的城市（例如 杭州→西安 变成 1300+ 公里）。
            try:
                start_loc = None
                end_loc = None
                
                if ip_city_or_adcode:
                    print(f"[地理编码] 检测到 IP 辅助城市/行政区：{ip_city_or_adcode}，将对比两种方案（含 city / 不含 city）")
                    ip_start_loc = ip_end_loc = None
                    plain_start_loc = plain_end_loc = None
                    
                    # 方案A：使用 IP 城市/行政区作为 city
                    try:
                        start_geo_ip = amap_client.geocode(req.start, city=ip_city_or_adcode)
                        end_geo_ip = amap_client.geocode(req.end, city=ip_city_or_adcode)
                        ip_start_loc = extract_lonlat_from_geocode(start_geo_ip)
                        ip_end_loc = extract_lonlat_from_geocode(end_geo_ip)
                    except Exception as e:
                        print(f"[地理编码] 使用 IP 城市({ip_city_or_adcode}) 地理编码失败：{e}")
                    
                    # 方案B：不带 city，交给高德自行判断城市
                    try:
                        start_geo_plain = amap_client.geocode(req.start)
                        end_geo_plain = amap_client.geocode(req.end)
                        plain_start_loc = extract_lonlat_from_geocode(start_geo_plain)
                        plain_end_loc = extract_lonlat_from_geocode(end_geo_plain)
                    except Exception as e:
                        print(f"[地理编码] 使用通用地理编码（不带 city）失败：{e}")
                    
                    # 根据两种方案的直线距离，选择更合理的一种
                    chosen_strategy = None
                    if ip_start_loc and ip_end_loc and plain_start_loc and plain_end_loc:
                        try:
                            dist_ip = _calculate_straight_distance_km(ip_start_loc, ip_end_loc)
                            dist_plain = _calculate_straight_distance_km(plain_start_loc, plain_end_loc)
                            print(f"[地理编码] IP 城市方案直线距离约 {dist_ip:.1f} km，通用方案约 {dist_plain:.1f} km")
                            
                            # 启发式选择：
                            # - 如果 IP 方案距离异常大（例如 > 300km）而通用方案明显更短（且 < 300km），优先选择通用方案；
                            # - 否则优先保留 IP 城市方案，以利用本地城市信息。
                            if dist_ip > 300 and dist_plain < dist_ip * 0.5 and dist_plain < 300:
                                start_loc, end_loc = plain_start_loc, plain_end_loc
                                chosen_strategy = "plain"
                                print("[地理编码] 检测到 IP 城市方案距离过长，优先采用通用方案以避免跨城误判")
                            else:
                                start_loc, end_loc = ip_start_loc, ip_end_loc
                                chosen_strategy = "ip"
                                print("[地理编码] 采用 IP 城市方案作为最终坐标")
                        except Exception as e:
                            print(f"[地理编码] 计算方案直线距离失败：{e}，将优先使用 IP 城市方案，如失败再回退通用方案")
                    
                    # 若只成功了一种方案，则直接使用那一种
                    if not chosen_strategy:
                        if ip_start_loc and ip_end_loc:
                            start_loc, end_loc = ip_start_loc, ip_end_loc
                            chosen_strategy = "ip_only"
                            print("[地理编码] 仅 IP 城市方案成功，采用该方案")
                        elif plain_start_loc and plain_end_loc:
                            start_loc, end_loc = plain_start_loc, plain_end_loc
                            chosen_strategy = "plain_only"
                            print("[地理编码] 仅通用地理编码方案成功，采用该方案")
                    
                    if not start_loc or not end_loc:
                        raise RuntimeError("两种地理编码方案均失败")
                else:
                    # 无 IP 辅助时，保持原有简单逻辑
                    start_geo = amap_client.geocode(req.start)
                    end_geo = amap_client.geocode(req.end)
                    start_loc = extract_lonlat_from_geocode(start_geo)
                    end_loc = extract_lonlat_from_geocode(end_geo)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"地理编码失败：{e}")

            # 调用对应出行方式的路线规划接口
            try:
                if transport_mode == "walking":
                    route_data = amap_client.walking_route(start_loc, end_loc)
                elif transport_mode == "transit":
                    # 公交路线规划可以使用城市信息进一步提高准确率
                    route_data = amap_client.transit_route(
                        start_loc,
                        end_loc,
                        city=ip_city_or_adcode,
                    )
                else:
                    route_data = amap_client.driving_route(start_loc, end_loc)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"高德路线规划调用失败：{e}")

            # 基本校验（公交返回 transits，驾车/步行返回 paths）
            route = route_data.get("route") or {}
            if transport_mode == "transit":
                route_items = route.get("transits") or []
                if not route_items:
                    raise HTTPException(status_code=500, detail="高德公交路线规划返回为空")
            else:
                route_items = route.get("paths") or []
                if not route_items:
                    raise HTTPException(status_code=500, detail="高德路线规划返回为空")

            route_json = json.dumps(route_data, ensure_ascii=False)

            # 步骤1：摘要
            summary_messages = [
                {
                    "role": "user",
                    "content": (
                        f"以下是高德路线规划的完整 JSON（出行方式：{transport_mode}），请给出简明摘要：\n"
                        f"{route_json}"
                    ),
                }
            ]
            summary_responses = run_agent_with_retry(
                summary_bot,
                summary_messages,
                max_retries=2,
                step_name="API 步骤1: 路线摘要",
            )
            route_text = _extract_assistant_content(summary_responses)

            # 步骤2：提取关键节点
            extract_messages = [
                {
                    "role": "user",
                    "content": (
                        f"出行方式：{transport_mode}\n"
                        "请从下方高德路线 JSON 中按行驶顺序抽取关键节点，并只输出 JSON 数组：\n"
                        f"{route_json}"
                    ),
                }
            ]
            waypoint_responses = run_agent_with_retry(
                waypoint_bot,
                extract_messages,
                max_retries=2,
                step_name="API 步骤2: 提取关键节点",
            )
            waypoint_text = _extract_assistant_content(waypoint_responses)
            waypoints_parsed = (
                extract_waypoint_names_from_json(waypoint_text) if waypoint_text else None
            )

            # 距离（米）：公交路线从 transits[0] 获取，驾车/步行从 paths[0] 获取
            distance_m = None
            try:
                if transport_mode == "transit":
                    transits = route.get("transits") or []
                    if transits:
                        transit = transits[0]
                        distance_str = transit.get("distance")
                        if distance_str is not None:
                            distance_m = float(distance_str)
                        
                        # 提取公交路线详细信息
                        try:
                            segments = transit.get("segments") or []
                            route_steps = []  # 路线步骤列表
                            final_alight_stop = None  # 最终下车点
                            
                            for i, segment in enumerate(segments):
                                step_info = {}
                                
                                if "walking" in segment:
                                    # 步行段
                                    walking = segment["walking"]
                                    distance = walking.get("distance", 0)
                                    duration = walking.get("duration", 0)
                                    step_info = {
                                        "type": "walking",
                                        "description": f"步行 {distance}米（约{int(duration/60)}分钟）",
                                        "distance": distance,
                                        "duration": duration
                                    }
                                elif "bus" in segment:
                                    # 公交段
                                    bus = segment["bus"]
                                    bus_name = bus.get("name", "未知线路")
                                    departure = bus.get("departure_stop", {})
                                    arrival = bus.get("arrival_stop", {})
                                    departure_name = departure.get("name", "未知站点")
                                    arrival_name = arrival.get("name", "未知站点")
                                    via_stops = bus.get("via_stops") or []
                                    stop_count = len(via_stops) + 2  # 途经站 + 出发站 + 到达站
                                    
                                    step_info = {
                                        "type": "bus",
                                        "description": f"乘坐{bus_name}，从「{departure_name}」到「{arrival_name}」（{stop_count}站）",
                                        "bus_name": bus_name,
                                        "departure_stop": departure_name,
                                        "arrival_stop": arrival_name,
                                        "stop_count": stop_count
                                    }
                                    
                                    # 记录最终下车点（最后一个公交段的到达站）
                                    if arrival.get("name"):
                                        final_alight_stop = {
                                            "name": arrival_name,
                                            "location": arrival.get("location", "")
                                        }
                                elif "subway" in segment:
                                    # 地铁段
                                    subway = segment["subway"]
                                    subway_name = subway.get("name", "未知线路")
                                    departure = subway.get("departure_stop", {})
                                    arrival = subway.get("arrival_stop", {})
                                    departure_name = departure.get("name", "未知站点")
                                    arrival_name = arrival.get("name", "未知站点")
                                    via_stops = subway.get("via_stops") or subway.get("stops") or []
                                    stop_count = len(via_stops) + 2
                                    
                                    step_info = {
                                        "type": "subway",
                                        "description": f"乘坐{subway_name}，从「{departure_name}」到「{arrival_name}」（{stop_count}站）",
                                        "subway_name": subway_name,
                                        "departure_stop": departure_name,
                                        "arrival_stop": arrival_name,
                                        "stop_count": stop_count
                                    }
                                    
                                    # 记录最终下车点（最后一个地铁段的到达站）
                                    if arrival.get("name"):
                                        final_alight_stop = {
                                            "name": arrival_name,
                                            "location": arrival.get("location", "")
                                        }
                                
                                if step_info:
                                    route_steps.append(step_info)
                            
                            # 构建完整的路线描述
                            route_description = " → ".join([step["description"] for step in route_steps])
                            
                            transit_info = {
                                "route_description": route_description,
                                "route_steps": route_steps,
                                "final_alight_stop": final_alight_stop,
                                "total_distance_m": distance_m,
                                "total_duration_s": transit.get("duration", 0)
                            }
                            
                            print(f"[步骤1] 提取公交路线信息：{route_description}")
                            if final_alight_stop:
                                print(f"[步骤1] 最终下车点：{final_alight_stop['name']}")
                        except Exception as e:
                            print(f"[步骤1] 提取公交路线信息失败：{e}")
                            import traceback
                            print(f"[步骤1] 异常堆栈：{traceback.format_exc()}")
                else:
                    paths = route.get("paths") or []
                    if paths:
                        distance_str = paths[0].get("distance")
                        if distance_str is not None:
                            distance_m = float(distance_str)
            except Exception:
                distance_m = None

            # 将结果写入缓存文件，供后续同一 A-B-出行方式直接复用
            _ROUTE_CACHE[route_key] = {
                "route_text": route_text,
                "waypoints": waypoints_parsed,
                "distance_m": distance_m,
                "start_loc": start_loc,
                "end_loc": end_loc,
                "transit_info": transit_info,  # 保存公交路线信息
                "cache_version": CACHE_VERSION,
            }
            _save_route_cache()

        # ===== 步骤3：沿途兴趣点（延迟加载，不在此步骤搜索） =====
        # 用户需要在路线规划完成后，主动触发兴趣点搜索
        pois_summary: Optional[List[Dict[str, str]]] = None
        poi_ids_all: List[str] = []
        print("[步骤3] 跳过自动兴趣点搜索，等待用户手动触发")

    logs = log_buffer.getvalue()

    return RouteResponse(
        summary=route_text,
        waypoints=waypoints_parsed,
        pois=pois_summary,
        pois_ids=poi_ids_all or None,
        logs=logs,
        transit_info=transit_info,
    )


@app.post("/api/poi_search", response_model=PoiSearchResponse)
async def poi_search_handler(req: PoiSearchRequest):
    """步骤3：根据路线拐角点搜索沿途兴趣点（按需触发）。"""
    log_buffer = io.StringIO()
    with contextlib.redirect_stdout(log_buffer):
        transport_mode = req.transport_mode or "driving"
        
        try:
            poi_bot = build_poi_search_agent()
        except Exception as e:
            print(f"[降级] 初始化 POI 搜索智能体失败，将跳过兴趣点搜索：{e}")
            poi_bot = None

        # 初始化高德客户端
        try:
            amap_client = AMapClient()
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"高德 API Key 配置有误或不可用：{e}")

        # 从缓存中获取路线的拐角点信息
        route_key = _make_route_key(req.start, req.end, transport_mode)
        waypoints_parsed = None
        
        if route_key in _ROUTE_CACHE:
            print(f"[兴趣点搜索] 从缓存读取拐角点：{req.start} -> {req.end} ({transport_mode})")
            cached = _ROUTE_CACHE[route_key]
            waypoints_parsed = cached.get("waypoints") or None
        else:
            print(f"[兴趣点搜索] 警告：未找到缓存的路线信息，无法进行兴趣点搜索")
            raise HTTPException(
                status_code=400, 
                detail="请先进行路线规划，系统需要拐角点信息才能搜索沿途兴趣点"
            )

        pois_summary: Optional[List[Dict[str, str]]] = None
        poi_ids_all: List[str] = []
        
        if waypoints_parsed:
            keyword = extract_search_keywords_from_user_input(req.poi) or req.poi.strip()
            
            # 智能选择搜索节点：确保包含起点、终点，以及均匀分布的中间节点
            search_waypoints = []
            total_points = len(waypoints_parsed)
            
            if total_points <= 10:
                # 节点数<=10，全部搜索
                search_waypoints = waypoints_parsed
                print(f"[POI搜索] 节点数={total_points}，将在所有节点搜索")
            else:
                # 节点数>10，智能采样：起点 + 均匀间隔的中间节点（最多8个）+ 终点
                search_waypoints.append(waypoints_parsed[0])  # 起点
                
                # 中间节点：均匀采样
                middle_indices = []
                step = (total_points - 2) / 8  # 排除起点和终点，取最多8个中间节点
                for i in range(1, 9):
                    idx = int(i * step)
                    if 0 < idx < total_points - 1:
                        middle_indices.append(idx)
                
                # 去重并排序
                middle_indices = sorted(set(middle_indices))
                for idx in middle_indices:
                    search_waypoints.append(waypoints_parsed[idx])
                
                search_waypoints.append(waypoints_parsed[-1])  # 终点
                
                print(f"[POI搜索] 节点数={total_points}，智能采样 {len(search_waypoints)} 个节点")
                print(f"[POI搜索] 采样节点索引：0(起点), {middle_indices}, {total_points-1}(终点)")
            
            poi_results: List[Dict[str, str]] = []

            def _wp_location(wp: Dict) -> Optional[str]:
                """从节点字典中提取坐标，支持数字和字符串格式"""
                lon = wp.get("lon")
                lat = wp.get("lat")
                
                # 尝试转换为数字格式
                try:
                    if lon is not None and lat is not None:
                        # 支持数字和字符串格式
                        lon_val = float(lon) if isinstance(lon, str) else lon
                        lat_val = float(lat) if isinstance(lat, str) else lat
                        if isinstance(lon_val, (int, float)) and isinstance(lat_val, (int, float)):
                            return f"{lon_val},{lat_val}"
                except (ValueError, TypeError):
                    pass
                
                # 如果坐标提取失败，尝试使用名称进行地理编码（作为备选）
                name = wp.get("name")
                if name:
                    try:
                        geo = amap_client.geocode(name)
                        loc = extract_lonlat_from_geocode(geo)
                        print(f"[POI搜索] 节点 {name} 使用地理编码获取坐标：{loc}")
                        return loc
                    except Exception as e:
                        print(f"[POI搜索] 节点 {name} 地理编码失败：{e}")
                        return None
                return None

            for i, wp in enumerate(search_waypoints):
                name = wp.get("name", f"节点{i+1}")
                loc = _wp_location(wp)
                if not loc:
                    print(f"[POI搜索] 节点 {name} 无法获取坐标，跳过")
                    continue
                try:
                    data = amap_client.around_search(location=loc, keywords=keyword, radius=500, page_size=5)
                    pois = data.get("pois") or data.get("pois", [])
                    pois_this_node: List[Dict[str, str]] = []
                    for p in pois[:5]:
                        if not isinstance(p, dict):
                            continue
                        entry = {
                            "name": str(p.get("name", "")),
                            "distance": str(p.get("distance", "")),
                            "id": str(p.get("id", "")),
                        }
                        pois_this_node.append(entry)
                        if entry["id"]:
                            poi_ids_all.append(entry["id"])
                    if pois_this_node:
                        poi_results.extend(pois_this_node)
                except Exception as e:
                    print(f"[POI搜索] 节点 {name} 周边搜索失败：{e}")
                # 控制请求节奏
                if i < len(search_waypoints) - 1:
                    time.sleep(0.3)

            # 若直连结果为空且有 MCP 备份，则尝试 MCP
            if not poi_results and poi_bot is not None:
                for i, wp in enumerate(search_waypoints):
                    name = wp.get("name", f"节点{i+1}")
                    prompt = (
                        f"请在节点「{name}」附近搜索「{keyword}」。\n"
                        f"调用周边搜索工具（maps_around_search，搜索半径建议 200-500 米）。\n"
                        f"只返回前 3-5 个结果的名称、距离和 id。"
                    )
                    search_messages = [{"role": "user", "content": prompt}]
                    search_responses = run_agent_with_retry(
                        poi_bot,
                        search_messages,
                        max_retries=1,
                        step_name=f"API 步骤3(备份): 节点 {name} 附近的「{keyword}」",
                    )
                    search_text = _extract_assistant_content(search_responses)
                    if i < len(search_waypoints) - 1:
                        time.sleep(0.3)
                    if not search_text:
                        continue
                    import re
                    pois_this_node: List[Dict[str, str]] = []
                    ids: List[str] = []
                    try:
                        obj = json.loads(search_text)
                        if isinstance(obj, list):
                            for p in obj:
                                if isinstance(p, dict):
                                    entry = {
                                        "name": str(p.get("name", "")),
                                        "distance": str(p.get("distance", "")),
                                        "id": str(p.get("id", "")),
                                    }
                                    pois_this_node.append(entry)
                                    if entry["id"]:
                                        ids.append(entry["id"])
                        elif isinstance(obj, dict) and "pois" in obj:
                            for p in obj.get("pois") or []:
                                if isinstance(p, dict):
                                    entry = {
                                        "name": str(p.get("name", "")),
                                        "distance": str(p.get("distance", "")),
                                        "id": str(p.get("id", "")),
                                    }
                                    pois_this_node.append(entry)
                                    if entry["id"]:
                                        ids.append(entry["id"])
                    except json.JSONDecodeError:
                        ids = re.findall(r'"id"\s*:\s*"([^"]+)"', search_text)

                    if ids:
                        poi_ids_all.extend(ids)
                    if pois_this_node:
                        poi_results.extend(pois_this_node)

            # 对POI结果进行去重
            if poi_results:
                seen_ids = set()
                seen_names = set()
                deduplicated_pois = []
                for poi in poi_results:
                    poi_id = poi.get("id", "").strip()
                    poi_name = poi.get("name", "").strip()
                    
                    if poi_id and poi_id in seen_ids:
                        continue
                    if not poi_id and poi_name and poi_name in seen_names:
                        continue
                    
                    deduplicated_pois.append(poi)
                    if poi_id:
                        seen_ids.add(poi_id)
                    if poi_name:
                        seen_names.add(poi_name)
                
                poi_results = deduplicated_pois
                poi_ids_all = list(dict.fromkeys(poi_ids_all))

            pois_summary = poi_results or []

    logs = log_buffer.getvalue()
    
    return PoiSearchResponse(
        pois=pois_summary,
        pois_ids=poi_ids_all or None,
        logs=logs,
    )


@app.post("/api/chat_poi", response_model=ChatPoiResponse)
async def chat_poi_handler(req: ChatPoiRequest):
    """对话式兴趣点搜索接口，支持多轮对话和需求提炼。"""
    log_buffer = io.StringIO()
    with contextlib.redirect_stdout(log_buffer):
        transport_mode = req.transport_mode or "driving"
        
        # 初始化对话助手（可选，如果失败则使用简单关键词提取）
        chat_agent = None
        try:
            chat_agent = build_chat_poi_agent()
            print(f"[对话POI] 对话助手初始化成功")
        except Exception as e:
            print(f"[对话POI] 对话助手初始化失败，将使用简单关键词提取：{e}")
            # 不抛出异常，继续使用简单模式
        
        # 初始化高德客户端
        try:
            amap_client = AMapClient()
        except Exception as e:
            raise HTTPException(
                status_code=503, 
                detail=f"高德 API Key 配置有误或不可用：{e}"
            )
        
        # 从缓存中获取路线的拐角点信息
        route_key = _make_route_key(req.start, req.end, transport_mode)
        waypoints_parsed = None
        
        if route_key in _ROUTE_CACHE:
            print(f"[对话POI] 从缓存读取拐角点：{req.start} -> {req.end} ({transport_mode})")
            cached = _ROUTE_CACHE[route_key]
            waypoints_parsed = cached.get("waypoints") or None
        else:
            print(f"[对话POI] 警告：未找到缓存的路线信息")
            raise HTTPException(
                status_code=400,
                detail="请先进行路线规划，系统需要拐角点信息才能搜索沿途兴趣点"
            )
        
        if not waypoints_parsed:
            raise HTTPException(
                status_code=400,
                detail="路线拐角点信息为空，无法进行兴趣点搜索"
            )
        
        # 构建对话消息（包含历史）
        messages = []
        
        # 添加系统上下文
        context_msg = (
            f"当前路线信息：从「{req.start}」到「{req.end}」，出行方式：{transport_mode}。\n"
            f"路线上有 {len(waypoints_parsed)} 个关键节点可供搜索周边兴趣点。"
        )
        messages.append({"role": "system", "content": context_msg})
        
        # 添加对话历史
        if req.conversation_history:
            messages.extend(req.conversation_history)
        
        # 添加用户当前消息
        messages.append({"role": "user", "content": req.message})
        
        print(f"[对话POI] 用户消息：{req.message}")
        print(f"[对话POI] 对话历史长度：{len(req.conversation_history or [])}")
        
        # 调用对话助手（如果可用）或使用简单模式
        assistant_reply = ""
        if chat_agent:
            try:
                responses = run_agent_with_retry(
                    chat_agent,
                    messages,
                    max_retries=2,
                    step_name="对话式POI助手",
                )
                assistant_reply = _extract_assistant_content(responses)
                print(f"[对话POI] 助手回复：{assistant_reply}")
            except Exception as e:
                print(f"[对话POI] 对话助手调用失败，使用简单模式：{e}")
                # 降级到简单模式
                assistant_reply = f"好的，我来帮你处理：{req.message}"
        else:
            # 简单模式：直接使用用户消息
            assistant_reply = f"好的，我来帮你处理：{req.message}"
            print(f"[对话POI] 使用简单模式处理用户消息")
        
        # 识别用户意图：搜索关键词 or 添加途经点
        extracted_keywords = None
        waypoint_name = None
        action_type = None
        
        # 检查是否是添加途经点（格式：「途经点：XXX」）
        waypoint_match = re.search(r'[「『【]途经点[：:]\s*([^」』】\n]+)[」』】]', assistant_reply)
        if waypoint_match:
            waypoint_name = waypoint_match.group(1).strip()
            action_type = "add_waypoint"
            print(f"[对话POI] 识别到途经点：{waypoint_name}")
        else:
            # 检查是否是搜索关键词（格式：「关键词：XXX」）
            keyword_match = re.search(r'[「『【]关键词[：:]\s*([^」』】\n]+)[」』】]', assistant_reply)
            if keyword_match:
                extracted_keywords = keyword_match.group(1).strip()
                action_type = "search"
                print(f"[对话POI] 识别到搜索关键词：{extracted_keywords}")
            else:
                # 简单模式：直接从用户消息中识别意图
                # 途经点关键词：去、到、经过、途经、经过、加一个、添加
                waypoint_patterns = [
                    r'去(.+?)(?:[，,。\.\s]|$)',
                    r'到(.+?)(?:[，,。\.\s]|$)',
                    r'经过(.+?)(?:[，,。\.\s]|$)',
                    r'途经(.+?)(?:[，,。\.\s]|$)',
                    r'加一个(.+?)(?:[，,。\.\s]|$)',
                    r'添加(.+?)(?:[，,。\.\s]|$)',
                ]
                
                for pattern in waypoint_patterns:
                    match = re.search(pattern, req.message)
                    if match:
                        waypoint_name = match.group(1).strip()
                        action_type = "add_waypoint"
                        assistant_reply = f"好的！我来帮你添加途经点「{waypoint_name}」到路线中。"
                        print(f"[对话POI] 简单模式识别到途经点：{waypoint_name}")
                        break
                
                # 如果没有识别为途经点，尝试提取搜索关键词
                if not action_type:
                    try:
                        extracted_keywords = extract_search_keywords_from_user_input(req.message)
                        if extracted_keywords:
                            action_type = "search"
                            assistant_reply = f"好的！我来帮你找沿途的「{extracted_keywords}」。"
                            print(f"[对话POI] 简单模式识别到搜索关键词：{extracted_keywords}")
                    except Exception as e:
                        print(f"[对话POI] 关键词提取失败：{e}")
        
        # 处理添加途经点
        pois_summary: Optional[List[Dict[str, str]]] = None
        poi_ids_all: List[str] = []
        waypoint_added = None
        
        if action_type == "add_waypoint" and waypoint_name:
            print(f"[对话POI] 开始添加途经点：{waypoint_name}")
            
            # 如果提供了session_id，添加到会话中
            if req.session_id:
                try:
                    session = _get_or_create_session(req.session_id)
                    
                    # 更新起终点
                    session["start"] = req.start
                    session["end"] = req.end
                    
                    # 解析途经点坐标
                    waypoint_location = None
                    try:
                        geo_data = amap_client.geocode(waypoint_name)
                        waypoint_location = extract_lonlat_from_geocode(geo_data)
                        print(f"[对话POI] 途经点坐标：{waypoint_location}")
                    except Exception as e:
                        print(f"[对话POI] 坐标解析失败：{e}")
                    
                    # 创建途经点对象
                    new_waypoint = {
                        "name": waypoint_name,
                        "location": waypoint_location,
                        "poi_id": None,
                        "order": len(session["waypoints"])
                    }
                    
                    # 添加到会话
                    session["waypoints"].append(new_waypoint)
                    waypoint_added = waypoint_name
                    
                    print(f"[对话POI] 成功添加途经点，当前共 {len(session['waypoints'])} 个")
                except Exception as e:
                    print(f"[对话POI] 添加途经点失败：{e}")
            else:
                print(f"[对话POI] 未提供session_id，无法添加途经点")
        
        # 处理搜索关键词
        elif action_type == "search" and extracted_keywords:
            print(f"[对话POI] 开始搜索关键词：{extracted_keywords}")
            
            # 智能选择搜索节点：确保包含起点、终点，以及均匀分布的中间节点
            search_waypoints = []
            total_points = len(waypoints_parsed)
            
            if total_points <= 10:
                # 节点数<=10，全部搜索
                search_waypoints = waypoints_parsed
                print(f"[对话POI] 节点数={total_points}，将在所有节点搜索")
            else:
                # 节点数>10，智能采样：起点 + 均匀间隔的中间节点（最多8个）+ 终点
                search_waypoints.append(waypoints_parsed[0])  # 起点
                
                # 中间节点：均匀采样
                middle_indices = []
                step = (total_points - 2) / 8  # 排除起点和终点，取最多8个中间节点
                for i in range(1, 9):
                    idx = int(i * step)
                    if 0 < idx < total_points - 1:
                        middle_indices.append(idx)
                
                # 去重并排序
                middle_indices = sorted(set(middle_indices))
                for idx in middle_indices:
                    search_waypoints.append(waypoints_parsed[idx])
                
                search_waypoints.append(waypoints_parsed[-1])  # 终点
                
                print(f"[对话POI] 节点数={total_points}，智能采样 {len(search_waypoints)} 个节点")
                print(f"[对话POI] 采样节点索引：0(起点), {middle_indices}, {total_points-1}(终点)")
            
            poi_results: List[Dict[str, str]] = []
            
            def _wp_location(wp: Dict) -> Optional[str]:
                """从节点字典中提取坐标，支持数字和字符串格式"""
                lon = wp.get("lon")
                lat = wp.get("lat")
                
                # 尝试转换为数字格式
                try:
                    if lon is not None and lat is not None:
                        # 支持数字和字符串格式
                        lon_val = float(lon) if isinstance(lon, str) else lon
                        lat_val = float(lat) if isinstance(lat, str) else lat
                        if isinstance(lon_val, (int, float)) and isinstance(lat_val, (int, float)):
                            return f"{lon_val},{lat_val}"
                except (ValueError, TypeError):
                    pass
                
                # 如果坐标提取失败，尝试使用名称进行地理编码（作为备选）
                name = wp.get("name")
                if name:
                    try:
                        geo = amap_client.geocode(name)
                        loc = extract_lonlat_from_geocode(geo)
                        print(f"[POI搜索] 节点 {name} 使用地理编码获取坐标：{loc}")
                        return loc
                    except Exception as e:
                        print(f"[POI搜索] 节点 {name} 地理编码失败：{e}")
                        return None
                return None
            
            for i, wp in enumerate(search_waypoints):
                name = wp.get("name", f"节点{i+1}")
                loc = _wp_location(wp)
                if not loc:
                    print(f"[对话POI] 节点 {name} 无法获取坐标，跳过")
                    continue
                try:
                    data = amap_client.around_search(
                        location=loc, 
                        keywords=extracted_keywords, 
                        radius=500, 
                        page_size=5
                    )
                    pois = data.get("pois") or []
                    pois_this_node: List[Dict[str, str]] = []
                    for p in pois[:5]:
                        if not isinstance(p, dict):
                            continue
                        entry = {
                            "name": str(p.get("name", "")),
                            "distance": str(p.get("distance", "")),
                            "id": str(p.get("id", "")),
                        }
                        pois_this_node.append(entry)
                        if entry["id"]:
                            poi_ids_all.append(entry["id"])
                    if pois_this_node:
                        poi_results.extend(pois_this_node)
                except Exception as e:
                    print(f"[对话POI] 节点 {name} 周边搜索失败：{e}")
                
                # 控制请求节奏
                if i < len(search_waypoints) - 1:
                    time.sleep(0.3)
            
            # 去重
            if poi_results:
                seen_ids = set()
                seen_names = set()
                deduplicated_pois = []
                for poi in poi_results:
                    poi_id = poi.get("id", "").strip()
                    poi_name = poi.get("name", "").strip()
                    
                    if poi_id and poi_id in seen_ids:
                        continue
                    if not poi_id and poi_name and poi_name in seen_names:
                        continue
                    
                    deduplicated_pois.append(poi)
                    if poi_id:
                        seen_ids.add(poi_id)
                    if poi_name:
                        seen_names.add(poi_name)
                
                poi_results = deduplicated_pois
                poi_ids_all = list(dict.fromkeys(poi_ids_all))
            
            pois_summary = poi_results or []
            print(f"[对话POI] 搜索完成，找到 {len(pois_summary)} 个兴趣点")
        
        # 更新对话历史
        updated_history = list(req.conversation_history or [])
        updated_history.append({"role": "user", "content": req.message})
        updated_history.append({"role": "assistant", "content": assistant_reply})
        
        # 限制历史长度（保留最近10轮对话）
        if len(updated_history) > 20:
            updated_history = updated_history[-20:]
    
    logs = log_buffer.getvalue()
    
    return ChatPoiResponse(
        reply=assistant_reply,
        pois=pois_summary,
        pois_ids=poi_ids_all or None,
        extracted_keywords=extracted_keywords,
        conversation_history=updated_history,
        action=action_type,
        waypoint_added=waypoint_added,
        logs=logs,
    )


@app.post("/api/poi_detail", response_model=PoiDetailResponse)
async def poi_detail_handler(req: PoiDetailRequest):
    """步骤4：根据 POI id 查询兴趣点详细信息（直接使用高德API，不依赖MCP）。"""
    log_buffer = io.StringIO()
    with contextlib.redirect_stdout(log_buffer):
        # 直接使用高德API，不依赖MCP服务
        try:
            client = AMapClient()
            print(f"[步骤4] 直接调用高德API查询POI详情：{req.poi_id}")
            
            # 调用高德POI详情API
            detail_data = client.poi_detail(req.poi_id)
            
            # 提取POI信息
            pois = detail_data.get("pois") or []
            if not pois:
                raise HTTPException(
                    status_code=404,
                    detail=f"未找到POI：{req.poi_id}",
                )
            
            poi_info = pois[0]  # 取第一个结果
            
            # 转换为JSON字符串格式（保持与之前MCP返回格式兼容）
            detail_text = json.dumps(poi_info, ensure_ascii=False, indent=2)
            print(f"[步骤4] POI详情查询成功：{poi_info.get('name', '未知')}")
            
        except HTTPException:
            raise
        except Exception as e:
            print(f"[步骤4] 高德API查询失败：{e}")
            raise HTTPException(
                status_code=500,
                detail=f"查询POI详情失败：{str(e)}",
            )

        # 额外计算"绕路距离"（如果前端提供了起终点）
        transport_mode = req.transport_mode or "driving"
        print(f"[步骤4] 开始计算绕路距离（出行方式：{transport_mode}）", flush=True)
        print(f"[步骤4] 接收到的参数：start={req.start}, end={req.end}, poi_id={req.poi_id}, transport_mode={transport_mode}", flush=True)
        
        # 从高德API返回的数据中提取POI信息
        poi_location = None
        poi_name = req.poi_id or "途经点"
        
        try:
            # 解析JSON数据
            poi_info = json.loads(detail_text) if isinstance(detail_text, str) else detail_text
            
            # 提取坐标（高德API格式：location字段为"lon,lat"）
            if isinstance(poi_info, dict):
                location = poi_info.get("location")
                if location and isinstance(location, str) and "," in location:
                    poi_location = location
                    print(f"[步骤4] 提取的POI坐标：{poi_location}", flush=True)
                
                # 提取名称
                name = poi_info.get("name")
                if name and isinstance(name, str):
                    poi_name = name
            elif isinstance(poi_info, list) and poi_info:
                # 如果是列表格式
                first = poi_info[0]
                if isinstance(first, dict):
                    location = first.get("location")
                    if location and isinstance(location, str) and "," in location:
                        poi_location = location
                        print(f"[步骤4] 提取的POI坐标：{poi_location}", flush=True)
                    name = first.get("name")
                    if name and isinstance(name, str):
                        poi_name = name
        except Exception as e:
            print(f"[步骤4] 解析POI信息失败：{e}", flush=True)
            # 如果解析失败，尝试使用_extract_poi_location作为备选
            poi_location = _extract_poi_location(detail_text)
            if poi_location:
                print(f"[步骤4] 使用备选方法提取的POI坐标：{poi_location}", flush=True)
        # 复用步骤1的缓存距离与坐标（若可用），减少一次基础路线调用
        cached_base_m = None
        cached_start_loc = None
        cached_end_loc = None
        if req.start and req.end:
            route_key = _make_route_key(req.start, req.end, transport_mode)
            cached_entry = _ROUTE_CACHE.get(route_key)
            if isinstance(cached_entry, dict):
                cached_base_m = cached_entry.get("distance_m")
                cached_start_loc = cached_entry.get("start_loc")
                cached_end_loc = cached_entry.get("end_loc")
                print(f"[步骤4] 读取缓存基础距离：{cached_base_m}，起点坐标：{cached_start_loc}，终点坐标：{cached_end_loc}", flush=True)
        extra_km = _compute_extra_distance_km(
            req.start,
            req.end,
            poi_location,
            transport_mode=transport_mode,
            cached_base_m=cached_base_m,
            cached_start_loc=cached_start_loc,
            cached_end_loc=cached_end_loc,
        )
        print(f"[步骤4] 计算得到的绕路距离：{extra_km} 公里", flush=True)

        # 生成"起点-POI-终点"路线摘要（驾车/步行支持途经点）与导航链接
        via_route_summary = None
        via_route_obj: Optional[Dict[str, Any]] = None
        via_nav_url = None
        if poi_location:
            try:
                client = AMapClient()
                # 起终点坐标：优先缓存，否则地理编码
                if cached_start_loc and "," in cached_start_loc:
                    start_loc = cached_start_loc
                else:
                    start_geo = client.geocode(req.start)
                    start_loc = extract_lonlat_from_geocode(start_geo)
                if cached_end_loc and "," in cached_end_loc:
                    end_loc = cached_end_loc
                else:
                    end_geo = client.geocode(req.end)
                    end_loc = extract_lonlat_from_geocode(end_geo)

                # 含途经点的完整路线（公交路线不支持途经点，需要特殊处理）
                alight_stop_name = None
                walk_nav_url = None
                show_add_to_route = True
                
                if transport_mode == "transit":
                    # 公交路线不支持途经点，需要找到最近的公交站，生成"公交站->POI"的步行导航
                    via_route_obj = None
                    via_route_summary = None
                    via_nav_url = None  # 公交模式不生成"加入路线"的导航链接
                    show_add_to_route = False  # 公交模式不显示"加入路线"按钮
                    
                    print(f"[步骤4] 公交模式：查找最近的公交站并生成步行导航")
                    
                    # 从缓存中获取公交路线信息
                    route_key = _make_route_key(req.start, req.end, transport_mode)
                    cached_route = _ROUTE_CACHE.get(route_key)
                    
                    if cached_route and poi_location:
                        try:
                            # 优先从缓存的 waypoints 中提取公交站信息（步骤一已经提取好了）
                            cached_waypoints = cached_route.get("waypoints") or []
                            bus_stops = []
                            
                            print(f"[步骤4] 从缓存中读取 {len(cached_waypoints)} 个节点，提取公交站...")
                            
                            for wp in cached_waypoints:
                                if not isinstance(wp, dict):
                                    continue
                                
                                # 提取公交站：segment_start（出发站）、via_stop（途经站）、segment_end（到达站）
                                wp_type = wp.get("type", "")
                                wp_note = wp.get("note", "")
                                
                                # 检查是否是公交/地铁站点
                                if (wp_type in ["segment_start", "via_stop", "segment_end"] and 
                                    ("公交" in wp_note or "地铁" in wp_note or "站" in wp_note)):
                                    wp_name = wp.get("name", "未知站点")
                                    wp_lon = wp.get("lon")
                                    wp_lat = wp.get("lat")
                                    
                                    # 构建坐标字符串
                                    if wp_lon is not None and wp_lat is not None:
                                        try:
                                            # 支持数字和字符串格式
                                            lon_val = float(wp_lon) if isinstance(wp_lon, str) else wp_lon
                                            lat_val = float(wp_lat) if isinstance(wp_lat, str) else wp_lat
                                            if isinstance(lon_val, (int, float)) and isinstance(lat_val, (int, float)):
                                                location = f"{lon_val},{lat_val}"
                                                bus_stops.append({
                                                    "name": wp_name,
                                                    "location": location
                                                })
                                                print(f"[步骤4] 提取公交站：{wp_name} ({location})，类型：{wp_type}，备注：{wp_note}")
                                        except (ValueError, TypeError) as e:
                                            print(f"[步骤4] 解析节点坐标失败：{wp_name}，错误：{e}")
                                            continue
                            
                            # 如果从缓存中没找到公交站，尝试重新调用API（作为备选）
                            if not bus_stops:
                                print(f"[步骤4] 缓存中未找到公交站，尝试重新调用API...")
                                transit_route_data = client.transit_route(start_loc, end_loc)
                                route = transit_route_data.get("route") or {}
                                transits = route.get("transits") or []
                                
                                if transits:
                                    transit = transits[0]
                                    segments = transit.get("segments") or []
                                    
                                    for segment in segments:
                                        if "bus" in segment:
                                            bus = segment["bus"]
                                            departure = bus.get("departure_stop", {})
                                            if departure.get("location"):
                                                bus_stops.append({
                                                    "name": departure.get("name", "未知站点"),
                                                    "location": departure.get("location")
                                                })
                                            via_stops = bus.get("via_stops") or []
                                            for via_stop in via_stops:
                                                if isinstance(via_stop, dict) and via_stop.get("location"):
                                                    bus_stops.append({
                                                        "name": via_stop.get("name", "未知站点"),
                                                        "location": via_stop.get("location")
                                                    })
                                            arrival = bus.get("arrival_stop", {})
                                            if arrival.get("location"):
                                                bus_stops.append({
                                                    "name": arrival.get("name", "未知站点"),
                                                    "location": arrival.get("location")
                                                })
                                        elif "subway" in segment:
                                            subway = segment["subway"]
                                            departure = subway.get("departure_stop", {})
                                            if departure.get("location"):
                                                bus_stops.append({
                                                    "name": departure.get("name", "未知站点"),
                                                    "location": departure.get("location")
                                                })
                                            via_stops = subway.get("via_stops") or subway.get("stops") or []
                                            for via_stop in via_stops:
                                                if isinstance(via_stop, dict) and via_stop.get("location"):
                                                    bus_stops.append({
                                                        "name": via_stop.get("name", "未知站点"),
                                                        "location": via_stop.get("location")
                                                    })
                                            arrival = subway.get("arrival_stop", {})
                                            if arrival.get("location"):
                                                bus_stops.append({
                                                    "name": arrival.get("name", "未知站点"),
                                                    "location": arrival.get("location")
                                                })
                            
                            # 找到距离POI最近的公交站
                            if bus_stops:
                                print(f"[步骤4] 找到 {len(bus_stops)} 个公交站，计算距离POI最近的站点")
                                min_distance = float('inf')
                                nearest_stop = None
                                
                                for stop in bus_stops:
                                    try:
                                        # 计算POI到公交站的直线距离（Haversine公式）
                                        dist = _calculate_straight_distance_km(poi_location, stop["location"])
                                        if dist < min_distance:
                                            min_distance = dist
                                            nearest_stop = stop
                                    except Exception as e:
                                        print(f"[步骤4] 计算POI到公交站 {stop['name']} 距离失败：{e}")
                                        continue
                                
                                if nearest_stop:
                                    alight_stop_name = nearest_stop["name"]
                                    stop_loc = nearest_stop["location"]
                                    print(f"[步骤4] 最近的公交站：{alight_stop_name}，距离POI {min_distance:.2f} 公里")
                                    
                                    # 生成从公交站到POI的步行导航链接
                                    walk_nav_url = (
                                        "https://ditu.amap.com/dir"
                                        f"?type=walk"
                                        f"&policy=0"
                                        f"&from[lnglat]={stop_loc}"
                                        f"&from[name]={quote(alight_stop_name)}"
                                        f"&to[lnglat]={poi_location}"
                                        f"&to[name]={quote(poi_name or '兴趣点')}"
                                        "&callnative=0&innersrc=uriapi&src=ama-agent"
                                    )
                                    print(f"[步骤4] 生成步行导航链接：{walk_nav_url}")
                                else:
                                    print(f"[步骤4] 无法找到最近的公交站")
                            else:
                                print(f"[步骤4] 公交路线中未找到公交站")
                        except Exception as e:
                            print(f"[步骤4] 获取公交站信息失败：{e}")
                            import traceback
                            print(f"[步骤4] 异常堆栈：{traceback.format_exc()}")
                    
                    # 如果找不到公交站，仍然生成一个从起点到POI的步行导航作为备选
                    if not walk_nav_url and poi_location:
                        print(f"[步骤4] 未找到公交站，生成从起点到POI的步行导航作为备选")
                        walk_nav_url = (
                            "https://ditu.amap.com/dir"
                            f"?type=walk"
                            f"&policy=0"
                            f"&from[lnglat]={start_loc}"
                            f"&from[name]={quote(req.start or '起点')}"
                            f"&to[lnglat]={poi_location}"
                            f"&to[name]={quote(poi_name or '兴趣点')}"
                            "&callnative=0&innersrc=uriapi&src=ama-agent"
                        )
                elif transport_mode == "walking":
                    via_route_data = client.walking_route(start_loc, end_loc, waypoints=poi_location)
                    via_route_obj = via_route_data.get("route") or via_route_data

                    # 用摘要助手生成简短描述
                    summary_bot = build_route_summary_agent()
                    via_route_json_str = json.dumps(via_route_data, ensure_ascii=False)
                    summary_messages = [
                        {
                            "role": "user",
                            "content": (
                                f"以下是带途经点的高德路线 JSON（出行方式：{transport_mode}，途经点已包含）：\n"
                                f"{via_route_json_str}"
                            ),
                        }
                    ]
                    summary_responses = run_agent_with_retry(
                        summary_bot,
                        summary_messages,
                        max_retries=1,
                        step_name="步骤4: 途经点路线摘要",
                    )
                    via_route_summary = _extract_assistant_content(summary_responses)
                else:  # driving
                    via_route_data = client.driving_route(start_loc, end_loc, waypoints=poi_location)
                    via_route_obj = via_route_data.get("route") or via_route_data

                    # 用摘要助手生成简短描述
                    summary_bot = build_route_summary_agent()
                    via_route_json_str = json.dumps(via_route_data, ensure_ascii=False)
                    summary_messages = [
                        {
                            "role": "user",
                            "content": (
                                f"以下是带途经点的高德路线 JSON（出行方式：{transport_mode}，途经点已包含）：\n"
                                f"{via_route_json_str}"
                            ),
                        }
                    ]
                    summary_responses = run_agent_with_retry(
                        summary_bot,
                        summary_messages,
                        max_retries=1,
                        step_name="步骤4: 途经点路线摘要",
                    )
                    via_route_summary = _extract_assistant_content(summary_responses)

                # 生成高德 Web 导航 URL（ditu.amap.com/dir）
                try:
                    # 基础参数：根据出行方式设置导航类型
                    if transport_mode == "driving":
                        mode_type = "car"
                        policy = 1  # 躲避拥堵
                    elif transport_mode == "walking":
                        mode_type = "walk"
                        policy = 0  # 步行不需要policy
                    elif transport_mode == "transit":
                        mode_type = "bus"  # 公交出行
                        policy = 2  # 公交使用policy=2
                    else:
                        mode_type = "car"
                        policy = 1
                    
                    # 公交模式已经在上面处理了，这里只处理驾车和步行
                    if transport_mode == "transit":
                        # 公交模式不生成 via_nav_url（已在上面设置为None）
                        pass
                    else:
                        # 驾车和步行使用原来的格式
                        via_nav_url = (
                            "https://ditu.amap.com/dir"
                            f"?type={mode_type}"
                            f"&policy={policy}"
                            f"&from[lnglat]={start_loc}"
                            f"&from[name]={quote(req.start or '')}"
                            f"&to[lnglat]={end_loc}"
                            f"&to[name]={quote(req.end or '')}"
                        )
                        
                        # 如果有途经点（POI），添加via参数（仅驾车和步行支持途经点）
                        if poi_location:
                            via_nav_url += (
                                f"&via[0][lnglat]={poi_location}"
                                f"&via[0][name]={quote(poi_name or '途经点')}"
                                f"&via[0][id]={quote(req.poi_id or '')}"
                            )
                        via_nav_url += "&callnative=0&innersrc=uriapi&src=ama-agent"
                except Exception as e:
                    print(f"[步骤4] 生成导航链接失败：{e}", flush=True)
                    via_nav_url = None
            except Exception as e:
                print(f"[步骤4] 生成途经点路线摘要失败：{e}", flush=True)
        
        # 同时在 stderr 输出，确保能在控制台看到（用于调试）
        print(f"[步骤4调试] 开始计算绕路距离", file=sys.stderr)
        print(f"[步骤4调试] 接收到的参数：start={req.start}, end={req.end}, poi_id={req.poi_id}", file=sys.stderr)
        print(f"[步骤4调试] 提取的POI坐标：{poi_location}", file=sys.stderr)
        print(f"[步骤4调试] 计算得到的绕路距离：{extra_km} 公里", file=sys.stderr)

    logs = log_buffer.getvalue()
    # 确保日志被正确捕获（调试用）
    if not logs or len(logs.strip()) == 0:
        print("⚠️ 警告：日志缓冲区为空！", file=sys.stderr)  # 使用 stderr 确保能看到
    else:
        print(f"✓ 日志已捕获，长度：{len(logs)} 字符", file=sys.stderr)  # 使用 stderr 确保能看到
        # 提取绕路距离计算相关的日志，输出到 stderr 方便查看
        log_lines = logs.split('\n')
        relevant_logs = [line for line in log_lines if '[绕路距离计算]' in line or '[POI坐标提取]' in line or '[步骤4]' in line]
        if relevant_logs:
            print(f"\n[关键日志摘要] 共 {len(relevant_logs)} 条相关日志：", file=sys.stderr)
            for log_line in relevant_logs[-20:]:  # 只显示最后20条
                print(f"  {log_line}", file=sys.stderr)
    
    return PoiDetailResponse(
        detail=detail_text,
        extra_distance_km=extra_km,
        via_route_summary=via_route_summary,
        via_route=via_route_obj,
        via_nav_url=via_nav_url,
        alight_stop_name=alight_stop_name,
        walk_nav_url=walk_nav_url,
        show_add_to_route=show_add_to_route,
        logs=logs,
    )


# ===== 途经点管理接口 =====
@app.post("/api/add_waypoint", response_model=AddWaypointResponse)
async def add_waypoint_handler(req: AddWaypointRequest):
    """添加途经点到会话"""
    log_buffer = io.StringIO()
    with contextlib.redirect_stdout(log_buffer):
        # 清理过期会话
        _cleanup_old_sessions()
        
        # 获取或创建会话
        session = _get_or_create_session(req.session_id)
        
        # 如果提供了起点和终点，更新会话信息
        if req.start:
            session["start"] = req.start
        if req.end:
            session["end"] = req.end
        
        # 更新出行方式到会话中
        transport_mode = req.transport_mode or "driving"
        session["transport_mode"] = transport_mode
        
        # 调试日志：打印出行方式和当前途经点数量
        existing_waypoints = session.get("waypoints", [])
        print(f"[添加途经点] 出行方式：{transport_mode}，当前途经点数量：{len(existing_waypoints)}")
        
        # 公交模式特殊检查：只允许添加一个途经点
        if transport_mode == "transit":
            if len(existing_waypoints) >= 1:
                print(f"[添加途经点] 公交模式限制：已有 {len(existing_waypoints)} 个途经点，不允许添加更多")
                logs = log_buffer.getvalue()
                return AddWaypointResponse(
                    success=False,
                    message="公交路线不支持添加多个中间节点。建议：在「" + existing_waypoints[0].get("name", "已添加的节点") + "」下车后，使用步行导航前往目的地。",
                    waypoints=[Waypoint(**wp) for wp in existing_waypoints],
                    logs=logs
                )
        else:
            # 驾车和步行模式：允许添加多个途经点
            print(f"[添加途经点] {transport_mode} 模式：允许添加多个途经点（当前 {len(existing_waypoints)} 个）")
        
        # 解析途经点坐标
        waypoint_location = None
        try:
            client = AMapClient()
            
            # 如果提供了POI ID，先尝试获取POI详情（直接使用高德API）
            if req.poi_id:
                print(f"[添加途经点] 通过POI ID获取详情：{req.poi_id}")
                try:
                    # 直接调用高德API获取POI详情
                    detail_data = client.poi_detail(req.poi_id)
                    pois = detail_data.get("pois") or []
                    if pois:
                        poi_info = pois[0]
                        location = poi_info.get("location")
                        if location and isinstance(location, str) and "," in location:
                            waypoint_location = location
                            print(f"[添加途经点] 从POI详情提取坐标：{waypoint_location}")
                except Exception as e:
                    print(f"[添加途经点] 获取POI详情失败：{e}")
            
            # 如果没有通过POI获取到坐标，尝试地理编码
            if not waypoint_location:
                print(f"[添加途经点] 地理编码：{req.waypoint_name}")
                geo_data = client.geocode(req.waypoint_name)
                waypoint_location = extract_lonlat_from_geocode(geo_data)
                print(f"[添加途经点] 地理编码结果：{waypoint_location}")
        except Exception as e:
            print(f"[添加途经点] 坐标解析失败：{e}")
            # 坐标解析失败不影响添加，可以后续再解析
        
        # 创建途经点对象
        new_waypoint = Waypoint(
            name=req.waypoint_name,
            location=waypoint_location,
            poi_id=req.poi_id,
            order=len(session["waypoints"])
        )
        
        # 添加到会话
        session["waypoints"].append(new_waypoint.dict())
        
        print(f"[添加途经点] 成功添加：{req.waypoint_name}，当前共 {len(session['waypoints'])} 个途经点")
    
    logs = log_buffer.getvalue()
    
    return AddWaypointResponse(
        success=True,
        message=f"成功添加途经点：{req.waypoint_name}",
        waypoints=[Waypoint(**wp) for wp in session["waypoints"]],
        logs=logs
    )


@app.post("/api/remove_waypoint", response_model=RemoveWaypointResponse)
async def remove_waypoint_handler(req: RemoveWaypointRequest):
    """从会话中删除途经点"""
    session = _get_or_create_session(req.session_id)
    
    if 0 <= req.waypoint_index < len(session["waypoints"]):
        removed = session["waypoints"].pop(req.waypoint_index)
        # 重新排序
        for i, wp in enumerate(session["waypoints"]):
            wp["order"] = i
        
        return RemoveWaypointResponse(
            success=True,
            message=f"成功删除途经点：{removed['name']}",
            waypoints=[Waypoint(**wp) for wp in session["waypoints"]]
        )
    else:
        return RemoveWaypointResponse(
            success=False,
            message="无效的途经点索引",
            waypoints=[Waypoint(**wp) for wp in session["waypoints"]]
        )


@app.post("/api/list_waypoints", response_model=ListWaypointsResponse)
async def list_waypoints_handler(req: ListWaypointsRequest):
    """列出会话中的所有途经点"""
    session = _get_or_create_session(req.session_id)
    
    return ListWaypointsResponse(
        waypoints=[Waypoint(**wp) for wp in session["waypoints"]],
        start=session.get("start"),
        end=session.get("end")
    )


@app.post("/api/optimize_route", response_model=OptimizeRouteResponse)
async def optimize_route_handler(req: OptimizeRouteRequest):
    """优化路线：重排序途经点并规划完整路径"""
    log_buffer = io.StringIO()
    with contextlib.redirect_stdout(log_buffer):
        transport_mode = req.transport_mode or "driving"
        
        # 获取会话
        session = _get_or_create_session(req.session_id)
        waypoints = session.get("waypoints", [])
        
        if not waypoints:
            return OptimizeRouteResponse(
                success=False,
                message="会话中没有途经点，无法优化路线",
                optimized_waypoints=[],
                logs=log_buffer.getvalue()
            )
        
        # 更新起终点
        session["start"] = req.start
        session["end"] = req.end
        
        try:
            client = AMapClient()
            
            # 步骤1：确保所有途经点都有坐标
            print(f"[路线优化] 步骤1：解析途经点坐标，共 {len(waypoints)} 个")
            for i, wp in enumerate(waypoints):
                if not wp.get("location"):
                    print(f"[路线优化] 解析途经点 {i+1}/{len(waypoints)}: {wp['name']}")
                    try:
                        geo_data = client.geocode(wp["name"])
                        wp["location"] = extract_lonlat_from_geocode(geo_data)
                        print(f"[路线优化] 解析成功：{wp['location']}")
                    except Exception as e:
                        print(f"[路线优化] 解析失败：{e}")
                        raise HTTPException(
                            status_code=400,
                            detail=f"无法解析途经点坐标：{wp['name']}"
                        )
            
            # 步骤2：获取起终点坐标
            print(f"[路线优化] 步骤2：解析起终点坐标")
            start_geo = client.geocode(req.start)
            end_geo = client.geocode(req.end)
            start_loc = extract_lonlat_from_geocode(start_geo)
            end_loc = extract_lonlat_from_geocode(end_geo)
            print(f"[路线优化] 起点：{start_loc}，终点：{end_loc}")
            
            # 步骤3：根据出行方式分别处理
            print(f"[路线优化] 步骤3：根据出行方式优化途经点")
            
            if transport_mode == "transit":
                # 公交路线：不支持多途经点优化，也不支持带途经点的公交规划。
                # 这里仅支持“从单个中间节点步行到终点”的简单导航，用于提醒用户在该节点下车后步行前往目的地。
                print(f"[路线优化] 公交路线模式：仅支持单个中间节点的步行导航，不支持多节点优化")
                
                # 验证：公交模式只支持单个途经点
                if len(waypoints) > 1:
                    logs = log_buffer.getvalue()
                    return OptimizeRouteResponse(
                        success=False,
                        message="公交路线不支持添加多个中间节点，请只添加一个下车节点。建议：在中间节点下车后，使用步行导航前往目的地。",
                        optimized_waypoints=[],
                        logs=logs
                    )
                
                # 仅使用第一个途经点作为“下车节点”
                alight_wp = waypoints[0]
                if not alight_wp.get("location"):
                    logs = log_buffer.getvalue()
                    return OptimizeRouteResponse(
                        success=False,
                        message=f"下车节点「{alight_wp.get('name', '未知')}」的坐标解析失败，无法规划步行路线",
                        optimized_waypoints=[],
                        logs=logs
                    )
                
                optimized_waypoints = [alight_wp]
                alight_loc = alight_wp["location"]
                print(f"[路线优化] 选定下车节点：{alight_wp['name']}，坐标：{alight_loc}")
                
                # 步骤4（公交专用）：规划“下车节点 -> 终点”的步行路线
                print(f"[路线优化] 步骤4（公交）：规划下车节点到终点的步行路线")
                route_data = client.walking_route(alight_loc, end_loc)
                
                # 步骤5：生成步行路线摘要
                print(f"[路线优化] 步骤5：生成步行路线摘要（下车节点 -> 终点）")
                route_json = json.dumps(route_data, ensure_ascii=False)
                
                summary_bot = build_route_summary_agent()
                summary_messages = [
                    {
                        "role": "user",
                        "content": (
                            "以下是从公交下车节点到终点的步行路线高德 JSON（出行方式：walking），请给出简明摘要，"
                            "并提醒用户：公交模式下不支持添加多个中间节点，只能选择一个下车节点后步行前往目的地：\n"
                            f"{route_json}"
                        ),
                    }
                ]
                summary_responses = run_agent_with_retry(
                    summary_bot,
                    summary_messages,
                    max_retries=2,
                    step_name="生成公交下车后步行路线摘要",
                )
                route_summary = _extract_assistant_content(summary_responses)
                
                # 提取距离和时间
                route = route_data.get("route") or {}
                paths = route.get("paths") or []
                total_distance_km = None
                total_duration_min = None
                if paths:
                    distance_str = paths[0].get("distance")
                    duration_str = paths[0].get("duration")
                    if distance_str:
                        total_distance_km = float(distance_str) / 1000.0
                    if duration_str:
                        total_duration_min = float(duration_str) / 60.0
                
                print(f"[路线优化] 步行距离：{total_distance_km} 公里，预计时间：{total_duration_min} 分钟")
                
                # 步骤6：仅生成“下车节点 -> 终点”的步行导航链接（不再返回多节点路径）
                print(f"[路线优化] 步骤6（公交）：生成下车节点到终点的步行导航链接")
                nav_url = None
                try:
                    mode_type = "walk"
                    policy = 0
                    nav_url = (
                        "https://ditu.amap.com/dir"
                        f"?type={mode_type}"
                        f"&policy={policy}"
                        f"&from[lnglat]={alight_loc}"
                        f"&from[name]={quote(alight_wp['name'] or '')}"
                        f"&to[lnglat]={end_loc}"
                        f"&to[name]={quote(req.end or '')}"
                        "&callnative=0&innersrc=uriapi&src=ama-agent"
                    )
                    print(f"[路线优化] 步行导航链接生成成功")
                except Exception as e:
                    print(f"[路线优化] 生成步行导航链接失败：{e}")
                
                # 公交模式处理完成，直接返回（不执行后续的多节点优化逻辑）
                session["waypoints"] = [wp for wp in optimized_waypoints]
                logs = log_buffer.getvalue()
                
                # 构造明确的提示消息
                walk_info = ""
                if total_distance_km is not None and total_duration_min is not None:
                    walk_info = f"（步行约 {total_distance_km:.1f} 公里，预计 {total_duration_min:.0f} 分钟）"
                
                return OptimizeRouteResponse(
                    success=True,
                    message=f"✅ 公交路线规划完成\n\n📌 下车节点：{alight_wp['name']}\n🚶 下车后步行至目的地{walk_info}\n\n⚠️ 提示：公交模式不支持添加多个中间节点，只能选择一个下车点。下方提供的是「下车节点 → 目的地」的步行导航按钮。",
                    optimized_waypoints=[Waypoint(**wp) for wp in optimized_waypoints],
                    route_summary=route_summary,
                    total_distance_km=total_distance_km,
                    total_duration_min=total_duration_min,
                    nav_url=nav_url,
                    logs=logs
                )
            else:
                # 驾车和步行：使用贪心算法优化多途经点顺序，然后调用高德带途经点路线规划
                print(f"[路线优化] 步骤3：优化途经点顺序（贪心算法）")
                
                # 驾车和步行支持优化
                unvisited = list(range(len(waypoints)))
                visited_order = []
                current_loc = start_loc
                
                while unvisited:
                    # 找到距离当前位置最近的途经点
                    min_distance = float('inf')
                    nearest_idx = -1
                    nearest_global_idx = -1
                    
                    for local_idx, global_idx in enumerate(unvisited):
                        wp_loc = waypoints[global_idx]["location"]
                        try:
                            # 使用直线距离估算（避免过多API调用）
                            distance = _calculate_straight_distance_km(current_loc, wp_loc)
                            if distance < min_distance:
                                min_distance = distance
                                nearest_idx = local_idx
                                nearest_global_idx = global_idx
                        except Exception as e:
                            print(f"[路线优化] 计算距离失败：{e}")
                            continue
                    
                    if nearest_idx >= 0:
                        visited_order.append(nearest_global_idx)
                        current_loc = waypoints[nearest_global_idx]["location"]
                        unvisited.pop(nearest_idx)
                        print(f"[路线优化] 选择途经点 {len(visited_order)}: {waypoints[nearest_global_idx]['name']}")
                
                # 按优化后的顺序重新排列
                optimized_waypoints = [waypoints[i] for i in visited_order]
                for i, wp in enumerate(optimized_waypoints):
                    wp["order"] = i
                
                # 步骤4：规划完整路线（包含所有途经点）
                print(f"[路线优化] 步骤4：规划完整路线（包含 {len(optimized_waypoints)} 个途经点）")
                
                # 构建途经点字符串（高德API格式：lon1,lat1;lon2,lat2;...）
                waypoints_str = ";".join([wp["location"] for wp in optimized_waypoints])
                print(f"[路线优化] 途经点坐标串：{waypoints_str[:100]}...")
                
                # 调用高德路线规划API
                route_data = None
                if transport_mode == "walking":
                    # 步行路线支持途经点
                    route_data = client.walking_route(start_loc, end_loc, waypoints=waypoints_str)
                else:
                    # 驾车路线支持途经点
                    route_data = client.driving_route(start_loc, end_loc, waypoints=waypoints_str)
                
                # 步骤5：生成路线摘要
                print(f"[路线优化] 步骤5：生成路线摘要")
                route_json = json.dumps(route_data, ensure_ascii=False)
                
                summary_bot = build_route_summary_agent()
                summary_messages = [
                    {
                        "role": "user",
                        "content": (
                            f"以下是高德路线规划的完整 JSON（出行方式：{transport_mode}，包含 {len(optimized_waypoints)} 个途经点），请给出简明摘要：\n"
                            f"{route_json}"
                        ),
                    }
                ]
                summary_responses = run_agent_with_retry(
                    summary_bot,
                    summary_messages,
                    max_retries=2,
                    step_name="生成优化路线摘要",
                )
                route_summary = _extract_assistant_content(summary_responses)
                
                # 提取距离和时间
                route = route_data.get("route") or {}
                paths = route.get("paths") or []
                total_distance_km = None
                total_duration_min = None
                
                if paths:
                    distance_str = paths[0].get("distance")
                    duration_str = paths[0].get("duration")
                    if distance_str:
                        total_distance_km = float(distance_str) / 1000.0
                    if duration_str:
                        total_duration_min = float(duration_str) / 60.0
                
                print(f"[路线优化] 总距离：{total_distance_km} 公里，总时间：{total_duration_min} 分钟")
                
                # 步骤6：生成包含所有途经点的导航链接
                print(f"[路线优化] 步骤6：生成导航链接")
                nav_url = None
                try:
                    if transport_mode == "driving":
                        mode_type = "car"
                        policy = 1
                    elif transport_mode == "walking":
                        mode_type = "walk"
                        policy = 0
                    else:
                        mode_type = "car"
                        policy = 1
                    
                    nav_url = (
                        "https://ditu.amap.com/dir"
                        f"?type={mode_type}"
                        f"&policy={policy}"
                        f"&from[lnglat]={start_loc}"
                        f"&from[name]={quote(req.start or '')}"
                        f"&to[lnglat]={end_loc}"
                        f"&to[name]={quote(req.end or '')}"
                    )
                    
                    # 添加所有途经点
                    for i, wp in enumerate(optimized_waypoints):
                        nav_url += (
                            f"&via[{i}][lnglat]={wp['location']}"
                            f"&via[{i}][name]={quote(wp['name'])}"
                        )
                        if wp.get("poi_id"):
                            nav_url += f"&via[{i}][id]={quote(wp['poi_id'])}"
                    
                    nav_url += "&callnative=0&innersrc=uriapi&src=ama-agent"
                    print(f"[路线优化] 导航链接生成成功")
                except Exception as e:
                    print(f"[路线优化] 生成导航链接失败：{e}")
            
            # 更新会话
            session["waypoints"] = [wp for wp in optimized_waypoints]
            
            logs = log_buffer.getvalue()
            
            return OptimizeRouteResponse(
                success=True,
                message=f"成功优化路线，包含 {len(optimized_waypoints)} 个途经点",
                optimized_waypoints=[Waypoint(**wp) for wp in optimized_waypoints],
                route_summary=route_summary,
                total_distance_km=total_distance_km,
                total_duration_min=total_duration_min,
                nav_url=nav_url,
                logs=logs
            )
        except HTTPException:
            raise
        except Exception as e:
            logs = log_buffer.getvalue()
            print(f"[路线优化] 优化失败：{e}")
            import traceback
            print(f"[路线优化] 异常堆栈：{traceback.format_exc()}")
            
            return OptimizeRouteResponse(
                success=False,
                message=f"路线优化失败：{str(e)}",
                optimized_waypoints=[Waypoint(**wp) for wp in waypoints],
                logs=logs
            )


def run_server(host="0.0.0.0", port=8000, reload=False, workers=2):
    """
    启动 FastAPI 服务器（供 Android 或其他环境调用）
    
    Args:
        host: 服务器监听地址，默认 "0.0.0.0"
        port: 服务器端口，默认 8000
        reload: 是否启用热重启，默认 False（Android 环境应设为 False）
        workers: worker 进程数量，默认 2（支持并发）
    """
    import uvicorn
    if reload:
        # 开发环境：使用 reload 模式（不支持 workers）
        uvicorn.run("api_server:app", host=host, port=port, reload=True)
    else:
        # 生产环境：使用 workers 支持并发
        uvicorn.run(
            "api_server:app",
            host=host,
            port=port,
            workers=workers,
            log_level="info",
            access_log=False
        )


if __name__ == "__main__":
    import uvicorn
    import sys
    # 检查命令行参数
    if len(sys.argv) > 1 and sys.argv[1] == "prod":
        # 生产模式：使用 workers 支持并发，不使用 reload
        uvicorn.run(
            "api_server:app",
            host="0.0.0.0",
            port=8000,
            workers=2,
            log_level="info",
            access_log=False
        )
    else:
        # 开发模式：reload=True 启用热重启，开发时修改代码会自动重启服务器
        # 注意：使用 reload 时必须使用导入字符串格式 "模块名:应用对象名"
        uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=True)