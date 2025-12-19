import os
import json
from pathlib import Path
from typing import Any, Dict, Optional

import httpx
from dotenv import load_dotenv

load_dotenv()


def _load_amap_api_key() -> str:
    """从 config.json 或环境变量中加载高德 API Key"""
    # 优先从 config.json 读取
    config_path = Path(__file__).with_name("config.json")
    if config_path.exists():
        try:
            with config_path.open("r", encoding="utf-8") as f:
                config = json.load(f)
                amap_cfg = config.get("amap", {})
                if amap_cfg and amap_cfg.get("api_key"):
                    return amap_cfg["api_key"]
        except (json.JSONDecodeError, KeyError):
            pass
    
    # 其次从环境变量读取
    return os.getenv("AMAP_API_KEY", "")


AMAP_API_KEY = _load_amap_api_key()


class AMapClient:
    """轻量封装高德 Web 服务"""

    def __init__(self, api_key: Optional[str] = None, timeout: float = 10.0):
        self.api_key = api_key or AMAP_API_KEY
        if not self.api_key:
            raise ValueError("AMAP_API_KEY is not set. Please set it in .env")
        self.client = httpx.Client(timeout=timeout)

    def _get(self, url: str, params: Dict[str, Any]) -> Dict[str, Any]:
        params = {"key": self.api_key, **params}
        resp = self.client.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()
        if data.get("status") != "1":
            raise RuntimeError(f"AMap API error: {data}")
        return data

    def ip_location(self, ip: Optional[str] = None) -> Dict[str, Any]:
        """
        IP 定位：根据 IP 获取所在城市/行政区信息。

        Args:
            ip: 可选，待查询的 IP 地址。如果为空，则由高德根据请求源 IP 判断，
                但在本项目中，我们通常会显式传入用户 IP。
        """
        url = "https://restapi.amap.com/v3/ip"
        params: Dict[str, Any] = {}
        if ip:
            params["ip"] = ip
        return self._get(url, params)

    def geocode(self, address: str, city: Optional[str] = None) -> Dict[str, Any]:
        """地理编码：地址 -> 经纬度"""
        url = "https://restapi.amap.com/v3/geocode/geo"
        params: Dict[str, Any] = {"address": address}
        if city:
            params["city"] = city
        return self._get(url, params)

    def driving_route(self, origin: str, destination: str, waypoints: Optional[str] = None, strategy: int = 4) -> Dict[str, Any]:
        """
        驾车路线规划
        origin/destination: 'lon,lat'
        waypoints: 途经点，多个途经点用 '|' 分隔，格式：'lon1,lat1|lon2,lat2'
        strategy: 路线规划策略
            - 0: 速度优先（不考虑路况）
            - 1: 费用优先（不走收费路段的最短距离）
            - 2: 距离优先（最短距离）
            - 3: 速度优先（考虑路况）
            - 4: 躲避拥堵（默认，与高德地图应用一致）
            - 5: 多策略（同时使用速度优先、费用优先、距离优先三个策略）
        """
        url = "https://restapi.amap.com/v3/direction/driving"
        params = {
            "origin": origin,
            "destination": destination,
            "extensions": "all",
            "strategy": strategy,  # 默认使用"躲避拥堵"策略，与高德地图应用保持一致
        }
        if waypoints:
            params["waypoints"] = waypoints
        return self._get(url, params)
    
    def walking_route(self, origin: str, destination: str, waypoints: Optional[str] = None) -> Dict[str, Any]:
        """
        步行路线规划
        origin/destination: 'lon,lat'
        waypoints: 途经点，多个途经点用 '|' 分隔，格式：'lon1,lat1|lon2,lat2'
        """
        url = "https://restapi.amap.com/v3/direction/walking"
        params = {
            "origin": origin,
            "destination": destination,
            "extensions": "all",
        }
        if waypoints:
            params["waypoints"] = waypoints
        return self._get(url, params)
    
    def transit_route(self, origin: str, destination: str, city: Optional[str] = None) -> Dict[str, Any]:
        """
        公共交通路线规划
        origin/destination: 'lon,lat'
        city: 城市代码或城市名称（可选，用于提高准确性）
        注意：公交路线规划不支持途经点
        """
        url = "https://restapi.amap.com/v3/direction/transit/integrated"
        params = {
            "origin": origin,
            "destination": destination,
            "extensions": "all",
        }
        if city:
            params["city"] = city
        return self._get(url, params)

    def around_search(self, location: str, keywords: str, radius: int = 500, page_size: int = 5) -> Dict[str, Any]:
        """
        周边搜索（place/around）
        location: 'lon,lat'
        keywords: 关键词
        radius: 搜索半径（米）
        page_size: 返回条数
        """
        url = "https://restapi.amap.com/v5/place/around"
        params = {
            "location": location,
            "keywords": keywords,
            "radius": radius,
            "page_size": page_size,
            "page_num": 1,
        }
        return self._get(url, params)
    
    def poi_detail(self, poi_id: str) -> Dict[str, Any]:
        """
        POI详情查询
        poi_id: POI的唯一标识符（如：B0FFHVFVO0）
        """
        url = "https://restapi.amap.com/v3/place/detail"
        params = {
            "id": poi_id,
            "extensions": "all",  # 返回详细信息
        }
        return self._get(url, params)


def extract_lonlat_from_geocode(geo_resp: Dict[str, Any]) -> str:
    """从地理编码结果中提取 'lon,lat' 字符串"""
    geocodes = geo_resp.get("geocodes") or []
    if not geocodes:
        raise ValueError("No geocode result found")
    location = geocodes[0].get("location")
    if not location:
        raise ValueError("No location field in geocode result")
    return location


