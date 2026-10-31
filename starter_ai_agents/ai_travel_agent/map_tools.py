"""旅行规划用的免费地图距离工具（基于 OpenStreetMap 数据，无需任何 API Key）。

`estimate_travel_distance` 供 travel_agent.py 里的 Planner Agent 作为 tool 使用：

    from map_tools import estimate_travel_distance

本模块只依赖 requests 和标准库，不依赖 streamlit / agno，可以脱离页面单独测试：

    python -c "import map_tools; print(map_tools.estimate_travel_distance('金閣寺', '清水寺', near='京都'))"

设计要点（以下均为本机实测结论）：
  · 全部使用免费、免 Key 的服务；任何异常都转成字符串返回，绝不让 Agent 崩溃
  · 地理编码三级兜底：Nominatim（OSM 官方）→ Photon（Komoot，OSM）→ Open-Meteo（GeoNames）
    本机实测 Nominatim 不可达、Photon/Open-Meteo 可用；某个源网络失败会熔断 5 分钟，
    避免每次调用都白等超时，网络恢复后自动重试；若所有源都被熔断还会兜底试一次
  · 同名异地校验：near 先定位城市，候选地点离它超过 1500 km 就判为同名异地；
    拉丁名补「, 城市」、中日韩名补「 城市」重试（两种写法都实测过，不能互换）
  · 路网距离/时间：OSRM 公共演示服务；它只有 car profile（传 walking/cycling 也会返回驾车数据），
    因此步行/骑行不用 OSRM，改为「直线距离 × 绕路系数 ÷ 平均速度」估算
"""

import re
import time
from math import asin, cos, radians, sin, sqrt

import requests

# (连接超时, 读取超时) 秒；超时不会抛给模型，只会触发降级/熔断，避免页面长时间卡住
_MAP_TIMEOUT = (3.5, 6)
_MAP_SOURCE_COOLDOWN = 300  # 某个源网络级失败后熔断 5 分钟，期间优先跳过，省掉重复超时
_DEAD_SOURCES = {}  # {源名称: 恢复时间戳}，进程内熔断表（本机 Nominatim 经常不可达）
_MAP_USER_AGENT = "ai-travel-agent/1.0 (Streamlit demo)"
_NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
_PHOTON_URL = "https://photon.komoot.io/api/"
_OPEN_METEO_URL = "https://geocoding-api.open-meteo.com/v1/search"
_OSRM_ROUTE_URL = "https://router.project-osrm.org/route/v1/driving/{coords}"

# 各交通方式的平均速度（km/h）与「直线距离 → 实际路程」绕路系数，用于估算兜底
_MODE_PROFILES = {
    "driving": {"speed": 35.0, "detour": 1.30, "label": "驾车"},
    "cycling": {"speed": 15.0, "detour": 1.30, "label": "骑行"},
    "walking": {"speed": 4.8, "detour": 1.25, "label": "步行"},
}

# near 消歧半径（km）：候选地点离提示城市超过这个距离，就认为命中的是同名异地
# （实测单独搜「金阁寺」会命中山西太原的同名寺庙，必须靠这个半径拦下来）
_MAP_HINT_RADIUS_KM = 120.0
# 双阈值：120 km 内可信；120~1500 km 照常返回但附提醒（near 可能填的是国家/大区，
# 例如 near="Japan" 时京都市离日本地理中心约 358 km）；超过 1500 km 才判定为同名异地
_MAP_HINT_HARD_LIMIT_KM = 1500.0
# 进程内地理编码缓存：{(地名, hint): 结果}；只缓存成功结果，失败下次会重新尝试
_GEOCODE_CACHE = {}


class _MapSourceDown(Exception):
    """某个免费地图服务当前不可用（超时、DNS 失败、连接被重置、HTTP 非 200 等）。"""


def _map_http_json(url: str, params: dict):
    """GET 一个 JSON 接口：成功返回解析后的 data，响应无内容返回 None。

    网络层失败统一抛出 _MapSourceDown，由调用方熔断该源并降级；
    因此异常不会一路抛到 Agent 里；响应不是 JSON 时按「无结果」处理。
    """
    try:
        resp = requests.get(
            url,
            params=params,
            headers={"User-Agent": _MAP_USER_AGENT},
            timeout=_MAP_TIMEOUT,
        )
    except Exception as exc:  # 超时 / DNS 失败 / 连接被重置（本机访问 Nominatim 就是这种）
        raise _MapSourceDown(f"{type(exc).__name__}") from exc
    if resp.status_code != 200:
        raise _MapSourceDown(f"HTTP {resp.status_code}")
    try:
        return resp.json()
    except Exception:
        return None


def _lookup_nominatim(place: str, tried: list):
    """Nominatim（OSM 官方）：覆盖最全，使用政策要求 ≤1 req/s。返回 (lat, lon, label) 或 None。"""
    data = _map_http_json(
        _NOMINATIM_URL, {"q": place, "format": "json", "limit": 1}
    )
    if not data:
        tried.append("Nominatim 无结果")
        return None
    hit = data[0]
    return float(hit["lat"]), float(hit["lon"]), hit.get("display_name") or place


def _lookup_photon(place: str, tried: list):
    """Photon（Komoot 提供的 OSM 免费地理编码）：对中文/日文地名支持较好。"""
    data = _map_http_json(_PHOTON_URL, {"q": place, "limit": 1})
    features = (data or {}).get("features") or []
    if not features:
        tried.append("Photon 无结果")
        return None
    feature = features[0]
    lon, lat = feature["geometry"]["coordinates"][:2]
    props = feature.get("properties") or {}
    label = ", ".join(
        str(x)
        for x in (
            props.get("name"),
            props.get("street"),
            props.get("city") or props.get("district"),
            props.get("country"),
        )
        if x
    )
    return float(lat), float(lon), label or place


def _lookup_open_meteo(place: str, tried: list):
    """Open-Meteo（GeoNames 数据）：城市级兜底，景点级的中文地名通常查不到。"""
    data = _map_http_json(
        _OPEN_METEO_URL,
        {"name": place, "count": 1, "language": "en", "format": "json"},
    )
    results = (data or {}).get("results") or []
    if not results:
        tried.append("Open-Meteo 无结果")
        return None
    hit = results[0]
    label = ", ".join(
        str(x) for x in (hit.get("name"), hit.get("admin1"), hit.get("country")) if x
    )
    return float(hit["latitude"]), float(hit["longitude"]), label or place


def _map_geocode(place: str, hint=None) -> dict:
    """地名 → {"lat", "lon", "label", "dist_km"}；全部失败时返回 {"error": 原因}。

    hint 是可选的 (纬度, 经度) 城市坐标：给了 hint 时优先采用落在 hint 附近
    （_MAP_HINT_RADIUS_KM 以内）的候选，用来消解同名异地问题
    （例如「金阁寺」在中国山西和日本京都各有一座）。
    三个源都只有远处候选时返回最近的那个，并带上 dist_km 让调用方决定怎么提示。
    成功结果写入 _GEOCODE_CACHE，避免 Planner 反复调用时重复请求。
    """
    cache_key = (place, hint)
    if cache_key in _GEOCODE_CACHE:
        return _GEOCODE_CACHE[cache_key]

    tried = []
    sources = (
        ("Nominatim", _lookup_nominatim),
        ("Photon", _lookup_photon),
        ("Open-Meteo", _lookup_open_meteo),
    )

    def _dist(candidate):
        """候选地点离 hint 的直线距离（km）；没有 hint 时返回 0，表示都可信。"""
        if hint is None:
            return 0.0
        return _haversine_km(hint[0], hint[1], candidate[0], candidate[1])

    def _query(name, lookup):
        """跑一个源；网络失败就熔断该源并返回 None（异常不外泄）。"""
        try:
            return lookup(place, tried)
        except _MapSourceDown as exc:
            _DEAD_SOURCES[name] = time.time() + _MAP_SOURCE_COOLDOWN
            tried.append(f"{name} 网络失败({exc})")
            return None

    candidates = []
    skipped = []  # 被熔断跳过的源，留作最后兜底
    for name, lookup in sources:
        if _DEAD_SOURCES.get(name, 0) > time.time():
            tried.append(f"{name} 熔断中")
            skipped.append((name, lookup))
            continue
        candidate = _query(name, lookup)
        if not candidate:
            continue
        candidates.append(candidate)
        if _dist(candidate) <= _MAP_HINT_RADIUS_KM:
            break  # 已拿到可信候选，不必再请求其他源

    # 所有源都没给出候选时，无视熔断再试一次：避免一次网络抖动就让后续几分钟全部查不到
    if not candidates:
        for name, lookup in skipped:
            candidate = _query(name, lookup)
            if not candidate:
                continue
            candidates.append(candidate)
            if _dist(candidate) <= _MAP_HINT_RADIUS_KM:
                break

    if not candidates:
        return {"error": "；".join(tried) or "所有地理编码源均无返回"}

    # 取离 hint 最近的候选：最近的若在半径内即为可信命中，否则说明命中的是同名异地
    dist, (lat, lon, label) = min(
        ((_dist(candidate), candidate) for candidate in candidates), key=lambda x: x[0]
    )
    result = {
        "lat": lat,
        "lon": lon,
        "label": label,
        "dist_km": None if hint is None else dist,
    }

    if len(_GEOCODE_CACHE) > 500:  # 简单的容量保护，避免长时间运行后无限增长
        _GEOCODE_CACHE.clear()
    _GEOCODE_CACHE[cache_key] = result
    return result


_CJK_PATTERN = re.compile(r"[\u3400-\u9fff\u3040-\u30ff\uac00-\ud7af]")  # 中日韩字符


def _map_resolve(place: str, near: str, hint) -> dict:
    """定位一个地点并尽量消除同名异地：先查原名，校验不过再补上城市重试。

    补城市的写法必须按语言区分（均为实测）：
      · 拉丁名用逗号 —— "Kinkaku-ji" 单独查会先命中巴西，"Kinkaku-ji, Kyoto" 才是京都的景点；
      · 中日韩名用空格 —— "祇園" 单独查会命中广岛，"祇園 京都" 才是京都的祇園；
        中日韩名若拼成 "祇園, 京都" 或简体名拼 "金阁寺 京都"，Photon 会直接查不到。
    """
    queries = [place]
    if near and near not in place:
        separator = " " if _CJK_PATTERN.search(place) else ", "
        queries.append(f"{place}{separator}{near}")

    fallback = None  # 只在远处找到的候选，最后作为提示信息返回
    last_error = None
    for query in queries:
        hit = _map_geocode(query, hint)
        if "error" in hit:
            last_error = hit["error"]
            continue
        if hint is None or hit["dist_km"] is None or hit["dist_km"] <= _MAP_HINT_RADIUS_KM:
            return hit
        if fallback is None:
            fallback = hit
    return fallback or {"error": last_error or "所有地理编码源均无返回"}


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Haversine 直线距离（km），作为路网数据不可用时的兜底。"""
    radius = 6371.0088
    d_lat = radians(lat2 - lat1)
    d_lon = radians(lon2 - lon1)
    a = (
        sin(d_lat / 2) ** 2
        + cos(radians(lat1)) * cos(radians(lat2)) * sin(d_lon / 2) ** 2
    )
    return 2 * radius * asin(sqrt(a))


def estimate_travel_distance(
    origin: str, destination: str, near: str = "", mode: str = "driving"
) -> str:
    """估算两个地点之间的距离和预计交通时间，用来判断景点能否排在同一天、路线是否绕远。

    数据来自免费的 OpenStreetMap 服务（地理编码 + OSRM 路网），无需 API Key；
    路网服务不可用时自动退回直线距离估算。返回值是一行中文文本，可直接阅读。
    推荐用法：安排某一天内景点先后顺序之前，对相邻景点各调用一次，把距离近、耗时短的排在一起。

    Args:
        origin: 起点名称，可以是城市、车站或景点，例如 "金阁寺"、"Kyoto Station"。
        destination: 终点名称，格式同 origin，例如 "清水寺"、"Gion, Kyoto"。
        near: 可选的城市/地区提示，用于消歧：先定位该城市，若候选地点离它超过 1500 km 就判为同名异地
            并返回 NOT_FOUND（120–1500 km 之间照常返回但会附带确认提醒）；留空则不校验。
        mode: 交通方式，"driving"（默认，驾车/打车）、"walking"（步行）、"cycling"（骑行）。

    Returns:
        str: 成功时形如
             "金阁寺, 京都市, 日本 -> 清水寺, 京都市, 日本: 直线 8.4 km，驾车路程 11.6 km，预计 31 分钟（OSRM 路网）"；
             地名解析失败时以 "NOT_FOUND:" 开头并说明原因；参数非法或服务异常时以 "ERROR:" 开头。
             任何情况下都返回字符串，不会抛异常。
    """
    origin = (origin or "").strip()
    destination = (destination or "").strip()
    near = (near or "").strip()
    mode = (mode or "driving").strip().lower()

    # 1) 参数校验：缺参数、交通方式不支持时直接返回可读提示，不抛异常
    if not origin or not destination:
        return (
            "ERROR: origin 和 destination 都必须提供，例如 "
            "estimate_travel_distance('金阁寺', '清水寺', near='京都')。"
        )
    if mode not in _MODE_PROFILES:
        return f"ERROR: 不支持的 mode='{mode}'，可选值：driving / walking / cycling。"

    # 2) near 用于消歧：先定位提示城市，再用它的坐标筛掉同名异地
    #    注意不要把 near 直接拼进查询串——实测「金阁寺, 京都」这种混合 token 会让 Photon 直接查不到
    hint = None
    if near:
        near_hit = _map_geocode(near)
        if "error" not in near_hit:
            hint = (near_hit["lat"], near_hit["lon"])

    def _resolve(place: str, role: str):
        """定位单个地点：成功返回 (结果字典, None, 提醒后缀)，失败返回 (None, 提示文本, "")。"""
        hit = _map_resolve(place, near, hint)
        if "error" in hit:
            return None, (
                f"NOT_FOUND: 无法定位{role} '{place}'（失败原因：{hit['error']}）。"
                "请换用更完整的名称（例如加上城市/国家），或用 near 参数指明城市。"
            ), ""
        note = ""
        if hint and hit["dist_km"] is not None:
            if hit["dist_km"] > _MAP_HINT_HARD_LIMIT_KM:
                return None, (
                    f"NOT_FOUND: 在 '{near}' 附近找不到{role} '{place}'，最近的同名地点是 "
                    f"'{hit['label']}'，离 '{near}' 约 {hit['dist_km']:.0f} km。"
                    "请改用当地语言或更完整的名称（例如 '金閣寺' 或 'Kinkaku-ji, Kyoto'）。"
                ), ""
            if hit["dist_km"] > _MAP_HINT_RADIUS_KM:
                # near 可能填的是国家/大区，不直接判错，只提醒模型确认
                note = (
                    f"{role}离 near='{near}' 约 {hit['dist_km']:.0f} km，"
                    "请确认是否是你期望的地点"
                )
        return hit, None, note

    # 3) 地理编码：失败时返回 NOT_FOUND 提示，明确区分「查不到」和「网络不可用」
    origin_hit, origin_error, origin_note = _resolve(origin, "起点")
    if origin_error:
        return origin_error
    destination_hit, destination_error, destination_note = _resolve(destination, "终点")
    if destination_error:
        return destination_error

    hint_note = "；".join(n for n in (origin_note, destination_note) if n)
    if hint_note:
        hint_note = "；注意：" + hint_note

    o_lat, o_lon, o_label = origin_hit["lat"], origin_hit["lon"], origin_hit["label"]
    d_lat, d_lon, d_label = (
        destination_hit["lat"],
        destination_hit["lon"],
        destination_hit["label"],
    )

    straight_km = _haversine_km(o_lat, o_lon, d_lat, d_lon)
    prefix = f"{o_label} -> {d_label}"

    # 同一个地点（不足 50 米）直接说明，避免模型以为还要坐车
    if straight_km < 0.05:
        return f"{prefix}: 两地直线距离不足 50 米，属于同一区域，可步行前往。"

    profile = _MODE_PROFILES[mode]

    # 没有 near 时无法校验同名异地，若距离离谱就提醒模型补参数，而不是只给个荒谬的数字
    far_note = ""
    if hint is None and straight_km > 500:
        far_note = "；距离异常远，可能是同名异地，请用 near 参数或更完整的名称确认"

    # 4) 驾车：优先用 OSRM 真实路网（免费公共演示服务，只有 car profile）
    if mode == "driving":
        try:
            data = _map_http_json(
                _OSRM_ROUTE_URL.format(coords=f"{o_lon},{o_lat};{d_lon},{d_lat}"),
                {"overview": "false", "alternatives": "false"},
            )
        except _MapSourceDown:
            data = None  # 路网服务不可用：直接走下面的估算兜底
        routes = (data or {}).get("routes") or []
        if (data or {}).get("code") == "Ok" and routes:
            road_km = routes[0]["distance"] / 1000.0
            minutes = routes[0]["duration"] / 60.0
            note = "OSRM 路网，未含实时路况"
            # 给模型两个明确信号：跨城 / 实际路程远大于直线（山、海、单行道）
            if straight_km > 200:
                note += "；直线距离已超过 200 km，通常不该安排在同一天"
            elif straight_km > 3 and road_km > 2.5 * straight_km:
                note += "；实际路程远大于直线距离，路上耗时可能明显更长"
            return (
                f"{prefix}: 直线 {straight_km:.1f} km，驾车路程 {road_km:.1f} km，"
                f"预计 {minutes:.0f} 分钟（{note}）{far_note}{hint_note}"
            )
        # OSRM 失败或没有路线结果时，继续走下面的估算兜底

    # 5) 兜底：直线距离 × 绕路系数 ÷ 平均速度
    #    步行/骑行固定走这里，因为 OSRM 公共演示服务只提供驾车 profile
    est_km = straight_km * profile["detour"]
    minutes = est_km / profile["speed"] * 60
    fallback_reason = "OSRM 不可用，" if mode == "driving" else ""
    return (
        f"{prefix}: 直线 {straight_km:.1f} km，估算{profile['label']}路程 {est_km:.1f} km，"
        f"预计{profile['label']}约 {minutes:.0f} 分钟"
        f"（{fallback_reason}按平均 {profile['speed']:.0f} km/h 估算，仅供参考）{far_note}{hint_note}"
    )
