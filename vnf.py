# vnf.py

# 사용 가능한 VNF 목록
VNF_LIST = ["FW", "IDS", "NAT", "LB", "VPN", "WAF", "DPI", "Proxy"]

# VNF별 CPU 요구량 (예: GFLOPs)
VNF_CPU_REQUIREMENTS = {
    "FW": 10,
    "IDS": 20,
    "NAT": 15,
    "LB": 5,
    "VPN": 8,
    "WAF": 12,
    "DPI": 18,
    "Proxy": 7
}

# VNF별 대역폭 요구량 (예: Mbps)
VNF_BW_REQUIREMENTS = {
    "FW": 2,
    "IDS": 3,
    "NAT": 2.5,
    "LB": 1,
    "VPN": 2,
    "WAF": 1.5,
    "DPI": 2.5,
    "Proxy": 1
}

# 위성의 자원 한계
SATELLITE_CPU_CAPACITY = 100
SATELLITE_BW_CAPACITY = 10
