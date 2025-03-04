# routing.py
import heapq
import numpy as np
from formula import calculate_propagation_delay


def dijkstra_path(grid, start, destination):
    """
    Dijkstra 알고리즘을 사용하여 start에서 destination까지 최단 경로 계산
    """
    rows, cols = len(grid), len(grid[0])
    dist = {(i, j): float('inf') for i in range(rows) for j in range(cols)}
    prev = {(i, j): None for i in range(rows) for j in range(cols)}
    dist[start] = 0
    pq = [(0, start)]

    while pq:
        d, node = heapq.heappop(pq)
        if node == destination:
            break
        if d > dist[node]:
            continue
        x, y = node
        neighbors = []
        if x > 0: neighbors.append((x - 1, y))
        if x < rows - 1: neighbors.append((x + 1, y))
        if y > 0: neighbors.append((x, y - 1))
        if y < cols - 1: neighbors.append((x, y + 1))

        for n in neighbors:
            current_sat = grid[x][y]
            neighbor_sat = grid[n[0]][n[1]]
            delay = calculate_propagation_delay(current_sat, neighbor_sat)
            weight = delay if not np.isnan(delay) else float('inf')
            if d + weight < dist[n]:
                dist[n] = d + weight
                prev[n] = node
                heapq.heappush(pq, (dist[n], n))

    # 경로 복원
    path = []
    node = destination
    if prev[node] is None and node != start:
        return []
    while node is not None:
        path.append(node)
        node = prev[node]
    return list(reversed(path))


def dijkstra_next_hop(grid, current, destination):
    """
    현재 위성에서 목적지까지의 다음 홉 결정
    """
    path = dijkstra_path(grid, current, destination)
    return path[1] if len(path) >= 2 else None


def dijkstra_path_to_targets(grid, start, targets):
    """
    여러 목적지(targets) 중 하나로의 최단 경로 계산
    """
    rows, cols = len(grid), len(grid[0])
    dist = {(i, j): float('inf') for i in range(rows) for j in range(cols)}
    prev = {(i, j): None for i in range(rows) for j in range(cols)}
    dist[start] = 0
    pq = [(0, start)]
    reached = None

    while pq:
        d, node = heapq.heappop(pq)
        if node in targets:
            reached = node
            break
        if d > dist[node]:
            continue
        x, y = node
        neighbors = []
        if x > 0: neighbors.append((x - 1, y))
        if x < rows - 1: neighbors.append((x + 1, y))
        if y > 0: neighbors.append((x, y - 1))
        if y < cols - 1: neighbors.append((x, y + 1))

        for n in neighbors:
            current_sat = grid[x][y]
            neighbor_sat = grid[n[0]][n[1]]
            delay = calculate_propagation_delay(current_sat, neighbor_sat)
            weight = delay if not np.isnan(delay) else float('inf')
            if d + weight < dist[n]:
                dist[n] = d + weight
                prev[n] = node
                heapq.heappush(pq, (dist[n], n))

    if reached is None:
        return []
    path = []
    node = reached
    while node is not None:
        path.append(node)
        node = prev[node]
    return list(reversed(path))


def dijkstra_next_hop_extended(grid, current, targets):
    """
    여러 목표(targets) 중 하나를 향한 다음 홉 결정
    """
    path = dijkstra_path_to_targets(grid, current, targets)
    return path[1] if len(path) >= 2 else None
