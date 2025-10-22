from collections import defaultdict, deque

INF = 10**9

# Min-Cost Max-Flow implementation using SPFA
class MinCostMaxFlow:
    def __init__(self, n):
        self.n = n
        self.graph = [[] for _ in range(n)]
        self.cap = {}
        self.cost = {}

    def add_edge(self, u, v, capacity, cost):
        self.graph[u].append(v)
        self.graph[v].append(u)
        self.cap[(u, v)] = capacity
        self.cap[(v, u)] = 0
        self.cost[(u, v)] = cost
        self.cost[(v, u)] = -cost

    def min_cost_flow(self, s, t, maxf):
        n = self.n
        flow, cost = 0, 0
        dist, parent = [0]*n, [0]*n

        while flow < maxf:
            dist = [INF]*n
            in_queue = [False]*n
            dist[s] = 0
            q = deque([s])
            while q:
                u = q.popleft()
                in_queue[u] = False
                for v in self.graph[u]:
                    if self.cap.get((u, v), 0) > 0 and dist[v] > dist[u] + self.cost[(u, v)]:
                        dist[v] = dist[u] + self.cost[(u, v)]
                        parent[v] = u
                        if not in_queue[v]:
                            q.append(v)
                            in_queue[v] = True
            if dist[t] == INF:
                break
            addf = maxf - flow
            v = t
            while v != s:
                u = parent[v]
                addf = min(addf, self.cap[(u, v)])
                v = u
            v = t
            while v != s:
                u = parent[v]
                self.cap[(u, v)] -= addf
                self.cap[(v, u)] += addf
                cost += addf * self.cost[(u, v)]
                v = u
            flow += addf
        return cost if flow == maxf else None


def solve():
    # Read input
    N, M = map(int, input().split())
    edges = [tuple(map(int, input().split())) for _ in range(M)]
    s1, s2 = map(int, input().split())
    t = int(input())

    # Build flow network
    # Each node v -> split into v_in = v*2, v_out = v*2 + 1
    total_nodes = 2 * (N + 1) + 2
    src = 0
    sink = total_nodes - 1
    mcmf = MinCostMaxFlow(total_nodes)

    def vin(v): return 2 * v
    def vout(v): return 2 * v + 1

    # Add node split edges
    for v in range(1, N + 1):
        if v == t:
            mcmf.add_edge(vin(v), vout(v), 2, 0)  # allow both scouts
        else:
            mcmf.add_edge(vin(v), vout(v), 1, 1)  # visiting this town costs 1

    # Add undirected edges
    for a, b in edges:
        mcmf.add_edge(vout(a), vin(b), 1, 0)
        mcmf.add_edge(vout(b), vin(a), 1, 0)

    # Super source connections
    mcmf.add_edge(src, vin(s1), 1, 0)
    mcmf.add_edge(src, vin(s2), 1, 0)

    # Sink connection
    mcmf.add_edge(vout(t), sink, 2, 0)

    result = mcmf.min_cost_flow(src, sink, 2)

    print(result if result is not None else "Impossible")
