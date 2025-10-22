import sys
import heapq

INF = 10**9

def parse_row(line):
    tokens = []
    i = 0
    line = line.strip()
    while i < len(line):
        if line[i].isspace():
            i += 1
            continue
        j = i
        # read number
        while j < len(line) and line[j].isdigit():
            j += 1
        num = int(line[i:j])
        ch = line[j]
        tokens.append((num, ch))
        i = j + 1
    return tokens

def build_grid_and_bricks(N, lines):
    grid = [['']*N for _ in range(N)]
    brick_id = [[-1]*N for _ in range(N)]
    bricks = []  
    for r in range(N):
        tokens = parse_row(lines[r])
        c = 0
        for length, ch in tokens:
            cells = []
            for k in range(length):
                grid[r][c+k] = ch
                cells.append((r, c+k))
            bricks.append({'type': ch, 'cells': cells})
            bid = len(bricks) - 1
            for (rr, cc) in cells:
                brick_id[rr][cc] = bid
            c += length
    return grid, bricks, brick_id

def build_adjacency(N, bricks, brick_id):
    adj = [set() for _ in range(len(bricks))]
    for r in range(N):
        for c in range(N):
            u = brick_id[r][c]
            for dr, dc in ((1,0),(-1,0),(0,1),(0,-1)):
                rr, cc = r+dr, c+dc
                if 0 <= rr < N and 0 <= cc < N:
                    v = brick_id[rr][cc]
                    if u != v:
                        adj[u].add(v)
                        adj[v].add(u)
    return adj

def solve_instance(N, lines):
    grid, bricks, brick_id = build_grid_and_bricks(N, lines)
    adj = build_adjacency(N, bricks, brick_id)

    S_ids = [i for i,b in enumerate(bricks) if b['type'] == 'S']
    D_ids = set(i for i,b in enumerate(bricks) if b['type'] == 'D')

    if not S_ids or not D_ids:
        return -1  

    node_cost = []
    for b in bricks:
        t = b['type']
        if t == 'G':
            node_cost.append(1)
        elif t == 'R':
            node_cost.append(INF)
        else:
            node_cost.append(0)

    dist = [INF] * len(bricks)
    pq = []
    for s in S_ids:
        dist[s] = 0
        heapq.heappush(pq, (0, s))

    while pq:
        d,u = heapq.heappop(pq)
        if d != dist[u]:
            continue
        if u in D_ids:
            return d
        for v in adj[u]:
            w = node_cost[v]
            if w >= INF:
                continue
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(pq, (nd, v))

    return -1  # unreachable

if __name__ == "__main__":
    data = sys.stdin.read().strip().splitlines()
    if not data: 
        sys.exit(0)
    N = int(data[0].strip())
    lines = data[1:1+N]
    ans = solve_instance(N, lines)
    sys.stdout.write(str(ans))

