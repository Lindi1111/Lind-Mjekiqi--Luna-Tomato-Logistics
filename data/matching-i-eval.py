class matching:
    def __init__(self, edges):
        self.edges = edges
        
    def fitness(self, x):
        used_a = set()
        used_b = set()
        used_c = set()
        total  = 0.0
        for bit, (a, b, c, w) in zip(x, self.edges):
            if bit == 0:
                continue
            if a in used_a or b in used_b or c in used_c:
                return [0]

            used_a.add(a)
            used_b.add(b)
            used_c.add(c)
            total += w

        return [-total]

    def get_bounds(self):
        m = len(self.edges)
        return ([0] * m, [1] * m )

def parse_input(filepath):
    edges = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("c") or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 4:
                continue
            a = int(parts[0])
            b = int(parts[1])
            c = int(parts[2])
            w = float(parts[3])
            edges.append((a, b, c, w))
    return edges

edges = parse_input("data/spoc4/ltl/random_harder_5000.txt")
udp   = matching(edges)