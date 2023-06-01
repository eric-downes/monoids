octoset = set(range(len(O)))
phi = {i:i for i in octoset}
for i in range(16):
    for j in range(16):
        add_or_err(G[i,j], O[phi[i], phi[j]], phi)        
phiset = set(phi)
l = 0
while l != len(phi):
    l = len(phi)
    for j in phiset:
        add_or_err(G[8,j], O[phi[8], phi[j]], phi)
    
    
looked = set()
l = 0
while len(phi) < len(G) and l != len(phi):
    l = len(s := set(phi))
    for i in s - looked:
        inv = np.argmin(G[i])
        neg = np.argmax(G[i] == 8)
        add_or_err(inv, np.argmin(O[phi[i]]), phi)
        add_or_err(neg, np.argmin(O[phi[i]]), phi)
        looked.add(i)
        print(len(phi))




class FinPoset:
    def update(self, a, b):
        up = lambda a,b: self.uppsetset[a].add(b)
        if (x: = self.decide(a,b)) <= 0:
            up(b,a)
        elif 0 <= x:
            up(a,b)
    def decide(self, a, b) -> bool:
        for rule in self.rules:
            self.op(
            
        
        
