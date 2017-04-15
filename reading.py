class ReadInitData:
    def __init__(self):
        self.gamma1 = .0
        self.gamma2 = .0
        self.sigma = .0
        self.N = 0
        self.time = []

    def Reading(self, fileName):
        koef = []
        with open(fileName, 'r') as f:
            fileStr = f.readline()
            koef = fileStr.split(' ')
        for i in range(len(koef)):
            koef[i] = float(koef[i])
        self.gamma1, self.gamma2, self.sigma, self.N = koef[0], koef[1], koef[2], koef[3]
        for i in range(4,len(koef)):
            self.time.append(koef[i])
    def GetData(self):
        return self.gamma1, self.gamma2, self.sigma, self.N, self.time

    def GetEst(self, fname):    
        str_file = []
        G1 = []
        G2 = []
        S = []
        with open(fname, 'r') as f:
            for line in f:
                str_file.append(line)
        for i in range(0, len(str_file)):
            s = str_file[i].expandtabs(1).rstrip()
            g1, g2, sigm = s.split(' ')
            G1.append(float(g1))
            G2.append(float(g2))
            S.append(float(sigm))
        return G1, G2, S

