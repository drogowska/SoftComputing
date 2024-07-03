
import numpy as np

class Letter():

    def __init__(self, row, col, lines):
        self.name = lines[0][:-1]
        lines = lines[1:]
        self.data = np.zeros(shape=(row, col), dtype=np.uint8)
        for r in range(row):            
            for c in range(col):
                if lines[r][c] == "-":
                    self.data[r, c] = 0
                elif lines[r][c] == "#":
                    self.data[r, c] = 1
                else:
                    raise ValueError(lines[r][c])
        self.data = self.data / np.sqrt(np.count_nonzero(self.data))
        self.data = np.reshape(self.data, (-1,))


class LetterSet():

    def __init__(self, filename):    
        with open(filename) as file:
            lines = file.readlines()
            n = int(lines[0])
            n_cols = int(lines[1])
            n_rows = int(lines[2])
            lines = lines[3:]
            set = []
            for i in range(n):
                lines = lines[1:] 
                set.append(Letter(n_rows, n_cols, lines[:n_rows + 1]))
                lines = lines[n_rows + 1:]
        self.set = set
        

class Madeline():

    def __init__(self, set):
        self.network = []
        for i in set: 
            self.network.append(Neuron(i.name, i.data))
    
    def test(self, set):
        for j in set:
            print("Wzorzec testowy: " + j.name )
            for i in self.network:
                i.calculate_output(j.data)
                print("Stopień podobieństwa do wzorca oryginalnego '" + i.name + "' wynosi: " + str(i.y))
            self.network.sort()
            max = self.network[0]
            print("Wzorzec został rozpoznany jako '" + max.name + "'\n")
    
class Neuron:

    def __init__(self, name, w):
        self.name = name
        self.w = w

    def calculate_output(self, x) :
        self.y = self.scalar_product(self.w, x)

    def scalar_product(self,x, y):
        s = 0
        N = x.shape[0]
        for i in range(N):
            s += x[i] * y[i]
        return s
    
    def calculate_delta(self, z):
        self.d = z - self.y

    def __lt__(self, __value: object) -> bool:
        return self.y > __value.y
    

train = LetterSet('./2.1/train.txt').set
net = Madeline(train)
test = LetterSet('./2.1/test.txt').set
net.test(test)