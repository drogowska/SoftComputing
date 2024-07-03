import copy
import sys
import cv2
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
from math import log10, sqrt
import pandas as pd
from sklearn.model_selection import train_test_split

class SOM():
    
    def __init__(self, data, orgin, no_neuron, quant, shape):
        self.network = []
        self.shape = shape
        self.quant = quant
        self.orgin = copy.deepcopy(orgin)
        l = int(0.001 * len(data))
        self.train = copy.deepcopy(data)
        np.random.shuffle(self.train)
        x_train ,x_test = train_test_split(data,test_size=0.8)
        self.train = x_train
        self.test = copy.deepcopy(data)
        self.pices_size = len(data[0])               # rozmiar małych bloczków
        self.n = int(len(data) / self.pices_size)    # ilość małych porcji obrzu
        for i in range(no_neuron): 
            self.network.append(Neuron(i, self.pices_size))
        return

    def fit(self, epochs, ni):
        for i in range(epochs):
            np.random.shuffle(self.train)
            for j in self.train: 
                np.random.shuffle(self.network)
                for n in self.network: 
                    n.calculate_output(j)
                self.network.sort(key=lambda x :x.y)
                self.network[0].update_w(ni)
        tbl = [n for n in self.network if n.mod]
        self.network = tbl
        # print('Final number of neurons: ', len(self.network))
        return

    def code(self):
        coded = []
        tmp = []
        X = []
        for i in range(len(self.test)): 
            self.calculate(self.test[i])
            coded.append(self.network[0].w)
            tmp.append(self.network[0].name)
            x = sqrt(self.network[0].scalar_product(self.orgin[i], self.orgin[i])) / self.quant
            X.append(int(x))
        return coded, tmp, X

    def decode(self, coded, tmp, X, file_name):
        decoded = []
        for i in range(len(tmp)):
            z = self.quant * X[i] * coded[i] 
            new = np.reshape(z, (int(sqrt(self.pices_size)),int(sqrt(self.pices_size))))
            decoded.append(new)
        decoded = self.resize_img(decoded)
        cv2.imwrite( './som/out/' + file_name  + '.png', decoded)
        return decoded

    def resize_img(self, filler):
        output = np.empty(self.shape)
        width, height = self.shape
        size = int(sqrt(self.pices_size)) 
        z = 0
        for i in range(0,width,size):
           for j in range(0,height,size):
                output[i : i + size, j : j + size ] = filler[z]
                z += 1
        return output
    
    def calculate(self, i):
        for n in self.network: 
            n.calculate_output(i)
        self.network.sort()
     
    
class Neuron:

    def __init__(self, name, N):
        self.name = name
        self.w =  np.random.uniform(-1, 1, N)
        self.w = self.w / np.linalg.norm(self.w)
        self.x = []
        self.mod = False

    def calculate_output(self, x) :
        self.x =  x
        self.y = self.scalar_product(self.w, x)

    def scalar_product(self,x, y):
        s = 0
        N = x.shape[0]
        for i in range(N):
            s += x[i] * y[i]
        return s
    
    def calculate_delta(self, z):
        self.d = z - self.y

    def __lt__(self, __value: object):
        return self.y > __value.y

    
    def update_w(self, ni):
        for i in range(len(self.x)):
            self.w[i] += ni * (self.x[i] - self.w[i])
        self.w = self.w / np.linalg.norm(self.w)
        self.mod = True

def mse(image_a, image_b):
    return np.mean((image_a.astype(float) - image_b.astype(float)) ** 2)

def psnr(img_a, img_b):
    return  10 * np.log10(255.0 ** 2 / mse(img_a, img_b)) 


def cr(original_img_arr, encoded_img_arr, crop_size, number_of_neurons, normalize=False):
    n_image_pixels = original_img_arr.shape[0] * original_img_arr.shape[1]
    n_crop_pixels = crop_size * crop_size
    not_compressed_size = n_image_pixels * 8
    compressed_size = (n_image_pixels / n_crop_pixels) * np.ceil(np.log2(number_of_neurons)) + n_crop_pixels * number_of_neurons * 8
    if normalize:
        compressed_size += (n_image_pixels / n_crop_pixels) * 8
    return not_compressed_size / compressed_size


def plots(df, file, piece_size):
    # plot compression ratio vs PSNR
    fig, ax = plt.subplots()
    ax.plot(df["compression_ratio"], df["PSNR"], '-')
    ax.plot(df["compression_ratio"], df["PSNR"], 'o')
    ax.set_xlabel("compression ratio")
    ax.set_ylabel("PSNR [dB]")
    plt.savefig('./SOM/data/psnrcr' + file + str(piece_size) + '.png')
    plt.show()

    # plot number of neurons vs PSNR
    fig, ax = plt.subplots()
    ax.plot(df["number_of_neurons"], df["PSNR"], '-')
    ax.plot(df["number_of_neurons"], df["PSNR"], 'o')
    ax.set_xlabel("number of neurons")
    ax.set_ylabel("PSNR [dB]")
    plt.savefig('./SOM/data/pc' + file + str(piece_size) + '.png')

    plt.show()


def init(file):
    img = cv2.imread('./SOM/data/' + file)
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    piece_size = 0
    piece_size = 8
    num_pieces_height = gray_image.shape[0] // piece_size
    num_pieces_width = gray_image.shape[1] // piece_size
    pieces_array = np.empty((num_pieces_height * num_pieces_width, piece_size * piece_size))
    normalise_array = np.empty((num_pieces_height * num_pieces_width, piece_size * piece_size))

    index = 0
    for i in range(num_pieces_height):
        for j in range(num_pieces_width):
            start_y = i * piece_size
            end_y = start_y + piece_size
            start_x = j * piece_size
            end_x = start_x + piece_size
            piece = gray_image[start_y:end_y, start_x:end_x]
            flattened_piece = piece.flatten()
            normalise_array[index] =  flattened_piece / np.linalg.norm(flattened_piece)
            pieces_array[index] = flattened_piece
            index += 1
    return pieces_array, normalise_array, (gray_image.shape[0], gray_image.shape[1]), gray_image, piece_size



neurons = [2,4,8,16,32,64,128,256]
learning_rate = 0.2
quant = 10
epochs = 10
# imgs = [ 'parrot.png',  'Cameraman.bmp',  'house.png']
imgs = ['barbara_gray.png']

for k in imgs:
    print('Image: ' + k)
    data, normalise, shape, img, piece_size = init(k)
    print('\nImage split into pieces size: ' + str(piece_size))
    cr_tab = []
    pnsr_tab = []
    for j in neurons:
        print('SOM made of ' + str(j) + ' neurons computing...')
        somm = SOM(normalise, data, j, quant, shape)
        somm.fit(epochs, learning_rate)
        coded, tmp, X = somm.code()
        text = k + '_p' + str(piece_size) + '_n' + str(j)
        reconstructed = somm.decode(coded, tmp, X, text )
        cra = (round(cr(img, reconstructed,piece_size,j),3))
        pr =  (round(psnr(img, reconstructed),3))
        print('CR:   ' + str(cra))
        print('PNSR: ' + str(pr) + ' dB \n')
        cr_tab.append(cra)
        pnsr_tab.append(pr)

    df = pd.DataFrame()
    df["number_of_neurons"] = neurons
    df["compression_ratio"] = cr_tab
    df["PSNR"] = pnsr_tab
    plots(df, k, piece_size)
    print('Done. Check output directory.')
    for i in range(len(neurons)):
        print( neurons[i], ' & ', pnsr_tab[i], ' & ', cr_tab[i], ' \\' )
