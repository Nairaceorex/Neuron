import numpy as np
import scipy.special
import imageio
import glob


# import matplotlib.pyplot
# import train1


class Neuron:

    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # задать кол-во узлов во входе, скрытом и выходе
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        # коэффициент обучения
        self.lr = learningrate
        # Матрица весовых коэфф. связей wih и who
        # Весовые коэфф. связей между узлом i и j(w_i_j)
        """
        self.wih = (np.random.rand(self.hnodes, self.inodes) - 0.5)
        self.who = (np.random.rand(self.onodes, self.hnodes) - 0.5)
        """

        self.wih = np.load("weight/wih.npy")
        self.who = np.load("weight/who.npy")

        # Функция активации
        self.activation_function = lambda x: scipy.special.expit(x)

        pass

    def query(self, inputs_list):
        # Преобразовать входные данные в список(2D)
        inputs = np.array(inputs_list, ndmin=2).T

        # Входящие сигналы для скрытого слоя
        hidden_inputs = np.dot(self.wih, inputs)
        # Исходящие сигналы для скрытого слоя
        hidden_outputs = self.activation_function(hidden_inputs)

        # Входящие сигналы для выхода
        final_inputs = np.dot(self.who, hidden_outputs)
        # Исходящие сигналы для выхода
        final_outputs = self.activation_function(final_inputs)

        return final_outputs


# кол-во входных, скрытых и выходных узлов
input_nodes = 784
hidden_nodes = 200
output_nodes = 10

# коэфф. обучения
learning_rate = 0.1

# экземпляр нейронной сети
n = Neuron(input_nodes, hidden_nodes, output_nodes, learning_rate)

our_own_dataset = []

for image_file_name in glob.glob('my_own_images/2828_my_own_?.png'):
    print("loading ... ", image_file_name)
    # use the filename to set the correct label
    label = int(image_file_name[-5:-4])
    # load image data from png files into an array
    img_array = imageio.imread(image_file_name, as_gray=True)
    # reshape from 28x28 to list of 784 values, invert values
    img_data = 255.0 - img_array.reshape(28, 28)
    # then scale data to range from 0.01 to 1.0
    img_data = (img_data / 255.0 * 0.99) + 0.01
    print(np.min(img_data))
    print(np.max(img_data))
    # append label and image data  to test data set
    record = np.append(label, img_data)
    print(record)
    our_own_dataset.append(record)
    pass

for item in range(10):

    # plot image
    # matplotlib.pyplot.imshow(our_own_dataset[item][1:].reshape(28,28), cmap='Greys', interpolation='None')

    # correct answer is first value
    correct_label = our_own_dataset[item][0]
    # data is remaining values
    inputs = our_own_dataset[item][1:]

    # query the network
    outputs = n.query(inputs)
    print(outputs)

    # the index of the highest value corresponds to the label
    label = np.argmax(outputs)
    print("network says ", label)
    # append correct or incorrect to list
    if (label == correct_label):
        print("match!")
    else:
        print("no match!")
        pass
