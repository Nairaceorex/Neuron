import numpy as np
import scipy.special
import imageio
import glob


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

        self.wih = np.load("weight/wih.npy")
        self.who = np.load("weight/who.npy")

        # Функция активации
        self.activation_function = lambda x: scipy.special.expit(x)

    def query(self, inputs_list):
        # Преобразовать входные данные в список
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
# журнал значений
our_dataset = []

for image_file_name in glob.glob('test_img/img_?.png'):
    print("loading ... ", image_file_name)
    # id объекта
    label = int(image_file_name[-5:-4])
    #  загружаем изображение
    img_array = imageio.imread(image_file_name, as_gray=True)
    # интвертируем цвета и задаем размерность массива
    img_data = 255.0 - img_array.reshape(28, 28)
    # переводим  rgb в значения от 0.01 до 0.99(значения сигмоиды)
    img_data = (img_data / 255.0 * 0.99) + 0.01
    # агружаем в журнал
    record = np.append(label, img_data)
    print(record)
    our_dataset.append(record)

for item in range(len(our_dataset)):

    # значение правильного ответа
    correct_label = our_dataset[item][0]
    # установка входных данных
    inputs = our_dataset[item][1:]

    # опрос сети
    outputs = n.query(inputs)
    print(outputs)

    # наибольшая вероятность по мнению сети
    label = np.argmax(outputs)
    print("network answer ", label)
    # append correct or incorrect to list
    if (label == correct_label):
        print("correct!")
    else:
        print("do not correct!")
