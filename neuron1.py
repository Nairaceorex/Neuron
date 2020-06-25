import numpy as np
import scipy.special


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

test_data_file = open("mnist_dataset/mnist_test.csv", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

# тест нейронки

# журнал оценок работы сети
scorecard = []

# перебрать записи в тестовом наборе
for record in test_data_list:
    # получить список из записей
    arr_mnist = record.split(',')
    # правильный ответ - первое значение
    correct_label = int(arr_mnist[0])
    print(correct_label, ' true mark')
    # масштабировать и сместить входные значения
    inputs = (np.asfarray(arr_mnist[1:]) / 255.0 * 0.99) + 0.01
    # опрос сети
    outputs = n.query(inputs)
    # индекс наибольшего значения - маркерное значение
    label = np.argmax(outputs)
    print(label, 'answer network')
    # присоединить оценку ответа сети к концу списка
    if (label == correct_label):
        # в случае правильного ответа сети присоединить к списку значение 1
        scorecard.append(1)
    else:
        # в случае правильного ответа сети присоединить к списку значение 1
        scorecard.append(0)

print(scorecard)
scorecard_array = np.asarray(scorecard)
print('efficience:', scorecard_array.sum() / scorecard_array.size)
