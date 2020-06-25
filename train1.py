import numpy as np
import scipy.special
import scipy.ndimage


class Neuron:

    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # задать кол-во узлов во входе, скрытом и выходе
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        # коэффициент обучения
        self.lr = learningrate

        self.wih = (np.random.rand(self.hnodes, self.inodes) - 0.5)
        self.who = (np.random.rand(self.onodes, self.hnodes) - 0.5)
        # Функция активации
        self.activation_function = lambda x: scipy.special.expit(x)

    def train(self, inputs_list, targets_list):
        # Преобразовать входные данные в список(2D)
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T

        # Входящие сигналы для скрытого слоя
        hidden_inputs = np.dot(self.wih, inputs)
        # Исходящие сигналы для скрытого слоя
        hidden_outputs = self.activation_function(hidden_inputs)

        # Входящие сигналы для выхода
        final_inputs = np.dot(self.who, hidden_outputs)
        # Исходящие сигналы для выхода
        final_outputs = self.activation_function(final_inputs)

        # ошибка = целевое значение - фактическое значение
        output_errors = targets - final_outputs

        # ошибки скрытого слоя - это ошибки output_errors,
        # распределенные пропорционально весовым коэффициентам связей
        # и рекомбинированные на скрытых узлах
        hidden_errors = np.dot(self.who.T, output_errors)

        # обновить весовые коэффициенты связей между скрытым и выходным слоями
        self.who += self.lr * np.dot((output_errors * final_outputs * (1 - final_outputs)),
                                     np.transpose(hidden_outputs))

        # обновить весовые коэффициенты связей между скрытым и входным слоями
        self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1 - hidden_outputs)), np.transpose(inputs))

        return hidden_errors

    # сохранение сети(весов)
    def wih_who(self):
        return self.wih

    def who_wih(self):
        return self.who


# кол-во входных, скрытых и выходных узлов
input_nodes = 784
hidden_nodes = 200
output_nodes = 10

# коэфф. обучения
learning_rate = 0.1

# экземпляр нейронной сети
n = Neuron(input_nodes, hidden_nodes, output_nodes, learning_rate)

# загрузка в тренировочный список данных MNIST
training_data_file = open("mnist_dataset/mnist_train.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

# тренеровка нейронки
epochs = 5
# перебор всех записей в тренировочном наборе данных
for e in range(epochs):
    for record in training_data_list:
        all_values = record.split(',')
        inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        targets = np.zeros(output_nodes) + 0.01
        targets[int(all_values[0])] = 0.99
        n.train(inputs, targets)

print('train end')
# сохранение сети(весов)
i = n.wih_who()
j = n.who_wih()

print(i)
print()
print(j)

np.save('weight/wih', i)
print('---wih saved---')
np.save('weight/who', j)
print("---who saved---")
