import sys
from PyQt5.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QLabel, QApplication, QPushButton, QComboBox, QProgressBar
from PyQt5.QtGui import QPixmap
import random
import pyqtgraph as pg
import numpy as np
import cmath
import math
from PIL import Image


class Backpropagation(QWidget):
    def __init__(self):
        super().__init__()
        self.width = 50 * 50
        self.input_count = 6
        self.output_count = 4
        # Высчитываем количество нейронов скрытой сети
        # Для этого применяется эвристический метод:
        # k = sqrt(a, b), где k - кол-во скрытых нейронов, a - кол-во нейронов на 1 слое, b - кол-во нейронов на 3 слое

        self.hidden_count = 2 # math.ceil(math.sqrt(self.width * self.output_count)) # количество скрытых нейронов

        self.weights_0_1 = np.random.normal(0.0, 2 ** -0.5, (self.hidden_count, self.width))
        # self.weights_Layer_2 = np.random.normal(0, 0.2, (self.input_count, self.hidden_count))
        self.weights_1_2 = np.random.normal(0.0, 1, (self.output_count, self.hidden_count))
        self.inputs_x = []
        self.baseball = []
        self.archery = []
        self.swimming = []
        self.hockey = []
        self.error_learn = []
        self.error_test = []
        self.count = 0
        self.iter = 1000
        self.init_ui()
        self.combo()

    def init_ui(self):

        layout2 = QHBoxLayout()
        layout3 = QHBoxLayout()
        layout4 = QVBoxLayout()
        layout5 = QVBoxLayout()
        self.btn = QPushButton("Выйти")
        self.comboBox = QComboBox()
        self.comboBox.addItems(["Стрельба из лука", "Бейсбол", "Плавание", "Хоккей"])
        self.progress = QProgressBar()
        self.progress.setMinimum(0)
        self.progress.setMaximum(self.iter * 4)
        self.label_pic = QLabel("")
        self.label_pic.setPixmap(QPixmap("archery.jpg"))
        self.btn_learn = QPushButton("Провести обучение")
        self.view = pg.PlotWidget()
        self.view.addLegend()
        self.curve2 = self.view.plot(pen='y', name="Ошибка тестирования")
        self.curve = self.view.plot(pen='r', name="Ошибка при обучении")
        self.view.showGrid(True, True)
        self.label_teor = QLabel("Теоретическое значение:")
        self.label_pract = QLabel("Полученное значение:")
        self.archery = self.get_pixel(r'pictures/archery.bmp')
        self.baseball = self.get_pixel(r'pictures/baseball.bmp')
        self.swimming = self.get_pixel(r'pictures/swimming.bmp')
        self.hockey = self.get_pixel(r'pictures/hockey.bmp')

        self.comboBox.addItems(["Зашумленная стрельба из лука",
                                "Зашумленный бейсбол",
                                "Зашумленное плавание",
                                "Зашумленный хоккей"])

        layout2.addWidget(self.comboBox)
        layout2.addWidget(self.label_pic)

        layout5.addWidget(self.btn_learn)
        layout5.addWidget(self.progress)
        layout5.addLayout(layout2)
        layout5.addWidget(self.label_teor)
        layout5.addWidget(self.label_pract)

        layout3.addLayout(layout5)
        layout3.addWidget(self.view)

        layout4.addLayout(layout3)
        layout4.addWidget(self.btn)
        self.setLayout(layout4)

        self.setGeometry(100, 50, 1200, 600)
        self.show()

    def combo(self):
        self.comboBox.activated[str].connect(self.on_activated)
        self.btn.clicked.connect(self.quit)
        self.btn_learn.clicked.connect(self.check)

    def check(self):
        self.inputs_x = []
        self.broke_archery = self.broke(self.archery)
        self.broke_baseball = self.broke(self.baseball)
        self.broke_swimming = self.broke(self.swimming)
        self.broke_hockey = self.broke(self.hockey)

        self.create_pic(self.archery, r'pictures/archery.jpg')
        self.create_pic(self.baseball, r'pictures/baseball.jpg')
        self.create_pic(self.swimming, r'pictures/swimming.jpg')
        self.create_pic(self.hockey, r'pictures/hockey.jpg')
        self.create_pic(self.broke_archery, r'pictures/br_archery.jpg')
        self.create_pic(self.broke_baseball, r'pictures/br_baseball.jpg')
        self.create_pic(self.broke_swimming, r'pictures/br_swimming.jpg')
        self.create_pic(self.broke_hockey, r'pictures/br_hockey.jpg')

        # self.weights_0_1 = np.random.normal(0.0, 2 ** -0.5, (self.hidden_count, self.width))
        # self.weights_Layer_2 = np.random.normal(0, 0.1, (self.input_count, self.hidden_count))
        # self.weights_1_2 = np.random.normal(0.0, 1, (self.output_count, self.hidden_count))

        self.inputs_x.append(self.archery)
        self.inputs_x.append(self.baseball)
        self.inputs_x.append(self.swimming)
        self.inputs_x.append(self.hockey)

        self.predict(self.inputs_x)
        self.random_plot()
        self.error_learn = []
        self.error_test = []

    def random_plot(self):
        self.curve.setData(self.error_learn)
        self.curve2.setData(self.error_test)

    def create_pic(self, data, name):
        img = Image.new('RGB', (50, 50), color=0)
        pix_tmp = data
        k = 0
        for i in range(int(self.width ** (0.5))):
            for j in range(int(self.width ** (0.5))):
                if pix_tmp[k] == 1:
                    img.putpixel((i, j), (255, 255, 255))
                else:
                    img.putpixel((i, j), (0, 0, 0))
                k = k + 1
        img = img.resize((250, 250), Image.ANTIALIAS)
        img.save(str(name))

    def on_activated(self):
        if self.comboBox.currentIndex() == 0:
            self.label_pic.setPixmap(QPixmap(r'pictures/archery.jpg'))
        if self.comboBox.currentIndex() == 1:
            self.label_pic.setPixmap(QPixmap(r'pictures/baseball.jpg'))
        if self.comboBox.currentIndex() == 2:
            self.label_pic.setPixmap(QPixmap(r'pictures/swimming.jpg'))
        if self.comboBox.currentIndex() == 3:
            self.label_pic.setPixmap(QPixmap(r'pictures/hockey.jpg'))
        if self.comboBox.currentIndex() == 4:
            self.label_pic.setPixmap(QPixmap(r'pictures/br_archery.jpg'))
        if self.comboBox.currentIndex() == 5:
            self.label_pic.setPixmap(QPixmap(r'pictures/br_baseball.jpg'))
        if self.comboBox.currentIndex() == 6:
            self.label_pic.setPixmap(QPixmap(r'pictures/br_swimming.jpg'))
        if self.comboBox.currentIndex() == 7:
            self.label_pic.setPixmap(QPixmap(r'pictures/br_hockey.jpg'))
        self.predict_pic()

    def predict_pic(self):
        if self.comboBox.currentIndex() == 0:
            out = self.fow(self.archery, forward=True)
            y = [1, 0, 0, 0]
        if self.comboBox.currentIndex() == 1:
            out = self.fow(self.baseball, forward=True)
            y = [0, 1, 0, 0]
        if self.comboBox.currentIndex() == 2:
            out = self.fow(self.swimming, forward=True)
            y = [0, 0, 1, 0]
        if self.comboBox.currentIndex() == 3:
            out = self.fow(self.hockey, forward=True)
            y = [0, 0, 0, 1]
        if self.comboBox.currentIndex() == 4:
            out = self.fow(self.broke_archery, forward=True)
            y = [1, 0, 0, 0]
        if self.comboBox.currentIndex() == 5:
            out = self.fow(self.broke_baseball, forward=True)
            y = [0, 1, 0, 0]
        if self.comboBox.currentIndex() == 6:
            out = self.fow(self.broke_swimming, forward=True)
            y = [0, 0, 1, 0]
        if self.comboBox.currentIndex() == 7:
            out = self.fow(self.broke_hockey, forward=True)
            y = [0, 0, 0, 1]
        if out.index(max(out)) == 0:
            self.label_pract.setText("Полученное значение на выходе:" + str(out))
        if out.index(max(out)) == 1:
            self.label_pract.setText("Полученное значение на выходе:" + str(out))
        if out.index(max(out)) == 2:
            self.label_pract.setText("Полученное значение на выходе:" + str(out))
        if out.index(max(out)) == 3:
            self.label_pract.setText("Полученное значение на выходе:" + str(out))
        self.label_teor.setText("Теоретическое значение на выходе:" + str(y))

    def quit(self):
        exit()

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # def activation(self, x, derive=False):
    #     if derive:
    #         i = 1 / (cmath.cosh(x)) ** 2
    #         return i.real
    #     return cmath.tanh(x).real
    def activation(self, x, derive=False):
        if derive:
            return self.sigmoid(x) * (1 - self.sigmoid(x))
        return self.sigmoid(x)

    def broke(self, data):
        temp = data.copy()
        changed = []
        chaff = 0.2
        for element in range(0, int(self.width * chaff)):
            index = random.randint(0, self.width - 1)
            if index not in changed:
                changed.append(index)
                if index >= temp.count(self):
                    print(index)
                if temp[index] == 1:
                    temp[index] = 0
                else:
                    temp[index] = 1
            else:
                element = element - 1
        return temp

    def get_pixel(self, pic):
        inputs_data = []
        picture = Image.open(pic)
        picture = picture.convert('RGB')
        (width, height) = picture.size
        self.width = width * height
        for x in range(width):
            for y in range(height):
                if picture.getpixel((x, y)) == (255, 255, 255):
                    inputs_data.append(1)
                else:
                    inputs_data.append(0)
        return inputs_data

    def predict(self, X):
        func_error = []
        test_error = []
        self.count = 0
        s = 1
        tmp_error = 0
        tmp_error2 = 0
        for g in range(self.iter):
            for ele in X:
                tmp = 0
                if s == 1:
                    y1 = np.array([1, 0, 0, 0])
                elif s == 2:
                    y1 = np.array([0, 1, 0, 0])
                elif s == 3:
                    y1 = np.array([0, 0, 1, 0])
                elif s == 4:
                    y1 = np.array([0, 0, 0, 1])
                    s = 0

                layer_1_output, output2 = self.fow(ele)

                broken_error = self.fow(self.broke_archery, forward=True)
                for i in range(len(broken_error)):
                    tmp += ([1, 0, 0, 0][i] - broken_error[i]) ** 2

                broken_error = self.fow(self.broke_baseball, forward=True)
                for i in range(len(broken_error)):
                    tmp += ([0, 1, 0, 0][i] - broken_error[i]) ** 2

                broken_error = self.fow(self.broke_swimming, forward=True)
                for i in range(len(broken_error)):
                    tmp += ([0, 0, 1, 0][i] - broken_error[i]) ** 2

                broken_error = self.fow(self.broke_hockey, forward=True)
                for i in range(len(broken_error)):
                    tmp += ([0, 0, 0, 1][i] - broken_error[i]) ** 2

                test_error.append(tmp)
                tmp_error2 += tmp
                temp = self.learn(ele, layer_1_output, output2, y1)
                func_error.append(temp)
                tmp_error += temp
                s += 1
                self.count += 1
                if ((s + 1) % 4) == 0:
                    self.error_learn.append(tmp_error)
                    self.error_test.append(tmp_error2)
                    tmp_error = 0
                    tmp_error2 = 0
                    self.random_plot()
                self.progress.setValue(self.count)
                QApplication.processEvents()
        self.error_learn = []
        self.error_test = []

    def fow(self, inputs, forward=False):

        # Первый слой
        layer_1_input = np.dot(self.weights_0_1, inputs)
        layer_1_output = np.array([self.activation(x) for x in layer_1_input])

        # # Второй слой
        # layer_2_input = np.dot(layer_1_output, self.weights_Layer_2)
        # layer_2_output = np.array([self.activation(x) for x in layer_2_input])

        # Третий слой
        output = np.dot(self.weights_1_2, layer_1_output)
        output2 = np.array([self.activation(x) for x in output])
        if forward:
            out = [round(el, 2) for el in output2.real]
            return out
        else:
            return layer_1_output, output2

    def learn(self, inputs, layer_1_output, output, y):

        error = []
        delta = []
        delta_1 = []
        delta_2 = []
        learn_rate = 0.5

        # ----------------- Вычисление ошибки внешнего слоя ------------------- #
        for n in range(len(output)):
            error.append(y[n] - output[n])
            delta.append(error[n] * self.activation(output[n], derive=True))

            #  ----------------- Вычисление ошибки скрытого слоя ---------------  #
        # for k in range(len(layer_2_output)):
        #     tmp = 0
        #     for i in range(len(error)):
        #         tmp += delta[i] * self.weights_Layer_output[k][i]
        #     delta_1.append(tmp * self.activation(layer_2_output[k], derive=True))

        for k in range(len(layer_1_output)):
            tmp = 0
            for i in range(len(delta_1)):
                tmp += delta_1[i] * self.weights_1_2[k][i]
            delta_2.append(tmp * self.activation(layer_1_output[k], derive=True))

            # ------------------ Корректировка весовых коэффициентов --------------------- #
        t = 0
        for el in self.weights_1_2:
            for j in range(len(el)):
                el[j] += layer_1_output[t] * delta[j] * learn_rate
            t += 1

        # k = 0
        # for el in self.weights_Layer_2:
        #     for i in range(len(layer_2_output)):
        #         el[i] += layer_1_output[k] * delta_1[i] * learn_rate
        #     k += 1

        k = 0
        for el1 in self.weights_0_1:
            for o in range(len(el1)):
                el1[o] += inputs[k] * delta_2[o] * learn_rate
            k += 1

        s_er = 0
        for i in range(len(y)):
            s_er += (y[i] - output[i]) ** 2

        return s_er


def main():
    app = QApplication(sys.argv)
    window = Backpropagation()
    window.show()
    app.exec_()

if __name__ == '__main__':
    main()
