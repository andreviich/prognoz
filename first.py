import numpy as np
import matplotlib.pyplot as plt
from gaus import Metod_Gaussa
from sympy import *


import numpy

# Для первого задания
def least_squares(points, axis=None):
   """
   Функция для аппроксимации массива точек прямой, основанная на
   методе наименьших квадратов.

   :param points: Входной массив точек формы [N, 2]
   :return: Numpy массив формы [N, 2] точек на прямой
   """

   x = points[:, 0]
   y = points[:, 1]
   # Для метода наименьших квадратов нам нужно, чтобы X был матрицей,
   # в которой первый столбец - единицы, а второй - x координаты точек
   X = np.vstack((np.ones(x.shape[0]), x)).T
   normal_matrix = np.dot(X.T, X)
   moment_matrix = np.dot(X.T, y)
   # beta_hat это вектор [перехват, наклон], рассчитываем его в
   # в соответствии с формулой.
   beta_hat = np.dot(np.linalg.inv(normal_matrix), moment_matrix)
   intercept = beta_hat[0]
   slope = beta_hat[1]
   # Теперь, когда мы знаем параметры прямой, мы можем
   # легко вычислить y координаты точек на прямой.
   y_hat = intercept + slope * x
   # Соберем x и y в единую матрицу, которую мы собираемся вернуть
   # в качестве результата.
   points_hat = np.vstack((x,y, y_hat)).T
   # plt.scatter(x,y)
   # plt.plot(x, y_hat)
   # plt.show()

   k = (y_hat[0] - y_hat[1]) / (x[0] - x[1])
   b = y_hat[1] - k*x[0]
   return (points_hat, "y = %.2f*x + %.2f" % (k, b), np.vstack((x,y)).T, [x, y], [x, y_hat])


if __name__ == '__main__':

   init = least_squares(np.asarray([[1,2],[4,6],[8,10]]))
   #1.a
   print(init[0])
   #1.b
   print("Уравнение прямой, проходящей через эти точки:")
   print(init[1])
   #1.c
   print("Дисперсия:")
   print(init[2].var())
   plt.scatter(init[3][0],init[3][1])
   plt.plot(init[4][0],init[4][1])
   plt.show()

# Для второго задания


x = symbols('x')
K = 3 #степень многочлена

#опорные точки
xi = [1, 1.25, 1.25, 1.5, 1.75, 2]
yi = [5.47863093565943e-837, 0.000244140625000000, 1, 0.0156250000000000, 0.177978515625000, 1]

def approksim(xi, yi, K):
    #поиск констант
    A = numpy.zeros([K + 1, K + 1])
    for i in range(K + 1):
        for j in range(K + 1):
            for t in xi: A[i, j] += t ** (2 * K - i - j)
    a = numpy.zeros([K + 1])
    for i in range(K + 1):
        for j in range(len(xi)): a[i] += yi[j] * xi[j] ** (K - i)
    const = []
    for j in Metod_Gaussa(A, a): const.append(float(j))
    #составление уравнения
    f = 0
    for i in range(len(const)):
        f += const[i] * x ** (len(const) - 1 - i)
    return f

#значения квадратичной функции, полученной МНК, в тех же точках
f2 = []
for i in xi: f2.append(approksim(xi, yi, 2).evalf(subs={'x': i}))
#отклонения (невязки в точках)
e2 = []
for i in range(len(xi)): e2.append(abs(f2[i] - yi[i]))

xfi = numpy.linspace(xi[0], xi[len(xi) - 1], 100)

f2i = [approksim(xi, yi, 2).subs(x, a) for a in xfi]
#2.a
print('Задание 2.а')
print('Массив точек')
print(xi, yi, f2i)



plt.plot(xfi, f2i, 'r')
plt.scatter(x=xi, y=yi)

plt.grid()
plt.show()

#2.b

print('Задание 2.б')
def calc_parabola_vertex(x1, y1, x2, y2, x3, y3):
    '''
    Adapted and modifed to get the unknowns for defining a parabola:
    http://stackoverflow.com/questions/717762/how-to-calculate-the-vertex-of-a-parabola-given-three-points
    '''

    denom = (x1-x2) * (x1-x3) * (x2-x3);
    A     = (x3 * (y2-y1) + x2 * (y1-y3) + x1 * (y3-y2)) / denom;
    B     = (x3*x3 * (y1-y2) + x2*x2 * (y3-y1) + x1*x1 * (y2-y3)) / denom;
    C     = (x2 * x3 * (x2-x3) * y1+x3 * x1 * (x3-x1) * y2+x1 * x2 * (x1-x2) * y3) / denom;
    print(x1, y1, x2, y2, x3, y3)
    return A,B,C

kfs = calc_parabola_vertex(xfi[0], f2i[0],xfi[1], f2i[1],xfi[2], f2i[2])

print(f"y = {kfs[0]}*x^2 + ({kfs[1]}*x) + ({kfs[2]})")

# 2.б


# Делаем единый массив из точек

arr = [[x,y] for x,y in zip(xi, yi)]

# 2.в
print(f"Дисперсия: {np.asarray(arr).var()}")

