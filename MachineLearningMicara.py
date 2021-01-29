import numpy as np
import matplotlib.pyplot as plt
import h5py
import PIL
import scipy
import scipy.misc

from PIL import Image
from scipy import ndimage

import cv2
import os


def load_images_from_folder(folder, size, bandera):
    """
       Carga las imagenes las comprime, estandariza y aplica efectos

        """
    images = []
    size = size
    count = 0
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))

        if img is not None:
            img = cv2.resize(img, size)
            img = img / 255  # esto centra y estandariza la database, aveces tambien restas el promedio
            # print(filename)
            images.append(img)
            if bandera:  # agregamos efectos para lograr mayor cantidad de entradas
                img2 = cv2.stylization(img, sigma_s=150, sigma_r=0.25)
                images.append(img2)
                img2 = cv2.flip(img, 1)
                images.append(img2)
        if count == 810:
            break
        count = count + 1

    return images


def sigmoid(z):
    """
    En este caso de clasificacion binaria vamos a utilizar sigmoid
    """
    s = 1 / (1 + np.exp(-z))
    return s


def initialize_with_zeros(dim):
    """
    En el caso de gradient descent no es nescesario inizializar de manera aleatoria
    de esta manera  tenemos lo siguiente
     """
    w = np.zeros((dim, 1), dtype=int)
    b = 0
    assert (w.shape == (dim, 1))
    assert (isinstance(b, float) or isinstance(b, int))
    return w, b


def propagate(w, b, X, Y):
    """
    Foward Propagation, de izquierda a derecha, estamos calculando una primera aproximacion
    Argumentos:
    w -- weights, numpay array de (num_px * num_px * 3, 1)
    b -- bias, escalar
    X -- nuestras entradas (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (contiene 0 si no es Ignacio, 1 si es Ignacio) tamaño(1, number of examples)

    Backwards, derecha a izquierda:
    cost -- el costo de la logist regression
    dw -- gradiente del costo con respecto w, las dimecniones son las misma que W
    db -- gradiente del costo con respecto b, las dimecniones son las misma que b

    """
    m = X.shape[1]
    # FORWARD PROPAGATION (FROM X TO COST)
    h = np.dot(w.T, X) + b
    z = h
    ac = 1 / (1 + np.exp(-z))  # compute activation
    cost1 = np.dot((1 - Y), (np.log(1 - ac)).T)
    cost2 = np.dot(Y, (np.log(ac)).T)  # compute cost
    cost = -1 / m * (cost1 + cost2)

    # BACKWARD PROPAGATION (TO FIND GRAD)

    dz = (ac - Y)
    db = np.sum(dz) / m
    dw = np.dot(X, dz.T) * 1 / m  # esto funciona aca por el tipo de W,en otros caso el correcto es cambiado dz y X


    assert (dw.shape == w.shape)
    assert (db.dtype == float)
    cost = np.squeeze(cost)
    # assert(cost.shape == ())

    grads = {"dw": dw,
             "db": db}

    return grads, cost


def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost=False):
    """
    Funcion de optimizacion de w y b
    This function optimizes w and b by running a gradient descent algorithm


    Argumentos:
    w -- weights, numpay array de (num_px * num_px * 3, 1)
    b -- bias, escalar
    X -- nuestras entradas (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (contiene 0 si no es Ignacio, 1 si es Ignacio) tamaño(1, number of examples)
    num_iterations -- numero de iterecion en el loop de optimizacion
    learning_rate -- es la constante que va en la optimizacion
    print_cost -- si queremos ver o no el costo cada 100 repeticiones

    Retorna:
    params -- diccionario conteniendo w y b
    grads -- diccionario que contiene los pesos y bias del gradient descent con respecto a la funcion de costo
    costs -- lista de los costos computados
    """
    costs = []
    for i in range(num_iterations):
        # Cost and gradient calculation

        grads, cost = propagate(w, b, X, Y)
        a = learning_rate

        dw = grads["dw"]
        db = grads["db"]

        # update rule
        w = w - a * dw
        b = b - a * db

        # Record the costs
        if i % 100 == 0:
            costs.append(cost)

        # Print the cost every 100 training iterations
        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))

    params = {"w": w,
              "b": b}

    grads = {"dw": dw,
             "db": db}

    return params, grads, costs


def predict(w, b, X):
    '''
   Utilizando todo lo caclulado usamos la activacion para determinar si la salida es "1" o "0"
    '''

    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)

    h = np.dot(w.T, X) + b
    z = h
    A = 1 / (1 + np.exp(-z))
    for i in range(A.shape[1]): # se puede utilizar otra funcion con trheshold, que esta mejor optimizada
        # Convert probabilities A[0,i] to actual predictions p[0,i]

        if A[0, i] <= 0.5:
            Y_prediction[0, i] = 0
        elif A[0, i] > 0.5:
            Y_prediction[0, i] = 1
    assert (Y_prediction.shape == (1, m))

    return Y_prediction


def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):
    """
    Utilza todas las funciones atneriores
    """



    # initialize parameters with zeros
    w, b = initialize_with_zeros(X_train.shape[0])

    # Gradient descent
    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost=True)


    w = parameters["w"]
    b = parameters["b"]

    # Predict test/train set examples
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)



    # Print train/test Errors
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test,
         "Y_prediction_train": Y_prediction_train,
         "w": w,
         "b": b,
         "learning_rate": learning_rate,
         "num_iterations": num_iterations}

    return d



#######################################################################################################################
testFotosYo = load_images_from_folder('C:/Users/Ignac/Desktop/Seminario/Fotos para reconocer/New folder', (64, 64),
                                      True)
fotosMias = load_images_from_folder('C:/Users/Ignac/Desktop/Seminario/Fotos para reconocer/Fotos mias', (64, 64), True)

# estos training sets no son arrays podria cambiar la funcion lo mejor seria cargalos directamente en hdf5
imagen_training_gente = load_images_from_folder('C:/Imagenes para ML/Gente/part1', (64, 64), 0)
imagen_test_gente = load_images_from_folder('C:/Imagenes para ML/Gente/test', (64, 64), 0)

index = 5
######################### convierto en array las matrices
fotosMiasArray = np.array(fotosMias[:])
fotosDeGente = np.array(imagen_training_gente[:])
m_train = fotosMiasArray.shape[0]
y_yo = np.ones((1, m_train))
y_no = np.zeros((1, fotosDeGente.shape[0]))
fotosMiasArray = fotosMiasArray.reshape(fotosMiasArray.shape[0], -1).T
fotosDeGente = fotosDeGente.reshape(fotosDeGente.shape[0], -1).T
train_set_y = np.append(y_yo, y_no)
train_set_x = np.append(fotosMiasArray, fotosDeGente, axis=1)
# myimagen = cv2.resize(image[index], (64, 64))
print("el tamaño de la fotosMiasArray es " + str(fotosMiasArray.shape))
print("cantidad de fotos mias" + str(np.shape(fotosMiasArray[1])))
print("el tamaño de la matriz fotosDeGente es " + str(fotosDeGente.shape))
print("el tamaño de la matriz  trainin_set es " + str(train_set_x.shape))
# train set

###################################### test set
image_test = np.array(testFotosYo[:])
image2_test = np.array(imagen_test_gente[:])
print("*****el tamaño de la matriz  image_test  " + str(image_test.shape))
print("el tamaño de la matriz  image2_test" + str(image2_test.shape))
y_yo_test = np.ones((1, image_test.shape[0]))
y_no_test = np.zeros((1, image2_test.shape[0]))

image_test = image_test.reshape(image_test.shape[0], -1).T
print("el tamaño de la matriz  image_set es " + str(fotosMiasArray.shape))
image2_test = fotosDeGente.reshape(image2_test.shape[0], -1).T

test_set_y = np.append(y_yo_test, y_no_test)
print("el tamaño de la matriz test es " + str(image_test.shape) + "***" + str(image2_test.shape) +
      "train_set_x " + str(train_set_y.shape))

test_set_x = np.append(image_test, image2_test, axis=1)

print("el tamaño de la matriz test_set es " + str(test_set_x.shape))
# a,length, height, depth = image.shape
# no sirve por que todo esta desarmado
# plt.imshow(image[index])
# plt.show()

print("Vectorizando las matrices")

print("train_set_x_flatten shape: " + str(train_set_x.shape))
print("train_set_Y_flatten shape: " + str(train_set_y.shape))
########################predictions#####################################################################################
d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations=1000, learning_rate=0.005, print_cost=True)

imagenes = load_images_from_folder('C:/Users/Ignac/Desktop/Seminario/Fotos para reconocer/Predecir', (64, 64), 0)

fotosPredecir = np.array(imagenes[:])
print("antes decambiar" + str(fotosPredecir.shape))
print("matriz de fotos parapredecir" + str(fotosPredecir.shape))
fotosPredecir = fotosPredecir.reshape(fotosPredecir.shape[0], -1).T

print("las fotosPredecir quedan de:" + str(fotosPredecir.shape))
print("las fotosPredecir cantidad:" + str(fotosPredecir.shape[1]))
print("una foto es:" + str(fotosPredecir[:, 1].shape))
y_resultado = predict(d["w"], d["b"], fotosPredecir)
print("y = " + str(np.squeeze(y_resultado)) + "" + "\" picture." + "shape:" + str(y_resultado.shape))
count = 0
# tengo que recorrer en el eje de las muestras
for i in range(fotosPredecir.shape[1]):
    y_resultado = predict(d["w"], d["b"],
                          np.array(fotosPredecir[:, i], ndmin=2).T)  # fuerzo a que no aparezca el tuplet
    reconstruida = np.array(fotosPredecir[:, i], ndmin=2).T.reshape(64, 64, 3)  # reconstruyo la imagen orignal

    plt.title({True: "Ignacio", False: "persona extraña"}[str(np.squeeze(y_resultado)) == '1.0'])
    plt.imshow(reconstruida)
    plt.show()
    print("y = " + str(np.squeeze(y_resultado)) + "" + "\" picture." + "shape:" + str(y_resultado.shape))
    count += 1 if str(np.squeeze(y_resultado)) == '1.0' else count + 0

# print("y = " + str(np.squeeze(my_predicted_image)) + "" + "\" picture." + "shape:" + str(my_predicted_image.shape))
