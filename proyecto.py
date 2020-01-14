# -*- coding: utf-8 -*-

import glob
import numpy as np
import pandas as pd
from keras.preprocessing import image
from keras_vggface.utils import preprocess_input

"""
import cv2

from keras.layers import Input, Dense, GlobalMaxPool2D, GlobalAvgPool2D, Dropout
from keras.models import Model

from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace

!pip install git+https://github.com/rcmalli/keras-vggface.git
"""

# Establecer semilla
np.random.seed(1)

def read_image(path):
    """
    Funcion que permite leer imagenes a partir de un archivo
    Args:
        path: Ruta de la imagen
    """
    img = image.load_img(path)
    img = np.array(img).astype(np.float)
    return preprocess_input(img, version=2)


def read_family_members_images(data_path):
    """
    Funcion que procesa la ruta especificada como parametro. Obtiene una lista
    con los miembros de cada familia, la cual tendra el formato "FXXXX/MIDY",
    y un diccionario con las rutas de las imagenes de cada miembro de cada familia.

    Args:
        data_path: Ruta de los archivos a procesar.
    
    Return:
        Devuelve una lista con los miembros de cada familia y un diccionario con
        las imagenes de cada miembro de cada familia.
    """
    # Leer la ruta proporcionada y obtener todos los directorios
    # Cada directorio esta asociado a una familia
    dirs = sorted(list(glob.glob(data_path + "*")))

    # Obtener los nombres de los directorios de las familias
    family_dirs = np.array([dir.split("/")[-1] for dir in dirs])

    # Obtener imagenes asociadas a cada directorio
    images = {f"{family}/{member.split('/')[-1]}": sorted(list(glob.glob(member + "/*.jpg")))
        for family in family_dirs for member in sorted(list(glob.glob(f"{data_path}{family}/*")))
    }
    
    family_members_list = list(images.keys())
    return family_members_list, images



def generate_datasets(families, test_prop=0.2, val_prop=0.2):
    """
    Funcion que permite generar los datasets de train, test y validacion
    a partir de un array de directorios, los cuales representan las familias.
    Los datos son mezclados para que se escoja de forma aleatoria.

    Args:
        families: Array con los directorios de las familias.
        test_prop: Proporcion de los datos totales que tienen que estar en el
                   conjunto de test.
        val_prop: Proporcion de los (datos_totales - datos_test) que tienen que
                  estar en el conjunto de validacion.
    
    Return:
        Devuelve un array con los directorios de las familias que forman el conjunto
        de train, otro para el conjunto de validacion y otro para el conjunto de test.
    """
    # Mezclar familias
    shuffle_families = np.copy(families)
    np.random.shuffle(shuffle_families)

    # Obtener la ultima proporcion de las familias y guardarla en el conjunto
    # de test
    idx_test = int(len(shuffle_families) * (1 - test_prop))
    test_dirs = shuffle_families[idx_test:]
    shuffle_families = shuffle_families[:idx_test]

    # Volver a mezclar familias para escoger conjunto de validacion
    np.random.shuffle(shuffle_families)

    # Obtener la ultima proporcion de las familias y guardarla en el conjunto
    # de validacion
    idx_val = int(len(shuffle_families) * (1 - val_prop))
    val_dirs = shuffle_families[idx_val:]
    
    # Obtener el resto de directorios y guardarlos en el conjunto de train
    train_dirs = shuffle_families[:idx_val]

    print(train_dirs.shape)
    print(val_dirs.shape)
    print(test_dirs.shape)

    return train_dirs, val_dirs, test_dirs


def batch_generator(dataset, relationships_path, batch_size=64, relationships_prop=0.2):
    relationships = pd.read_csv(relationships_path)
    relationships = list(zip(relationships.p1.values, relationships.p2.values))
    
    while True:
        left_images = []
        right_images = []
        targets = []
    
        # Elegir los 1's
        while len(left_images) < int(batch_size*relationships_prop):
            # Escogemos una linea aleatoria del CSV
            ind = np.random.choice(relationships)

            # Comprobamos que los individuos estan en el train
            if ind[0] in dataset and ind[1] in dataset:
                # Elegimos aleatoriamente una imagen de esos individuos
                left_images.append( read_image( np.random.choice(dataset[ind[0]]) ) )
                right_images.append( read_image( np.random.choice(dataset[ind[1]]) ) )
                targets.append(1.)

        
        # Elegir los 0's
        while len(left_images) < int(batch_size):
            # Accedemos dos individuos diferentes aleatorios del dataset
            ind = np.random.choice(dataset, 2, replace=False)

            # Comprobamos si son parientes
            if (ind[0],ind[1]) in relationships or (ind[1],ind[0]) in relationships:
                # En caso afirmativo añadimos con etiqueta 1
                left_images.append( read_image( np.random.choice(ind) ) )
                right_images.append( read_image( np.random.choice(ind) ) )
                targets.append(1.)
            else:
                # En caso contrario con etiqueta 0
                left_images.append( read_image( np.random.choice(ind) ) )
                right_images.append( read_image( np.random.choice(ind) ) )
                targets.append(0.)

        print(left_images, right_images, targets)            
        yield [left_images, right_images], targets




train_folders_path = "data/train/"

train_relationships = "data/train_relationships.csv"


dirs, images = read_family_members_images(train_folders_path)
"""
for k, v in images.items():
    print(f"{k} -> {v}")
"""
train_dirs, val_dirs, test_dirs = generate_datasets(dirs)

batch_generator(train_dirs, train_relationships)