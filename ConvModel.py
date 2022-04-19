import numpy as np

class ConvModel:
    """
    ConvModel
    Esta funcion utiliza un modelo convolucional para obtener caracteristicas.
    (Convierte imagenes en vectores)

    Attributes
    ----------
    dimensions : list
        dimensiones de entrada de las imagenes.

    Methods
    -------
    fit(X)
        transforma los datos introducidos en vectores
    """
    def __init__(self, sequential_model):
        self.dimensions = None
        self.__seqmodel = sequential_model
        self.__output_images = []

    def fit(self, X):
        """
        transforma los datos introducidos en vectores

        Parameters
        ----------
        X : numpy-array
            imagenes en formato (imagenes, alto, ancho, capas)
        """
        self.dimensions = X.shape
        images, height, weight, layers = self.dimensions
        print("fitting model... converting images to vectors")
        for i in range(0, images, 1):
            output = self.__seqmodel.predict(np.expand_dims(X[i, :, :, :], axis=0))
            self.__output_images.append(output)
        self.__output_images = np.vstack(self.__output_images)
        
    def get_features(self):
        """
        Obtiene las caracteristicas generadas en un arreglo.
        """
        return self.__output_images





