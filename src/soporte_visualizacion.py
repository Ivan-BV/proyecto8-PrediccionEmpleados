
import pandas as pd
import numpy as np
from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns
import math
from scipy.stats import zscore

# Tratamiento de datos
# -----------------------------------------------------------------------
import numpy as np
import pandas as pd

# Otros objetivos
# -----------------------------------------------------------------------
import math

# Gráficos
# -----------------------------------------------------------------------
import seaborn as sns
import matplotlib.pyplot as plt


# Para tratar el problema de desbalance
# -----------------------------------------------------------------------
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek

def relacion_vr_categoricas(dataframe: pd.DataFrame, variable_respuesta: str, paleta = "mako", tamanio_grafica = (15,10)):
    df_cat = Visualizador(dataframe).separar_dataframes()[1]
    cols_categoricas = df_cat.columns
    num_filas = math.ceil(len(cols_categoricas)/2)
    fig, axes = plt.subplots(nrows=num_filas, ncols=2, figsize=tamanio_grafica)
    axes = axes.flat

    for indice, columna in enumerate(cols_categoricas):
        datos_agrupados = dataframe.groupby(columna)[variable_respuesta].mean().reset_index().sort_values(variable_respuesta, ascending=False)
        sns.barplot(x = columna,
                    y = variable_respuesta,
                    data = datos_agrupados,
                    ax = axes[indice],
                    palette = paleta)
        axes[indice].tick_params(rotation=90)
        axes[indice].set_title(f"Relación entre {columna} y {variable_respuesta}")
        axes[indice].set_xlabel("")
    plt.tight_layout()

class Visualizador:
    """
    Clase para visualizar la distribución de variables numéricas y categóricas de un DataFrame.

    Attributes:
    - dataframe (pandas.DataFrame): El DataFrame que contiene las variables a visualizar.

    Methods:
    - __init__: Inicializa el VisualizadorDistribucion con un DataFrame y un color opcional para las gráficas.
    - separar_dataframes: Separa el DataFrame en dos subconjuntos, uno para variables numéricas y otro para variables categóricas.
    - plot_numericas: Grafica la distribución de las variables numéricas del DataFrame.
    - plot_categoricas: Grafica la distribución de las variables categóricas del DataFrame.
    - plot_relacion2: Visualiza la relación entre una variable y todas las demás, incluyendo variables numéricas y categóricas.
    """

    def __init__(self, dataframe):
        """
        Inicializa el VisualizadorDistribucion con un DataFrame y un color opcional para las gráficas.

        Parameters:
        - dataframe (pandas.DataFrame): El DataFrame que contiene las variables a visualizar.
        - color (str, opcional): El color a utilizar en las gráficas. Por defecto es "grey".
        """
        self.dataframe = dataframe

    def exploracion_datos(self):
        """
        Realiza una exploración básica de los datos en el DataFrame dado e imprime varias estadísticas descriptivas.

        Parameters:
        -----------
        dataframe : pandas DataFrame. El DataFrame que se va a explorar.

        Returns:
        --------
        None

        Imprime varias estadísticas descriptivas sobre el DataFrame, incluyendo:
        - El número de filas y columnas en el DataFrame.
        - El número de valores duplicados en el DataFrame.
        - Una tabla que muestra las columnas con valores nulos y sus porcentajes.
        - Las principales estadísticas de las variables numéricas en el DataFrame.
        - Las principales estadísticas de las variables categóricas en el DataFrame.

        """

        print(f"El número de filas es {self.dataframe.shape[0]} y el número de columnas es {self.dataframe.shape[1]}")

        print("\n----------\n")

        print(f"En este conjunto de datos tenemos {self.dataframe.duplicated().sum()} valores duplicados")

        
        print("\n----------\n")


        print("Los columnas con valores nulos y sus porcentajes son: ")
        dataframe_nulos = self.dataframe.isnull().sum()

        display((dataframe_nulos[dataframe_nulos.values >0] / self.dataframe.shape[0]) * 100)

        print("\n----------\n")
        print("Las principales estadísticas de las variables númericas son:")
        display(self.dataframe.describe().T)

        print("\n----------\n")
        print("Las principales estadísticas de las variables categóricas son:")
        display(self.dataframe.describe(include = "O").T)

        print("\n----------\n")
        print("Las características principales del dataframe son:")
        display(self.dataframe.info())

    def separar_dataframes(self):
        """
        Separa el DataFrame en dos subconjuntos, uno para variables numéricas y otro para variables categóricas.

        Returns:
        - pandas.DataFrame: DataFrame con variables numéricas.
        - pandas.DataFrame: DataFrame con variables categóricas.
        """
        return self.dataframe.select_dtypes(include=np.number), self.dataframe.select_dtypes(include=["O", "category"])
    
    def plot_numericas(self, color="grey", tamano_grafica=(15, 5)):
        """
        Grafica la distribución de las variables numéricas del DataFrame.

        Parameters:
        - color (str, opcional): El color a utilizar en las gráficas. Por defecto es "grey".
        - tamaño_grafica (tuple, opcional): El tamaño de la figura de la gráfica. Por defecto es (15, 5).
        """
        lista_num = self.separar_dataframes()[0].columns
        fig, axes = plt.subplots(nrows = 2, ncols = math.ceil(len(lista_num)/2), figsize=tamano_grafica, sharey=True)
        axes = axes.flat
        for indice, columna in enumerate(lista_num):
            sns.histplot(x=columna, data=self.dataframe, ax=axes[indice], color=color, bins=20)
        plt.suptitle("Distribución de variables numéricas")
        plt.tight_layout()

        if len(lista_num) % 2 !=0:
            fig.delaxes(axes[-1])


    def plot_categoricas(self, color="grey", tamano_grafica=(40, 10)):
        """
        Grafica la distribución de las variables categóricas del DataFrame.

        Parameters:
        - color (str, opcional): El color a utilizar en las gráficas. Por defecto es "grey".
        - tamaño_grafica (tuple, opcional): El tamaño de la figura de la gráfica. Por defecto es (15, 5).
        """
        lista_cat = self.separar_dataframes()[1].columns
        fig, axes = plt.subplots(2, math.ceil(len(lista_cat) / 2), figsize=tamano_grafica)
        axes = axes.flat
        for indice, columna in enumerate(lista_cat):
            sns.countplot(x=columna, data=self.dataframe, order=self.dataframe[columna].value_counts().index,
                          ax=axes[indice], color=color)
            axes[indice].tick_params(rotation=90)
            axes[indice].set_title(columna)
            axes[indice].set(xlabel=None)

        plt.suptitle("Distribución de variables categóricas")
        plt.tight_layout()

        if len(lista_cat) % 2 !=0:
            fig.delaxes(axes[-1])


    def plot_relacion(self, vr, tamano_grafica=(40, 12)):

        lista_num = self.separar_dataframes()[0].columns
        lista_cat = self.separar_dataframes()[1].columns

        fig, axes = plt.subplots(3, math.ceil(len(self.dataframe.columns) / 3), figsize=tamano_grafica)
        axes = axes.flat

        for indice, columna in enumerate(self.dataframe.columns):
            if columna == vr:
                fig.delaxes(axes[indice])
            elif columna in lista_num:
                sns.histplot(x = columna, 
                             hue = vr, 
                             data = self.dataframe, 
                             ax = axes[indice], 
                             palette = "magma", 
                             legend = False)
                
            elif columna in lista_cat:
                sns.countplot(x = columna,
                              hue = vr,
                              data = self.dataframe,
                              ax = axes[indice],
                              palette = "magma"
                              )

            axes[indice].set_title(f"Relación {columna} vs {vr}")   

        plt.tight_layout()
    
        
    def deteccion_outliers(self, color = "grey"):

        """
        Detecta y visualiza valores atípicos en un DataFrame.

        Params:
            - dataframe (pandas.DataFrame):  El DataFrame que se va a usar

        Returns:
            No devuelve nada

        Esta función selecciona las columnas numéricas del DataFrame dado y crea un diagrama de caja para cada una de ellas para visualizar los valores atípicos.
        """

        lista_num = self.separar_dataframes()[0].columns

        fig, axes = plt.subplots(2, ncols = math.ceil(len(lista_num)/2), figsize=(15,5))
        axes = axes.flat

        for indice, columna in enumerate(lista_num):
            sns.boxplot(x=columna, data=self.dataframe, 
                        ax=axes[indice], 
                        color=color, 
                        flierprops={'markersize': 4, 'markerfacecolor': 'orange'})

        if len(lista_num) % 2 != 0:
            fig.delaxes(axes[-1])

        
        plt.tight_layout()

    def correlacion(self, tamano_grafica = (7, 5)):

        """
        Visualiza la matriz de correlación de un DataFrame utilizando un mapa de calor.

        Params:
            - dataframe : pandas DataFrame. El DataFrame que contiene los datos para calcular la correlación.

        Returns:
        No devuelve nada 

        Muestra un mapa de calor de la matriz de correlación.

        - Utiliza la función `heatmap` de Seaborn para visualizar la matriz de correlación.
        - La matriz de correlación se calcula solo para las variables numéricas del DataFrame.
        - La mitad inferior del mapa de calor está oculta para una mejor visualización.
        - Permite guardar la imagen del mapa de calor como un archivo .png si se solicita.

        """

        plt.figure(figsize = tamano_grafica )

        mask = np.triu(np.ones_like(self.dataframe.corr(numeric_only=True), dtype = np.bool_))

        sns.heatmap(data = self.dataframe.corr(numeric_only = True), 
                    annot = True, 
                    vmin=-1,
                    vmax=1,
                    cmap="magma",
                    linecolor="black", 
                    fmt='.1g', 
                    mask = mask)
        

