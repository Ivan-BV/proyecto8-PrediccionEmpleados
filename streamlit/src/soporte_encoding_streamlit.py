import pandas as pd

# Para pruebas estadísticas
# -----------------------------------------------------------------------
from scipy.stats import chi2_contingency

from IPython.display import display

def detectar_orden_cat(df, lista_categoricas, var_res):
    lista_ordenas = []
    lista_desordenadas = []
    for categorica in lista_categoricas:
        print(f"Estamos evaluando la variable: {categorica.upper()}")
        df_cross_tab_gender = pd.crosstab(df[categorica], df[var_res])
        display(df_cross_tab_gender)

        chi2, p, dof, expected = chi2_contingency(df_cross_tab_gender)

        if p < 0.05:
            print(f"La variable categorica {categorica.upper()} si tiene orden\n")
            lista_ordenas.append(categorica)
        else:
            print(f"La variable categorica {categorica.upper()} no tiene orden\n")
            lista_desordenadas.append(categorica)

    return lista_ordenas, lista_desordenadas

def detectar_orden_cat_2(lista_categoricas):
    # Definir manualmente las categorías que tienen un orden lógico
    categorias_ordenadas = ['BusinessTravel', 'Department', 'EducationField', 'JobRole', 'MaritalStatus', 'EnvironmentSatisfaction', 'JobSatisfaction', 'WorkLifeBalance', 'JobInvolvement']
    
    lista_ordenas = []
    lista_desordenadas = []
    
    for categorica in lista_categoricas:
        if categorica in categorias_ordenadas:
            lista_ordenas.append(categorica)
        else:
            lista_desordenadas.append(categorica)
    
    return lista_ordenas, lista_desordenadas