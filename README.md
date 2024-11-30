
# 📊 Proyecto 8: Predicción de empleados

## 📖 Descripción del Proyecto

Este proyecto aborda uno de los mayores retos de Recursos Humanos: la rotación de empleados. La pregunta central es: ¿qué factores influyen en que un empleado decida quedarse o irse de la empresa? Para responder a esta pregunta, hemos utilizado datos ficticios de encuestas, métricas de desempeño y otras características de los empleados para desarrollar un modelo predictivo que determine la probabilidad de que un empleado se vaya.

El análisis va más allá de los números y se enfoca en comprender el impacto de las decisiones empresariales en la vida de las personas, proponiendo mejoras que podrían ayudar a las empresas a ser mejores lugares de trabajo.

## 🎯 Objetivos del proyecto

- Construir un modelo de machine learning que prediga la retención o rotación de los empleados.
- Identificar los factores más relevantes que influyen en la decisión de un empleado de quedarse o irse.
- Proponer estrategias y recomendaciones basadas en los resultados del modelo para mejorar la retención de empleados.

## 🗂️ Estructura del Proyecto

El proyecto está organizado de la siguiente manera:

```bash
├── datos/                # Conjuntos de datos sin procesar y ya procesados
│   ├── output/           # Datos procesados y resultados finales
│   └── raw/              # Datos en bruto (sin procesar)
│
├── flask/                # Archivos relacionados con el despliegue en Flask
│
├── modelos/              # Modelos predictivos
│
├── notebooks/            # Notebooks con el contenido y análisis de datos
│
├── src/                  # Scripts para la limpieza y procesamiento de datos
│
├── streamlit/            # Archivos relacionados con la aplicación en Streamlit
│
├── README.md             # Descripción general del proyecto e instrucciones
└── requirements.txt      # Lista de dependencias del proyecto
```

## 🛠️ Instalación y Requisitos

Este proyecto utiliza [Python 3.12](https://docs.python.org/3.12/) y requiere las siguientes bibliotecas para la ejecución y análisis:

- [pandas 2.2.3](https://pandas.pydata.org/docs/)
- [matplotlib 3.9.2](https://matplotlib.org/stable/index.html)
- [seaborn 0.13.2](https://seaborn.pydata.org/tutorial.html)
- [scikit-learn 1.5.2](https://scikit-learn.org/stable/)
- [imbalanced-learn 0.12.4](https://imbalanced-learn.org/stable/)
- [streamlit 1.40.2](https://docs.streamlit.io/)
- [Flask 3.1.0](https://flask.palletsprojects.com/)

Para instalar las dependencias, puedes ejecutar el siguiente comando dentro de un entorno virtual:

```bash
pip install -r requirements.txt
```

## 📊 Resultados y Conclusiones

## 🔄 Próximos Pasos

- Implementar un sistema de monitoreo en la empresa que permita recopilar datos en tiempo real sobre la satisfacción y desempeño de los empleados para ajustar el modelo con datos actualizados.
- Probar técnicas de ensamblaje para combinar varios modelos y mejorar la precisión de las predicciones.

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Si deseas colaborar en este proyecto, por favor abre un pull request o una issue en este repositorio.

## ✒️ Autores

Iván Bravo - Autor principal del proyecto.