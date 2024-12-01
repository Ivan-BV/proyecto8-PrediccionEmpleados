import streamlit as st
import pandas as pd
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

import sys
sys.path.append("/streamlit/")
import src.soporte_encoding_streamlit as ses

from IPython.display import display

# Configurar la página de Streamlit
st.set_page_config(
    page_title="Predicción de Retención de Empleados",
    page_icon="👥",
    layout="centered",
)

# Título y descripción
st.title("👥 Predicción de Retención de Empleados con Machine Learning")
st.write("Utiliza esta aplicación para predecir si un empleado permanecerá o dejará la empresa, basándote en sus características laborales y personales. ¡Descubre qué factores son clave! 🚀")

# Mostrar una imagen relacionada con empleados y retención
st.image(
    "https://images.unsplash.com/photo-1521737604893-d14cc237f11d",
    caption="El talento de tus empleados es el activo más importante.",
    use_container_width=True,
)

# Cargar los modelos y transformadores entrenados
def load_models():
    with open('../modelos/modelo5/transformers/transformer_one.pkl', 'rb') as f:
        oh = pickle.load(f)
    with open('../modelos/modelo5/transformers/transformer_target.pkl', 'rb') as f:
        target = pickle.load(f)
    with open('../modelos/modelo5/mejor_modelo/mejor_modelo.pkl', 'rb') as f:
        model = pickle.load(f)
    data_pickle = pd.read_pickle("../modelos/modelo5/datos/data_final.pkl")
    df = pd.DataFrame(data_pickle)

    return oh, target, model, df

oh, target, model, df = load_models()

# Formularios de entrada
education_dict = {
    'Below College': 1,
    'College': 2,
    'Bachelor': 3,
    'Master': 4,
    'Doctor': 5
}

education_field_dict = {
    'Life Sciences': 'Ciencias de la Vida',
    'Medical': 'Medicina',
    'Marketing': 'Marketing',
    'Technical Degree': 'Grado Técnico',
    'Human Resources': 'Recursos Humanos',
    'Other': 'Otro'
}

environment_satisfaction_dict = {
    'Low': 1,
    'Medium': 2,
    'High': 3,
    'Very High': 4
}

job_satisfaction_dict = {
    'Low': 1,
    'Medium': 2,
    'High': 3,
    'Very High': 4
}

work_life_balance_dict = {
    'Bad': 1,
    'Good': 2,
    'Better': 3,
    'Best': 4
}

job_involvement_dict = {
    'Low': 1,
    'Medium': 2,
    'High': 3,
    'Very High': 4
}

performance_rating_dict = {
    'Low': 1,
    'Good': 2,
    'Excellent': 3,
    'Outstanding': 4
}

job_role_dict = {
    'Representante de Salud': 'Healthcare Representative',
    'Científico de Investigación': 'Research Scientist',
    'Ejecutivo de Ventas': 'Sales Executive',
    'Recursos Humanos': 'Human Resources',
    'Director de Investigación': 'Research Director',
    'Técnico de Laboratorio': 'Laboratory Technician',
    'Director de Manufactura': 'Manufacturing Director',
    'Representante de Ventas': 'Sales Representative',
    'Gerente': 'Manager'
}

department_dict = {
    'Ventas': "Sales",
    'Investigación y Desarrollo': "Research & Development",
    'Recursos Humanos': "Human Resources"
}

business_travel_dict = {
    'Rara vez': "Travel_Rarely",
    'Frecuentemente': "Travel_Frequently",
    'Nunca': "Non-Travel"
}

marital_status_dict = {
    'Soltero': 'Single',
    'Casado': 'Married',
    'Divorciado': 'Divorced'
}

gender_dict = {
    'Masculino': "Male",
    'Femenino': "Female"
}



# Mejorar la detección de categorías ordenadas usando conocimiento de dominio
def detectar_orden_cat_con_conocimiento(df, lista_categoricas):
    # Definir manualmente las categorías que tienen un orden lógico
    categorias_ordenadas = [
        'Education', 'JobLevel', 'StockOptionLevel', 'EnvironmentSatisfaction',
        'JobSatisfaction', 'WorkLifeBalance', 'JobInvolvement', 'PerformanceRating'
    ]
    
    lista_ordenas = []
    lista_desordenadas = []
    
    for categorica in lista_categoricas:
        if categorica in categorias_ordenadas:
            lista_ordenas.append(categorica)
        else:
            lista_desordenadas.append(categorica)
    
    return lista_ordenas, lista_desordenadas

# Formularios de entrada
st.header("🔧 Características del Empleado")
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Edad", min_value=df["Age"].astype(int).min(), max_value=df["Age"].astype(int).max(), step=1)
    business_travel = st.selectbox("Frecuencia de Viajes de Trabajo", list(business_travel_dict.keys()), help="Frecuencia con la que el empleado viaja por trabajo.")
    department = st.selectbox("Departamento", list(department_dict.keys()), help="Selecciona el departamento del empleado.")
    distance_from_home = st.slider("Distancia desde Casa (km)", min_value=df["DistanceFromHome"].astype(int).min(), max_value=df["DistanceFromHome"].astype(int).max(), step=1)
    education = st.selectbox("Nivel Educativo", list(education_dict.keys()), help="Selecciona el nivel educativo del empleado.")
    education_field = st.selectbox("Campo de Educación", list(education_field_dict.keys()), help="Campo de estudio del empleado.")
    gender = st.selectbox("Género", list(gender_dict.keys()), help="Género del empleado.")
    job_level = st.slider("Nivel del Puesto (1-5)", min_value=1, max_value=5, value=2, help="Nivel del puesto actual del empleado.")
    job_role = st.selectbox("Rol del Trabajo", list(job_role_dict.keys()), help="Rol actual del empleado en la empresa.")
    years_with_curr_manager = st.slider("Años con el Actual Gerente", min_value=df["YearsWithCurrManager"].astype(int).min(), max_value=df["YearsWithCurrManager"].astype(int).max(), step=1)

with col2:
    marital_status = st.selectbox("Estado Civil", list(marital_status_dict.keys()), help="Estado civil del empleado.")
    monthly_income = st.number_input("Ingreso Mensual ($)", min_value=df["MonthlyIncome"].astype(int).min(), max_value=df["MonthlyIncome"].astype(int).max(), step=100)
    num_companies_worked = st.slider("Número de Empresas Anteriores", min_value=df["NumCompaniesWorked"].astype(int).min(), max_value=df["NumCompaniesWorked"].astype(int).max(), step=1)
    percent_salary_hike = st.slider("Incremento Salarial (%)", min_value=df["PercentSalaryHike"].astype(int).min(), max_value=df["PercentSalaryHike"].astype(int).max(), step=1)
    stock_option_level = st.slider("Nivel de Opciones sobre Acciones (0-3)", min_value=0, max_value=3, value=1)
    total_working_years = st.slider("Años Totales de Experiencia", min_value=df["TotalWorkingYears"].astype(int).min(), max_value=df["TotalWorkingYears"].astype(int).max(), step=1)
    training_times_last_year = st.slider("Capacitaciones en el Último Año", min_value=df["TrainingTimesLastYear"].astype(int).min(), max_value=df["TrainingTimesLastYear"].astype(int).max(), step=1)
    years_at_company = st.slider("Años en la Empresa", min_value=df["YearsAtCompany"].astype(int).min(), max_value=df["YearsAtCompany"].astype(int).max(), step=1)
    years_since_last_promotion = st.slider("Años desde la Última Promoción", min_value=df["YearsSinceLastPromotion"].astype(int).min(), max_value=df["YearsSinceLastPromotion"].astype(int).max(), step=1)
    

# Otras características
st.header("🔧 Características Adicionales del Entorno Laboral")
col3, col4 = st.columns(2)

with col3:
    environment_satisfaction = st.slider("Satisfacción con el Entorno (1-4)", min_value=1, max_value=4, value=3, help="Nivel de satisfacción con el entorno laboral del empleado.")
    job_satisfaction = st.slider("Satisfacción Laboral (1-4)", min_value=1, max_value=4, value=3, help="Nivel de satisfacción laboral del empleado.")
    work_life_balance = st.slider("Equilibrio Vida-Trabajo (1-4)", min_value=1, max_value=4, value=3, help="Nivel de equilibrio entre vida y trabajo del empleado.")

with col4:
    job_involvement = st.slider("Involucramiento en el Trabajo (1-4)", min_value=1, max_value=4, value=3, help="Grado de involucramiento del empleado en su trabajo.")
    performance_rating = st.slider("Evaluación de Desempeño (1-4)", min_value=1, max_value=4, value=3, help="Evaluación de desempeño del empleado.")
    overtime = st.selectbox("Horas Extra", ["Sí", "No"], help="Indica si el empleado realiza horas extra.")

# Botón para realizar la predicción
if st.button("💡 Predecir Retención"):
    # Crear DataFrame con los datos ingresados
    new_employee = pd.DataFrame({
        'Age': [age],
        'BusinessTravel': [business_travel_dict[business_travel]],
        'Department': [department_dict[department]],
        'DistanceFromHome': [distance_from_home],
        'Education': [education_dict[education]],
        'EducationField': [education_field],
        'Gender': [gender_dict[gender]],
        'JobLevel': [job_level],
        'JobRole': [job_role_dict[job_role]],
        'MaritalStatus': [marital_status_dict[marital_status]],
        'MonthlyIncome': [monthly_income],
        'NumCompaniesWorked': [num_companies_worked],
        'PercentSalaryHike': [percent_salary_hike],
        'StockOptionLevel': [stock_option_level],
        'TotalWorkingYears': [total_working_years],
        'TrainingTimesLastYear': [training_times_last_year],
        'YearsAtCompany': [years_at_company],
        'YearsSinceLastPromotion': [years_since_last_promotion],
        'YearsWithCurrManager': [years_with_curr_manager],
        'EnvironmentSatisfaction': [environment_satisfaction],
        'JobSatisfaction': [job_satisfaction],
        'WorkLifeBalance': [work_life_balance],
        'JobInvolvement': [job_involvement],
        'PerformanceRating': [performance_rating],
        'Overtime': [1 if overtime == 'Sí' else 0],
    })

    # Aplicar preprocesamiento
    #new_employee_encoded = pd.get_dummies(new_employee, drop_first=True)  # Codificar variables categóricas
    #new_employee_scaled = scaler.transform(new_employee_encoded)  # Escalar los valores

    #lista_ordenadas, lista_desordenadas = ses.detectar_orden_cat_2(new_employee.select_dtypes(["object", "category"]).columns)
    #diccionario_encoding = {"target": lista_ordenadas,
    #                        "onehot": lista_desordenadas}
    
    lista_ordenadas = ['BusinessTravel', 'Department', 'EducationField', 'JobRole', 'MaritalStatus', 'EnvironmentSatisfaction', 'JobSatisfaction', 'WorkLifeBalance', 'JobInvolvement']
    lista_desordenadas = ['Education', 'Gender', 'JobLevel', 'StockOptionLevel', 'PerformanceRating']
    
    #df_new_employee = pd.DataFrame(new_employee)
    pd.set_option("display.max_columns", None)

    trans_target = target.transform(new_employee)

    df_target = pd.DataFrame(trans_target)

    #new_employee_filtered = new_employee[lista_desordenadas].apply(lambda x: pd.Series([val if val in oh.categories_[i] else 'Unknown' for i, val in enumerate(x)]))
    trans_one_hot = oh.transform(df_target[lista_desordenadas])

    df_oh = pd.DataFrame(trans_one_hot.toarray(), columns=oh.get_feature_names_out())
    
    display(df_oh.columns)

    # Realizar la predicción
    prediction = model.predict(df_oh)[0]

    # Mostrar el resultado
    if prediction == 1:
        st.error("🔴 El empleado tiene una alta probabilidad de dejar la empresa.")
    else:
        st.success("🟢 El empleado probablemente permanecerá en la empresa.")

# Pie de página
st.markdown(
    """
    ---
    **Proyecto creado para analizar la retención de empleados utilizando ciencia de datos.**  
    Desarrollado con ❤️ usando Streamlit.
    """,
    unsafe_allow_html=True,
)
