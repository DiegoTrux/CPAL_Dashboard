import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import networkx as nx
from plotly.subplots import make_subplots
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import numpy as np

def generar_histograma(df):
    fig = px.histogram(df, x='x', title='Histograma de x')
    return fig

def generar_scatter_plot(df):
    fig = px.scatter(df, x='x', y='y', title='Scatter plot de x vs y')
    return fig


def crear_grafico_dirigido(df):
    # Eliminar duplicados
    df = df.drop_duplicates()

    # Crear un gráfico dirigido
    G = nx.DiGraph()

    # Agregar nodos (variables) al gráfico
    variables = ['ALUMNOID', 'CURSOS_MATRICULADOS', 'CURSOS_MAT_APROB', 'PROMEDIO_PERIODO', 'NOMBREESPECIALIDAD']

    # Añadir etiquetas con valores calculados a los nodos
    node_labels = {
        'ALUMNOID': f'Total: {len(df)}',
        'CURSOS_MATRICULADOS': f"Promedio: {df['CURSOS_MATRICULADOS'].mean():.0f}",
        'CURSOS_MAT_APROB': f"Promedio: {df['CURSOS_MAT_APROB'].mean():.0f}",
        'PROMEDIO_PERIODO': f"Promedio: {df['PROMEDIO_PERIODO'].mean():.2f}",
        'NOMBREESPECIALIDAD': f"Total: {len(df['NOMBREESPECIALIDAD'].unique())}",
    }

    G.add_nodes_from(variables)

    # Añadir aristas (conexiones) entre las variables
    edges = [('CURSOS_MATRICULADOS', 'ALUMNOID'), ('CURSOS_MAT_APROB', 'CURSOS_MATRICULADOS'),
             ('PROMEDIO_PERIODO', 'CURSOS_MAT_APROB'), ('NOMBREESPECIALIDAD', 'ALUMNOID')]

    G.add_edges_from(edges)

    # Definir posiciones fijas para cada nodo
    fixed_positions = {
        'ALUMNOID': (0, 1),
        'CURSOS_MATRICULADOS': (1, 1.5),
        'CURSOS_MAT_APROB': (2, 2),
        'PROMEDIO_PERIODO': (3, 1.4),
        'NOMBREESPECIALIDAD': (1, 0.4)
    }

    # Crear el gráfico con Plotly
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = fixed_positions[edge[0]]
        x1, y1 = fixed_positions[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=2, color='#888'),
        hoverinfo='none',
        mode='lines'
    )

    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = fixed_positions[node]
        node_x.append(x)
        node_y.append(y)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=[f"{node}<br>{node_labels[node]}" for node in G.nodes()],
        textposition=[
            'top center' if node == 'NOMBREESPECIALIDAD' else 'bottom center'
            for node in G.nodes()
        ],
        hoverinfo='text',
        marker=dict(
            showscale=False,
            color='#A0BAD7',
            size=20,
            line_width=2)
    )

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title='<br>Resumen Estadístico de Alumnos',
                        height=600,
                        width=1400,
                        titlefont_size=16,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        annotations=[dict(
                            showarrow=False,
                            xref="paper", yref="paper",
                            x=1, y=-1)],
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        plot_bgcolor='white'
                        )
                    )

    return fig

def crear(df):
    # Eliminar duplicados
    df = df.drop_duplicates()
    df = df.dropna()

    # Crear un gráfico dirigido
    G = nx.DiGraph()

    # Agregar nodos (variables) al gráfico
    variables = ['SEXO', 'NACIONALIDAD', 'PROVINCIA', 'DISTRITO',
                 'PROFESION', 'EXP_PROFESIONAL', 'FINAN_RECPROPIOS', 'FINAN_APOYOFAM',
                 'FINAN_BANCA', 'FINAN_CENTROTRABAJO', 'FINAN_OTROS', 'UNIVERSIDAD',
                 'GRADOACADEMICO']

    # Añadir etiquetas con valores calculados a los nodos
    node_labels = {
        'SEXO': f"Moda: {df['SEXO'].mode().values[0]}",
        'NACIONALIDAD': f"Nacionalidades: {len(df['NACIONALIDAD'].unique())}",
        'PROVINCIA': f"Provincias: {len(df['PROVINCIA'].unique())}",
        'DISTRITO': f"Distritos: {len(df['DISTRITO'].unique())}",
        'PROFESION': f"Moda: {df['PROFESION'].mode().values[0]}",
        'EXP_PROFESIONAL': f"Moda: {df['EXP_PROFESIONAL'].mode().values[0]}",
        'FINAN_RECPROPIOS': f"Total: {df['FINAN_RECPROPIOS'].sum():.0f}",
        'FINAN_APOYOFAM': f"Total: {df['FINAN_APOYOFAM'].sum():.0f}",
        'FINAN_BANCA': f"Total: {df['FINAN_BANCA'].sum():.0f}",
        'FINAN_CENTROTRABAJO': f"Total: {df['FINAN_CENTROTRABAJO'].sum():.0f}",
        'FINAN_OTROS': f"Total: {df['FINAN_OTROS'].sum():.0f}",
        'UNIVERSIDAD': f"Moda: {df['UNIVERSIDAD'].mode().values[0]}",
        'GRADOACADEMICO': f"Moda: {df['GRADOACADEMICO'].mode().values[0]}",
        'ALUMNOID' : ''
    }

    G.add_nodes_from(variables)

    # Añadir aristas (conexiones) entre las variables
    edges = [('SEXO', 'ALUMNOID'), ('NACIONALIDAD', 'ALUMNOID'),
             ('PROVINCIA', 'NACIONALIDAD'), ('DISTRITO', 'PROVINCIA'), ('PROFESION', 'ALUMNOID'),
             ('EXP_PROFESIONAL', 'PROFESION'), ('FINAN_RECPROPIOS', 'EXP_PROFESIONAL'),
             ('FINAN_APOYOFAM', 'EXP_PROFESIONAL'), ('FINAN_BANCA', 'EXP_PROFESIONAL'),
             ('FINAN_CENTROTRABAJO', 'EXP_PROFESIONAL'), ('FINAN_OTROS', 'EXP_PROFESIONAL'),
             ('UNIVERSIDAD', 'ALUMNOID'), ('GRADOACADEMICO', 'UNIVERSIDAD')]

    G.add_edges_from(edges)

    # Definir posiciones fijas para cada nodo
    fixed_positions = {
        'ALUMNOID': (0, -1.5),
        'CURSOS_MATRICULADOS': (1, 2),
        'CURSOS_MAT_APROB': (2, 2),
        'PROMEDIO_PERIODO': (3, 1),
        'ESPECIALIDADID': (7, 0), ###
        'NOMBREESPECIALIDAD': (5, 1),
        'SEXO': (-3, -1.5),
        'NACIONALIDAD': (0, -4.5), ###
        'PROVINCIA': (0, -8), ###
        'DISTRITO': (-3, -8),
        'PROFESION': (0, 1),
        'EXP_PROFESIONAL': (0, 3.5),
        'FINAN_RECPROPIOS': (5, 6),
        'FINAN_APOYOFAM': (4, 2.5),
        'FINAN_BANCA': (-5, 6), ###
        'FINAN_CENTROTRABAJO': (0, 6),
        'FINAN_OTROS': (-4, 2.5),
        'UNIVERSIDAD': (3, -1.5),
        'GRADOACADEMICO': (6, -1.5)
    }

    # Aplicar desplazamientos a las posiciones fijas
    desplazamiento_x = 0.10
    desplazamiento_y = -0.10
    adjusted_positions = {node: (x + desplazamiento_x, y + desplazamiento_y) for node, (x, y) in fixed_positions.items()}

    # Crear el gráfico con Plotly
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = fixed_positions[edge[0]]
        x1, y1 = fixed_positions[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=2, color='#888'),
        hoverinfo='none',
        mode='lines'
    )

    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = fixed_positions[node]
        node_x.append(x)
        node_y.append(y)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=[f"{node}<br>{node_labels[node]}" for node in G.nodes()],
        textposition = [
            'middle right' if node in ['PROVINCIA', 'NACIONALIDAD'] else
            'top center' if node == 'DISTRITO' else
            'bottom center'
            for node in G.nodes()
        ],
        hoverinfo='text',
        marker=dict(
            showscale=False,
            color='#A0BAD7',
            size=20,
            line_width=2)
    )

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title='<br>Resumen Estadístico de Matriculas',
                        height=600,
                        width=1400,
                        titlefont_size=16,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        annotations=[dict(
                            showarrow=False,
                            xref="paper", yref="paper",
                            x=1, y=-1)],
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        plot_bgcolor='white'
                    )
                )
    return fig

def crear_violin_plot(df):
    # Calcular estadísticas
    q1 = df['AGE'].quantile(0.25)
    q2 = df['AGE'].quantile(0.5)
    q3 = df['AGE'].quantile(0.75)
    iqr = q3 - q1
    lcl = q1 - 1.5 * iqr
    ucl = q3 + 1.5 * iqr

    # Crear el Violin Plot con Plotly
    fig = go.Figure()

    fig.add_trace(go.Violin(x=df['AGE'], line_color='blue', name='Edades', box_visible=True, meanline_visible=True))

    # Agregar líneas para los cuartiles y límites de control
    fig.add_shape(type='line', x0=q1, y0=-1.2, x1=q1, y1=1.2, line=dict(color='red', width=2, dash='dash'))
    fig.add_shape(type='line', x0=q2, y0=-1.2, x1=q2, y1=1.2, line=dict(color='green', width=2, dash='dash'))
    fig.add_shape(type='line', x0=q3, y0=-1.2, x1=q3, y1=1.2, line=dict(color='blue', width=2, dash='dash'))
    fig.add_shape(type='line', x0=lcl, y0=-1.2, x1=lcl, y1=1.2, line=dict(color='orange', width=2, dash='dash'))
    fig.add_shape(type='line', x0=ucl, y0=-1.2, x1=ucl, y1=1.2, line=dict(color='purple', width=2, dash='dash'))

    # Agregar anotaciones para los valores de q1, q2, q3, lcl y ucl
    annotations = [
        dict(
            x=q1+2.5,
            y=-1,
            xref="x",
            yref="y",
            text=f"q1={q1:.2f}",
            showarrow=False,
            arrowhead=7,
            ax=0,
            ay=-40
        ),
        dict(
            x=q2+2.5,
            y=-1,
            xref="x",
            yref="y",
            text=f"q2={q2:.2f}",
            showarrow=False,
            arrowhead=7,
            ax=0,
            ay=-40
        ),
        dict(
            x=q3+2.5,
            y=-1,
            xref="x",
            yref="y",
            text=f"q3={q3:.2f}",
            showarrow=False,
            arrowhead=7,
            ax=0,
            ay=-40
        ),
        dict(
            x=lcl+2.5,
            y=-1,
            xref="x",
            yref="y",
            text=f"LCL={lcl:.2f}",
            showarrow=False,
            arrowhead=7,
            ax=0,
            ay=-40
        ),
        dict(
            x=ucl+2.5,
            y=-1,
            xref="x",
            yref="y",
            text=f"UCL={ucl:.2f}",
            showarrow=False,
            arrowhead=7,
            ax=0,
            ay=-40
        )
    ]

    fig.update_layout(title='Distribucion de Edades',
                      #xaxis_title='Edades',
                      #yaxis_title='Densidad',
                      violinmode='overlay',
                      height=600,
                      width=1400,
                      legend=dict(
                          yanchor="top",
                          y=0.99,
                          xanchor="left",
                          x=0.01
                      ),
                      annotations=annotations)

    return fig



import plotly.graph_objects as go

def crear_grafico_cascada(df):
    # Seleccionar las columnas de interés del DataFrame
    df_financiero = df[['FINAN_RECPROPIOS', 'FINAN_APOYOFAM', 'FINAN_BANCA', 'FINAN_CENTROTRABAJO', 'FINAN_OTROS']]
    
    # Contar la frecuencia de cada combinación de variables financieras
    combinaciones_frecuencia = df_financiero.groupby(df_financiero.columns.tolist()).size().reset_index(name='Frecuencia')
    
    # Ordenar por la combinación de variables
    combinaciones_frecuencia = combinaciones_frecuencia.sort_values(df_financiero.columns.tolist())
    
    # Crear etiquetas para las combinaciones
    etiquetas = ['.'.join([str(int(row[var])) for var in df_financiero.columns]) for _, row in combinaciones_frecuencia.iterrows()]
    
    # Definir las descripciones para cada combinación
    descripciones = {
        '0.0.0.0.0': 'sin datos',
        '0.0.0.0.1': 'FINAN_OTROS',   # Cat1
        '0.0.0.1.0': 'FINAN_CENTROTRABAJO',   # Cat2
        '0.0.0.1.1': 'FINAN_CENTROTRABAJO + FINAN_OTROS',   # Cat3
        '0.0.1.0.0': 'FINAN_BANCA',   # Cat4
        '0.0.1.1.0': 'FINAN_BANCA + FINAN_CENTROTRABAJO',   # Cat5
        '0.1.0.0.0': 'FINAN_APOYOFAM',   # Cat6
        '0.1.0.1.0': 'FINAN_APOYOFAM + FINAN_CENTROTRABAJO',   # Cat7
        '1.0.0.0.0': 'FINAN_RECPROPIOS',   # Cat8
        '1.0.0.0.1': 'FINAN_RECPROPIOS + FINAN_OTROS',   # Cat9
        '1.0.0.1.0': 'FINAN_RECPROPIOS + FINAN_CENTROTRABAJO',   # Cat10
        '1.0.0.1.1': 'FINAN_RECPROPIOS + FINAN_CENTROTRABAJO + FINAN_OTROS',   # Cat11
        '1.0.1.0.0': 'FINAN_RECPROPIOS + FINAN_BANCA',   # Cat12
        '1.0.1.1.0': 'FINAN_RECPROPIOS + FINAN_BANCA + FINAN_CENTROTRABAJO',   # Cat13
        '1.1.0.0.0': 'FINAN_RECPROPIOS + FINAN_APOYOFAM',   # Cat14
        '1.1.0.1.0': 'FINAN_RECPROPIOS + FINAN_APOYOFAM + FINAN_CENTROTRABAJO',   # Cat15
        '1.1.1.0.0': 'FINAN_RECPROPIOS + FINAN_APOYOFAM + FINAN_BANCA',   # Cat16
        '1.1.1.1.0': 'FINAN_RECPROPIOS + FINAN_APOYOFAM + FINAN_BANCA + FINAN_CENTROTRABAJO'   # Cat17
    }
    
    # Crear un nuevo diccionario para las etiquetas numeradas
    etiquetas_numeradas = {key: f'Cat{i+1}' for i, key in enumerate(descripciones.keys())}
    
    # Calcular los valores para el gráfico de cascada
    y = combinaciones_frecuencia['Frecuencia'].tolist()
    x = [etiquetas_numeradas[etiqueta] for etiqueta in etiquetas]
    
    # Calcular las medidas y el texto para cada barra
    measure = ['relative'] * len(y)
    text = [str(val) for val in y]
    
    # Crear el gráfico de cascada
    fig = go.Figure(go.Waterfall(
        name = "Leyenda de categorías", 
        orientation = "v",
        measure = measure,
        x = x,
        textposition = "outside",
        text = text,
        y = y,
        increasing=dict(marker=dict(color='skyblue')),  # Establecer el color para los valores positivos
        decreasing=dict(marker=dict(color='skyblue')),  # Establecer el color para los valores negativos
        totals=dict(marker=dict(color='skyblue')),      # Establecer el color para los totales
        connector = {"line":{"color":"rgb(63, 63, 63)"}}
    ))
    
    # Agregar trazos invisibles para las descripciones
    for key, value in descripciones.items():
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=10, color='rgba(255, 255, 255, 0)'),
            showlegend=True,
            name=f'{etiquetas_numeradas[key]}: {value}'
        ))
    
    # Configurar diseño del gráfico
    fig.update_layout(
        title={
            'text': "Distribución de Variables Financieras<br>['FINAN_RECPROPIOS', 'FINAN_APOYOFAM', 'FINAN_BANCA', 'FINAN_CENTROTRABAJO', 'FINAN_OTROS']",
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_title='Combinación de Variables Financieras',
        yaxis_title='Frecuencia',
        xaxis_tickangle=-90,
        height=600,
        width=1400,
        showlegend=True,
        plot_bgcolor='white',
        xaxis=dict(gridcolor='lightgrey'),
        yaxis=dict(gridcolor='lightgrey', range=[0, len(df_financiero)+80])
    )
    
    return fig

def entrenar_y_graficar_arbol_decision(df):
    # Eliminar duplicados en el DataFrame
    df = df.drop_duplicates()
    df.drop(['LLAVE', 'FLG_DESERTION_PRED', 'ALUMNOID'], axis=1, inplace=True)

    # Primera iteración con todas las variables
    df1 = df.copy()

    fig = make_subplots(rows=2, cols=1, subplot_titles=("Top 5 importancia con todas las variables", "Top 5 importancia sin la variable más importante"))

    for i, df_iter in enumerate([df1, None], start=1):
        if i == 1:
            # Primera iteración con todas las variables
            X = df_iter.drop('FLG_DESERTION', axis=1)
        else:
            # Segunda iteración sin la variable más importante de la primera iteración
            df_iter = df1.drop(columns=[top1_feature])
            X = df_iter.drop('FLG_DESERTION', axis=1)

        y = df['FLG_DESERTION']

        # Dividir los datos en conjuntos de entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Crear y entrenar el modelo de Árbol de Decisión
        DTC = DecisionTreeClassifier(class_weight={0: 1, 1: 5}, max_depth=5, min_samples_split=10)
        DTC.fit(X_train, y_train)

        # Mostrar Importancia de las Características
        importances = DTC.feature_importances_
        indices = np.argsort(importances)[::-1][:5]  # Obtener los índices de las 5 características más importantes
        features = X_train.columns[indices]

        # Convertir importancias a porcentaje
        importances_percentage = (importances[indices] / importances.sum()) * 100

        if i == 1:
            # Guardar la característica más importante de la primera iteración
            top1_feature = features[0]

        # Ordenar características e importancias de mayor a menor
        sorted_indices = np.argsort(importances_percentage)
        sorted_features = [features[i] for i in sorted_indices]
        sorted_importances = [importances_percentage[i] for i in sorted_indices]

        # Agregar gráfico al subplot
        fig.add_trace(go.Bar(
            x=sorted_importances,
            y=sorted_features,
            orientation='h',
            text=[f'{imp:.2f}%' for imp in sorted_importances],  # Agregar etiquetas de texto con porcentaje
            textposition='auto'
        ), row=i, col=1)

    fig.update_layout(
        height=600,
        width=1400,
        #title='Comparación de Importancia de Características',
        showlegend=False
    )

    return fig

def create_charts(dataset, dataset2):
    # Merge de los DataFrames para distrito
    merged = dataset.merge(dataset2[['LLAVE', 'FLG_DESERTION']], how='left', on='LLAVE')
    counts_departamento = merged.groupby(['DISTRITO', 'FLG_DESERTION']).size().reset_index(name='COUNT')
    top_departments = counts_departamento.groupby('DISTRITO')['COUNT'].sum().nlargest(5).index
    counts_top5_departamento = counts_departamento[counts_departamento['DISTRITO'].isin(top_departments)]
    counts_top5_departamento = counts_top5_departamento.sort_values(by='COUNT')

    # Merge de los DataFrames para universidad
    counts_universidad = merged.groupby(['UNIVERSIDAD', 'FLG_DESERTION']).size().reset_index(name='COUNT')
    top_universities = counts_universidad.groupby('UNIVERSIDAD')['COUNT'].sum().nlargest(5).index
    counts_top5_universidad = counts_universidad[counts_universidad['UNIVERSIDAD'].isin(top_universities)]
    counts_top5_universidad = counts_top5_universidad.sort_values(by='COUNT')

    # Merge de los DataFrames para grado académico
    counts_gradoacademico = merged.groupby(['GRADOACADEMICO', 'FLG_DESERTION']).size().reset_index(name='COUNT')
    top_grados = counts_gradoacademico.groupby('GRADOACADEMICO')['COUNT'].sum().nlargest(5).index
    counts_top5_gradoacademico = counts_gradoacademico[counts_gradoacademico['GRADOACADEMICO'].isin(top_grados)]
    counts_top5_gradoacademico = counts_top5_gradoacademico.sort_values(by='COUNT')

    # Merge de los DataFrames para profesión
    counts_profesion = merged.groupby(['PROFESION', 'FLG_DESERTION']).size().reset_index(name='COUNT')
    top_profesiones = counts_profesion.groupby('PROFESION')['COUNT'].sum().nlargest(5).index
    counts_top5_profesion = counts_profesion[counts_profesion['PROFESION'].isin(top_profesiones)]
    counts_top5_profesion = counts_top5_profesion.sort_values(by='COUNT')
    counts_top5_profesion['PROFESION'] = counts_top5_profesion['PROFESION'].str.replace('TECNOLOGIA', 'TEC')

    # Preparar los subplots
    fig = make_subplots(rows=2, cols=2, subplot_titles=("Por Distrito", "Por Grado Académico", "Por Universidad", "Por Profesión"))

    # Gráfico por distrito
    for flag, color in [(0, 'rgba(0, 0, 255, 0.5)'), (1, 'rgba(255, 0, 0, 0.5)')]:
        flag_data = counts_top5_departamento[counts_top5_departamento['FLG_DESERTION'] == flag]
        fig.add_trace(go.Bar(
            y=flag_data['DISTRITO'],
            x=flag_data['COUNT'],
            orientation='h',
            name=f'FLG_DESERTION: {flag}',
            marker_color=color,
            showlegend=False,
        ), row=1, col=1)

    # Gráfico por grado académico
    for flag, color in [(0, 'rgba(0, 0, 255, 0.5)'), (1, 'rgba(255, 0, 0, 0.5)')]:
        flag_data = counts_top5_gradoacademico[counts_top5_gradoacademico['FLG_DESERTION'] == flag]
        fig.add_trace(go.Bar(
            y=flag_data['GRADOACADEMICO'],
            x=flag_data['COUNT'],
            orientation='h',
            name=f'FLG_DESERTION: {flag}',
            marker_color=color,
            showlegend=False,
        ), row=1, col=2)

    # Gráfico por universidad
    for flag, color in [(0, 'rgba(0, 0, 255, 0.5)'), (1, 'rgba(255, 0, 0, 0.5)')]:
        flag_data = counts_top5_universidad[counts_top5_universidad['FLG_DESERTION'] == flag]
        fig.add_trace(go.Bar(
            y=flag_data['UNIVERSIDAD'],
            x=flag_data['COUNT'],
            orientation='h',
            name=f'FLG_DESERTION: {flag}',
            marker_color=color,
        ), row=2, col=1)

    # Gráfico por profesión
    for flag, color in [(0, 'rgba(0, 0, 255, 0.5)'), (1, 'rgba(255, 0, 0, 0.5)')]:
        flag_data = counts_top5_profesion[counts_top5_profesion['FLG_DESERTION'] == flag]
        fig.add_trace(go.Bar(
            y=flag_data['PROFESION'],
            x=flag_data['COUNT'],
            orientation='h',
            name=f'FLG_DESERTION: {flag}',
            marker_color=color,
            showlegend=True,
        ), row=2, col=2)

    # Configurar diseño del gráfico
    fig.update_layout(
        title='Análisis por Distrito, Grado Académico, Universidad y Profesión (Top 5)',
        height=600,
        width=1400,
        showlegend=False,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
        barmode='stack'
    )

    return fig

def create_scatter_plot_study_period(dataset):
    # Filtrar los datos donde STUDY_PERIOD sea menor a 2 años (365 * 2 días) y AGE sea menor a 80
    df = dataset[(dataset['STUDY_PERIOD'] < 365 * 2) & (dataset['AGE'] < 80)]

    # Mapear los valores de FLG_DESERTION_PRED a colores
    color_map = {0: 'blue', 1: 'red'}

    # Definir tamaño mínimo para los puntos
    min_size = 10

    # Ajustar el tamaño de los puntos asegurando un mínimo
    df['adjusted_size'] = df['EXP_PROFESIONAL'].apply(lambda x: max(x, min_size))

    # Crear el gráfico de dispersión con plotly.graph_objects
    fig = go.Figure()

    # Añadir trazado para cada categoría en color_map
    for key, color in color_map.items():
        df_filtered = df[df['FLG_DESERTION'] == key]
        fig.add_trace(go.Scatter(
            x=df_filtered['AGE'],
            y=df_filtered['STUDY_PERIOD'],
            mode='markers',
            marker=dict(
                size=df_filtered['adjusted_size'],  # Tamaño de los puntos según EXP_PROFESIONAL
                color=color,  # Color según FLG_DESERTION_PRED
            ),
            text=df_filtered.apply(lambda row: f"NOMBREESPECIALIDAD: {row['NOMBREESPECIALIDAD']} <br>STUDY_PERIOD: {row['STUDY_PERIOD']} <br>AGE: {row['AGE']} <br>FLG_DESERTION: {row['FLG_DESERTION']} <br>EXP_PROFESIONAL: {row['EXP_PROFESIONAL']}", axis=1),  # Texto al pasar el mouse sobre los puntos
            hoverinfo='text',  # Información que aparece al pasar el mouse sobre los puntos
            name=f'FLG_DESERTION: {key}'  # Nombre para mostrar en la leyenda
        ))

    # Personalizar el diseño del gráfico
    fig.update_layout(
        title='Scatter Plot con Color de Puntos por FLG_DESERTION',
        height=600,
        width=1400,
        xaxis_title='Edad',
        yaxis_title='Periodo de Estudio',
        showlegend=True,  # Mostrar leyenda
        legend_title='Predicción de Deserción',  # Título de la leyenda
    )

    return fig
