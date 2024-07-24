import dash
import math
from dash import dcc, html
from dash.dependencies import Input, Output, State
import pandas as pd
import plotly.express as px
import plots_CPAL as plots_CPAL

# Leer datos y eliminar duplicados
df_data_cruda = pd.read_excel("https://github.com/DiegoTrux/CPAL_Dashboard/blob/main/df_data_cruda_prueba.xlsx?raw=true")
df_data_cruda = df_data_cruda.drop_duplicates()
df_data_cruda.loc[:, df_data_cruda.select_dtypes(include=['object']).columns] = df_data_cruda.select_dtypes(include=['object']).fillna('VALOR VACIO')

# Leer datos y eliminar duplicados
df_modelo = pd.read_excel("https://github.com/DiegoTrux/CPAL_Dashboard/blob/main/df_modelo_prueba.xlsx?raw=true")
df_modelo = df_modelo.drop_duplicates()
df_modelo.loc[:, df_modelo.select_dtypes(include=['object']).columns] = df_modelo.select_dtypes(include=['object']).fillna('VALOR VACIO')

# Crear la aplicación Dash
app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server

# Función para generar tarjetas
def generar_tarjeta(valor, descripcion):
    return html.Div([
        html.Div(descripcion, style={'font-size': '20px'}),
        html.Div(f"{valor}", style={'font-size': '24px', 'font-weight': 'bold'})
    ], style={'width': '25%', 'display': 'inline-block', 'text-align': 'center', 'backgroundColor': 'white', 'color': 'black', 'padding': '10px', 'borderRadius': '10px'})

app.layout = html.Div([
    html.Img(src=app.get_asset_url('logo.png'), style={'height': '20%', 'width': '20%', 'float': 'left'}),
    html.H1("ANÁLISIS DE ABANDONO CPAL", style={'textAlign': 'center', 'marginTop': '50px'}),
    
    html.Div(id='tarjetas', style={'display': 'flex', 'justify-content': 'space-around', 'margin': '20px 0'}),

    html.Div([
        html.Div([
            html.Div("Periodo", style={'margin-top': '20px'}),
            dcc.Dropdown(
                id='Periodo',
                options=[{'label': i, 'value': i} for i in ['Todas'] + list(df_data_cruda['PERIODO'].unique())],
                value='Todas'
                ),
            
            html.Div("Especialidades", style={'margin-top': '20px'}),
            dcc.Dropdown(
                id='Especialidades',
                options=[{'label': i, 'value': i} for i in ['Todas'] + list(df_data_cruda['NOMBREESPECIALIDAD'].unique())],
                value='Todas'
                ),
            
            html.Div("Distritos", style={'margin-top': '20px'}),
            dcc.Dropdown(
                id='Distritos',
                options=[{'label': i, 'value': i} for i in ['Todas'] + list(df_data_cruda['DISTRITO'].unique())],
                value='Todas'
                ),
            
            html.Div("Profesión", style={'margin-top': '20px'}),
            dcc.Dropdown(
                id='Profesión',
                options=[{'label': i, 'value': i} for i in ['Todas'] + list(df_data_cruda['PROFESION'].unique())],
                value='Todas'
                ),
            
            html.Div("Grado Académico", style={'margin-top': '20px'}),
            dcc.Dropdown(
                id='Grado Académico',
                options=[{'label': i, 'value': i} for i in ['Todas'] + list(df_data_cruda['GRADOACADEMICO'].unique())],
                value='Todas'
                ),
            
            html.Div("Universidad", style={'margin-top': '20px'}),
            dcc.Dropdown(
                id='Universidad',
                options=[{'label': i, 'value': i} for i in ['Todas'] + list(df_data_cruda['UNIVERSIDAD'].unique())],
                value='Todas'
                ),
            
            html.Div("Edad", style={'margin-top': '20px'}),
            dcc.Dropdown(
                id='Edad',
                options=[{'label': i, 'value': i} for i in ['Todas'] + list(df_data_cruda['AGE'].unique())],
                value='Todas'
                ),
        ], style={'width': '20%', 'display': 'inline-block', 'vertical-align': 'top', 'padding': '20px'}),

        html.Div([
            html.Div([
                dcc.Tabs(id="tabs-mode", value='analitico', children=[
                    dcc.Tab(label='Analisis Descriptivo', value='analitico'),
                    dcc.Tab(label='Analisis Predictivo', value='predictivo'),
                ]),
            ]),
            html.Div(id='tabs-content')
        ], style={'width': '75%', 'display': 'inline-block'}),
    ]),
])

def filter_dataframe(df, periodo, especialidades, distritos, profesion, grado_academico, universidad, edad):
    if periodo and periodo != 'Todas':
        df = df[df['PERIODO'] == periodo]
    if especialidades and especialidades != 'Todas':
        df = df[df['NOMBREESPECIALIDAD'] == especialidades]
    if distritos and distritos != 'Todas':
        df = df[df['DISTRITO'] == distritos]
    if profesion and profesion != 'Todas':
        df = df[df['PROFESION'] == profesion]
    if grado_academico and grado_academico != 'Todas':
        df = df[df['GRADOACADEMICO'] == grado_academico]
    if universidad and universidad != 'Todas':
        df = df[df['UNIVERSIDAD'] == universidad]
    if edad and edad != 'Todas':
        df = df[df['AGE'] == int(edad)]
    return df

@app.callback(
    Output('tarjetas', 'children'),
    [Input('Periodo', 'value'),
     Input('Especialidades', 'value'),
     Input('Distritos', 'value'),
     Input('Profesión', 'value'),
     Input('Grado Académico', 'value'),
     Input('Universidad', 'value'),
     Input('Edad', 'value')]
)
def update_tarjetas(periodo, especialidades, distritos, profesion, grado_academico, universidad, edad):
    df_filtered = filter_dataframe(df_data_cruda, periodo, especialidades, distritos, profesion, grado_academico, universidad, edad)
    df_filtered_modelo = df_modelo[df_modelo['LLAVE'].isin(list(df_filtered['LLAVE'].unique()))]
    
    cantidad_estudiantes = len(df_filtered['ALUMNOID'].unique())
    promedio_edad = math.ceil(df_filtered['AGE'].mean())
    promedio_cursos_llevados = math.ceil(df_filtered['APPROVED_COURSES'].mean())
    promedio_experiencia_laboral = math.ceil(df_filtered_modelo.loc[df_filtered_modelo['EXP_PROFESIONAL'] != 0, 'EXP_PROFESIONAL'].mean())

    return [
        generar_tarjeta(f"{cantidad_estudiantes:,}", "Cantidad de Estudiantes"),
        generar_tarjeta(promedio_edad, "Promedio de Edad"),
        generar_tarjeta(promedio_cursos_llevados, "Promedio de Cursos Llevados"),
        generar_tarjeta(promedio_experiencia_laboral, "Promedio de Años de Experiencia Laboral")
    ]

@app.callback(Output('tabs-content', 'children'),
              [Input('tabs-mode', 'value')])

def render_content(tab):
    if tab == 'analitico':
        return html.Div([
            dcc.Tabs(id="tabs-analitico", value='alumnos', children=[
                dcc.Tab(label='Alumnos', value='alumnos'),
                dcc.Tab(label='Matrículas', value='matriculas'),
                dcc.Tab(label='Edades', value='edades'),
                dcc.Tab(label='Financiamientos', value='financiamientos'),
            ]),
            html.Div(id='tabs-analitico-content')
        ])
    elif tab == 'predictivo':
        return html.Div([
            dcc.Tabs(id="tabs-predictivo", value='arbol', children=[
                dcc.Tab(label='Importancia de variables', value='arbol'),
                dcc.Tab(label='Variables Cualitativas', value='roc'),
                dcc.Tab(label='Variables Cuantitativas', value='confusion'),
            ]),
            html.Div(id='tabs-predictivo-content')
        ])

@app.callback(
    Output('tabs-analitico-content', 'children'),
    [Input('tabs-analitico', 'value')],
    [State('Periodo', 'value'),
     State('Especialidades', 'value'),
     State('Distritos', 'value'),
     State('Profesión', 'value'),
     State('Grado Académico', 'value'),
     State('Universidad', 'value'),
     State('Edad', 'value')]
)
def render_analitico_content(tab, periodo, especialidades, distritos, profesion, grado_academico, universidad, edad):
    df_filtered = filter_dataframe(df_data_cruda, periodo, especialidades, distritos, profesion, grado_academico, universidad, edad)
    if tab == 'alumnos':
        return dcc.Graph(id='alumnos-graph', figure=plots_CPAL.crear_grafico_dirigido(df_filtered))
    elif tab == 'matriculas':
        return dcc.Graph(id='matriculas-graph', figure=plots_CPAL.crear(df_filtered))
    elif tab == 'edades':
        return dcc.Graph(id='edades-graph', figure=plots_CPAL.crear_violin_plot(df_filtered))
    elif tab == 'financiamientos':
        return dcc.Graph(id='financiamientos-graph', figure=plots_CPAL.crear_grafico_cascada(df_filtered))

@app.callback(
    Output('tabs-predictivo-content', 'children'),
    [Input('tabs-predictivo', 'value')],
    [State('Periodo', 'value'),
     State('Especialidades', 'value'),
     State('Distritos', 'value'),
     State('Profesión', 'value'),
     State('Grado Académico', 'value'),
     State('Universidad', 'value'),
     State('Edad', 'value')]
)
def render_predictivo_content(tab, periodo, especialidades, distritos, profesion, grado_academico, universidad, edad):
    df_filtered = filter_dataframe(df_data_cruda, periodo, especialidades, distritos, profesion, grado_academico, universidad, edad)
    df_filtered_modelo = df_modelo[df_modelo['LLAVE'].isin(list(df_filtered['LLAVE'].unique()))]
    if tab == 'arbol':
        return dcc.Graph(id='arbol-graph', figure=plots_CPAL.entrenar_y_graficar_arbol_decision(df_filtered_modelo))
    elif tab == 'roc':
        return dcc.Graph(id='roc-graph', figure=plots_CPAL.create_charts(df_filtered, df_filtered_modelo))
    elif tab == 'confusion':
        return dcc.Graph(id='confusion-graph', figure=plots_CPAL.create_scatter_plot_study_period(df_filtered_modelo))

if __name__ == '__main__':
    app.run_server(debug=False)
