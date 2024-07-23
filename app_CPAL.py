import dash
import math
from dash import dcc, html
from dash.dependencies import Input, Output
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

# Valores aleatorios para las tarjetas
cantidad_estudiantes = len(df_data_cruda['ALUMNOID'].unique())
promedio_edad = math.ceil(df_data_cruda['AGE'].mean())
promedio_cursos_llevados = math.ceil(df_data_cruda['APPROVED_COURSES'].mean())
promedio_experiencia_laboral = math.ceil(df_modelo.loc[df_modelo['EXP_PROFESIONAL'] != 0, 'EXP_PROFESIONAL'].mean())

# Leer datos desde un archivo Excel
df = pd.read_excel('ruta_al_archivo.xlsx')  # Reemplaza 'ruta_al_archivo.xlsx' con la ruta correcta

# Crear algunas figuras de ejemplo
def scatter_plot():
    fig = px.scatter(df, x=df.columns[0], y=df.columns[1], title="Scatter Plot de Ejemplo")
    return fig

def generar_tarjeta(valor, descripcion):
    return html.Div([
        html.H3(f"{valor}"),
        html.P(descripcion)
    ], style={'width': '24%', 'display': 'inline-block', 'textAlign': 'center', 'backgroundColor': '#1e488f', 'color': 'white', 'padding': '10px', 'borderRadius': '10px'})

app.layout = html.Div([
    html.Img(src=app.get_asset_url('logo.png'), style={'height': '20%', 'width': '20%', 'float': 'left'}),
    html.H1("ANÁLISIS DE ABANDONO CPAL", style={'textAlign': 'center', 'marginTop': '50px'}),
    
    html.Div([
        html.Div([
            html.Div("Cantidad de Estudiantes", style={'font-size': '20px'}),
            html.Div(f"{cantidad_estudiantes:,} mil", style={'font-size': '24px', 'font-weight': 'bold'})
        ], style={'width': '25%', 'display': 'inline-block', 'text-align': 'center'}),
        
        html.Div([
            html.Div("Promedio de Edad", style={'font-size': '20px'}),
            html.Div(f"{promedio_edad}", style={'font-size': '24px', 'font-weight': 'bold'})
        ], style={'width': '25%', 'display': 'inline-block', 'text-align': 'center'}),
        
        html.Div([
            html.Div("Promedio de Cursos Llevados", style={'font-size': '20px'}),
            html.Div(f"{promedio_cursos_llevados}", style={'font-size': '24px', 'font-weight': 'bold'})
        ], style={'width': '25%', 'display': 'inline-block', 'text-align': 'center'}),
        
        html.Div([
            html.Div("Promedio de Años de Experiencia Laboral", style={'font-size': '20px'}),
            html.Div(f"{promedio_experiencia_laboral}", style={'font-size': '24px', 'font-weight': 'bold'})
        ], style={'width': '25%', 'display': 'inline-block', 'text-align': 'center'}),
    ], style={'display': 'flex', 'justify-content': 'space-around', 'margin': '20px 0'}),

    html.Div([


        html.Div([
            html.Div("Periodo", style={'margin-top': '20px'}),
            dcc.Dropdown(
                options=[{'label': i, 'value': i} for i in ['Todas'] + list(df_data_cruda['PERIODO'].unique())],
                value='Todas'
                ),
            
            html.Div("Especialidades", style={'margin-top': '20px'}),
            dcc.Dropdown(
                options=[{'label': i, 'value': i} for i in ['Todas'] + list(df_data_cruda['NOMBREESPECIALIDAD'].unique())],
                value='Todas'
                ),
            
            html.Div("Distritos", style={'margin-top': '20px'}),
            dcc.Dropdown(
                options=[{'label': i, 'value': i} for i in ['Todas'] + list(df_data_cruda['DISTRITO'].unique())],
                value='Todas'
                ),
            
            html.Div("Profesión", style={'margin-top': '20px'}),
            dcc.Dropdown(
                options=[{'label': i, 'value': i} for i in ['Todas'] + list(df_data_cruda['PROFESION'].unique())],
                value='Todas'
                ),
            
            html.Div("Grado Académico", style={'margin-top': '20px'}),
            dcc.Dropdown(
                options=[{'label': i, 'value': i} for i in ['Todas'] + list(df_data_cruda['GRADOACADEMICO'].unique())],
                value='Todas'
                ),
            
            html.Div("Universidad", style={'margin-top': '20px'}),
            dcc.Dropdown(
                options=[{'label': i, 'value': i} for i in ['Todas'] + list(df_data_cruda['UNIVERSIDAD'].unique())],
                value='Todas'
                ),
            
            html.Div("Edad", style={'margin-top': '20px'}),
            dcc.Dropdown(
                options=[{'label': i, 'value': i} for i in ['Todas'] + list(df_data_cruda['AGE'].unique())],
                value='Todas'
                ),
        ], style={'width': '20%', 'display': 'inline-block', 'vertical-align': 'top', 'padding': '20px'}),

        html.Div([
            html.Div([
                dcc.Tabs(id="tabs-mode", value='analitico', children=[
                    dcc.Tab(label='Modo Analítico', value='analitico'),
                    dcc.Tab(label='Modo Predictivo', value='predictivo'),
                ]),
            ]),
            html.Div(id='tabs-content')
        ], style={'width': '75%', 'display': 'inline-block'}),
    ]),
])

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
                dcc.Tab(label='Árbol decisión', value='arbol'),
                dcc.Tab(label='Curva ROC', value='roc'),
                dcc.Tab(label='Matriz Confusión', value='confusion'),
            ]),
            html.Div(id='tabs-predictivo-content')
        ])

@app.callback(Output('tabs-analitico-content', 'children'),
              [Input('tabs-analitico', 'value')])

def render_analitico_content(tab):
    if tab == 'alumnos':
        return dcc.Graph(id='alumnos-graph', figure=plots_CPAL.crear_grafico_dirigido(df_data_cruda))
    elif tab == 'matriculas':
        return dcc.Graph(id='matriculas-graph', figure=plots_CPAL.crear(df_data_cruda))
    elif tab == 'edades':
        return dcc.Graph(id='edades-graph', figure=plots_CPAL.crear_violin_plot(df_data_cruda))
    elif tab == 'financiamientos':
        return dcc.Graph(id='financiamientos-graph', figure=plots_CPAL.crear_grafico_cascada(df_data_cruda))

@app.callback(Output('tabs-predictivo-content', 'children'),
              [Input('tabs-predictivo', 'value')])

def render_predictivo_content(tab):
    if tab == 'arbol':
        return dcc.Graph(id='arbol-graph', figure=plots_CPAL.entrenar_y_graficar_arbol_decision(df_modelo))
    elif tab == 'roc':
        return dcc.Graph(id='roc-graph', figure=plots_CPAL.create_charts(df_data_cruda,df_modelo))
    elif tab == 'confusion':
        return dcc.Graph(id='confusion-graph', figure=plots_CPAL.create_scatter_plot_study_period(df_modelo))

if __name__ == '__main__':
    app.run_server(debug=False)
