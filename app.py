import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import pandas as pd

# Configuración de la página
st.set_page_config(
    page_title="Calculadora de Dinámica",
    page_icon="🚀",
    layout="wide"
)

def ecuaciones_movimiento(t, y, g, k, m):
    """
    Sistema de ecuaciones diferenciales para proyectil con resistencia del aire
    y = [x, y, vx, vy]
    """
    x, y_pos, vx, vy = y
    
    # Velocidad total
    v_total = np.sqrt(vx**2 + vy**2)
    
    # Fuerzas de resistencia del aire (proporcional a v²)
    if v_total > 0:
        fx_drag = -k * v_total * vx
        fy_drag = -k * v_total * vy
    else:
        fx_drag = 0
        fy_drag = 0
    
    # Ecuaciones diferenciales
    dx_dt = vx
    dy_dt = vy
    dvx_dt = fx_drag / m
    dvy_dt = -g + fy_drag / m
    
    return [dx_dt, dy_dt, dvx_dt, dvy_dt]

def resolver_proyectil(v0, angulo, m, k, g, t_max):
    """
    Resuelve el movimiento del proyectil con resistencia del aire
    """
    # Convertir ángulo a radianes
    angulo_rad = np.radians(angulo)
    
    # Velocidades iniciales
    vx0 = v0 * np.cos(angulo_rad)
    vy0 = v0 * np.sin(angulo_rad)
    
    # Condiciones iniciales [x0, y0, vx0, vy0]
    y0 = [0, 0, vx0, vy0]
    
    # Resolver EDO
    sol = solve_ivp(
        ecuaciones_movimiento,
        [0, t_max],
        y0,
        args=(g, k, m),
        dense_output=True,
        events=lambda t, y, g, k, m: y[1],  # Evento: y = 0
        rtol=1e-8
    )
    
    return sol

def calcular_sin_resistencia(v0, angulo, g, t_max):
    """
    Calcula trayectoria sin resistencia del aire (referencia)
    """
    angulo_rad = np.radians(angulo)
    vx0 = v0 * np.cos(angulo_rad)
    vy0 = v0 * np.sin(angulo_rad)
    
    # Tiempo de vuelo teórico
    t_vuelo = 2 * vy0 / g
    t_max_real = min(t_max, t_vuelo)
    
    t = np.linspace(0, t_max_real, 1000)
    x = vx0 * t
    y = vy0 * t - 0.5 * g * t**2
    
    # Filtrar valores negativos de y
    mask = y >= 0
    
    return t[mask], x[mask], y[mask]

def crear_grafico(sol, t_sin_res, x_sin_res, y_sin_res, parametros):
    """
    Crea el gráfico comparativo de trayectorias
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Evaluar solución con resistencia
    t_eval = np.linspace(0, sol.t[-1], 1000)
    y_eval = sol.sol(t_eval)
    x_res = y_eval[0]
    y_res = y_eval[1]
    
    # Gráfico 1: Trayectorias
    ax1.plot(x_sin_res, y_sin_res, 'b--', linewidth=2, label='Sin resistencia')
    ax1.plot(x_res, y_res, 'r-', linewidth=2, label='Con resistencia')
    ax1.set_xlabel('Distancia horizontal (m)')
    ax1.set_ylabel('Altura (m)')
    ax1.set_title('Comparación de Trayectorias')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_xlim(0, max(max(x_sin_res), max(x_res)) * 1.1)
    ax1.set_ylim(0, max(max(y_sin_res), max(y_res)) * 1.1)
    
    # Gráfico 2: Velocidades
    vx_res = y_eval[2]
    vy_res = y_eval[3]
    v_total_res = np.sqrt(vx_res**2 + vy_res**2)
    
    # Velocidad sin resistencia
    v0 = parametros['v0']
    angulo_rad = np.radians(parametros['angulo'])
    vx0 = v0 * np.cos(angulo_rad)
    vy0 = v0 * np.sin(angulo_rad)
    vy_sin_res = vy0 - parametros['g'] * t_sin_res
    v_total_sin_res = np.sqrt(vx0**2 + vy_sin_res**2)
    
    ax2.plot(t_sin_res, v_total_sin_res, 'b--', linewidth=2, label='Sin resistencia')
    ax2.plot(t_eval, v_total_res, 'r-', linewidth=2, label='Con resistencia')
    ax2.set_xlabel('Tiempo (s)')
    ax2.set_ylabel('Velocidad total (m/s)')
    ax2.set_title('Velocidad vs Tiempo')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    return fig

# Título principal
st.title("🚀 Calculadora de Dinámica - Proyectil con Resistencia del Aire")
st.markdown("---")

# Sidebar para parámetros
st.sidebar.header("Parámetros del Proyectil")

# Parámetros iniciales
v0 = st.sidebar.slider("Velocidad inicial (m/s)", 10, 200, 50)
angulo = st.sidebar.slider("Ángulo de lanzamiento (°)", 5, 85, 45)
m = st.sidebar.slider("Masa del proyectil (kg)", 0.1, 10.0, 1.0, 0.1)

st.sidebar.subheader("Condiciones Ambientales")
g = st.sidebar.slider("Aceleración gravitacional (m/s²)", 9.0, 10.0, 9.81, 0.01)
k = st.sidebar.slider("Coeficiente de resistencia del aire", 0.0, 1.0, 0.1, 0.01)

st.sidebar.subheader("Configuración de Simulación")
t_max = st.sidebar.slider("Tiempo máximo de simulación (s)", 1, 50, 20)
mostrar_grafico = st.sidebar.checkbox("Mostrar gráfico", value=True)
mostrar_tabla = st.sidebar.checkbox("Mostrar tabla de datos", value=False)

# Almacenar parámetros
parametros = {
    'v0': v0,
    'angulo': angulo,
    'm': m,
    'g': g,
    'k': k,
    't_max': t_max
}

# Calcular soluciones
try:
    # Con resistencia del aire
    sol_resistencia = resolver_proyectil(v0, angulo, m, k, g, t_max)
    
    # Sin resistencia del aire
    t_sin_res, x_sin_res, y_sin_res = calcular_sin_resistencia(v0, angulo, g, t_max)
    
    # Resultados
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("📊 Resultados del Análisis")
        
        # Alcance y altura máxima con resistencia
        t_eval = np.linspace(0, sol_resistencia.t[-1], 1000)
        y_eval = sol_resistencia.sol(t_eval)
        x_res = y_eval[0]
        y_res = y_eval[1]
        
        alcance_res = x_res[-1]
        altura_max_res = np.max(y_res)
        tiempo_vuelo_res = sol_resistencia.t[-1]
        
        # Resultados sin resistencia
        alcance_sin_res = x_sin_res[-1]
        altura_max_sin_res = np.max(y_sin_res)
        tiempo_vuelo_sin_res = t_sin_res[-1]
        
        st.subheader("Con Resistencia del Aire")
        st.write(f"🎯 **Alcance:** {alcance_res:.2f} m")
        st.write(f"📈 **Altura máxima:** {altura_max_res:.2f} m")
        st.write(f"⏱️ **Tiempo de vuelo:** {tiempo_vuelo_res:.2f} s")
        
        st.subheader("Sin Resistencia del Aire")
        st.write(f"🎯 **Alcance:** {alcance_sin_res:.2f} m")
        st.write(f"📈 **Altura máxima:** {altura_max_sin_res:.2f} m")
        st.write(f"⏱️ **Tiempo de vuelo:** {tiempo_vuelo_sin_res:.2f} s")
    
    with col2:
        st.header("📈 Análisis Comparativo")
        
        # Diferencias porcentuales
        diff_alcance = ((alcance_sin_res - alcance_res) / alcance_sin_res) * 100
        diff_altura = ((altura_max_sin_res - altura_max_res) / altura_max_sin_res) * 100
        diff_tiempo = ((tiempo_vuelo_sin_res - tiempo_vuelo_res) / tiempo_vuelo_sin_res) * 100
        
        st.subheader("Efecto de la Resistencia del Aire")
        st.write(f"🎯 **Reducción en alcance:** {diff_alcance:.1f}%")
        st.write(f"📈 **Reducción en altura máxima:** {diff_altura:.1f}%")
        st.write(f"⏱️ **Reducción en tiempo de vuelo:** {diff_tiempo:.1f}%")
        
        # Energía cinética inicial
        energia_inicial = 0.5 * m * v0**2
        st.write(f"⚡ **Energía cinética inicial:** {energia_inicial:.2f} J")
        
        # Momento inicial
        momento_inicial = m * v0
        st.write(f"🏃 **Momento inicial:** {momento_inicial:.2f} kg⋅m/s")
    
    # Mostrar gráfico
    if mostrar_grafico:
        st.markdown("---")
        st.header("📊 Visualización")
        
        fig = crear_grafico(sol_resistencia, t_sin_res, x_sin_res, y_sin_res, parametros)
        st.pyplot(fig)
    
    # Mostrar tabla de datos
    if mostrar_tabla:
        st.markdown("---")
        st.header("📋 Tabla de Datos")
        
        # Crear DataFrame
        n_points = min(100, len(t_eval))
        indices = np.linspace(0, len(t_eval)-1, n_points, dtype=int)
        
        df = pd.DataFrame({
            'Tiempo (s)': t_eval[indices],
            'X con resistencia (m)': x_res[indices],
            'Y con resistencia (m)': y_res[indices],
            'Vx con resistencia (m/s)': y_eval[2][indices],
            'Vy con resistencia (m/s)': y_eval[3][indices]
        })
        
        st.dataframe(df, use_container_width=True)

except Exception as e:
    st.error(f"Error en el cálculo: {str(e)}")
    st.info("Intenta ajustar los parámetros o reducir el tiempo de simulación.")

# Información teórica
st.markdown("---")
st.header("📚 Información Teórica")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Ecuaciones del Movimiento")
    st.latex(r"""
    \begin{align}
    \frac{dx}{dt} &= v_x \\
    \frac{dy}{dt} &= v_y \\
    \frac{dv_x}{dt} &= -\frac{k \cdot v_{total} \cdot v_x}{m} \\
    \frac{dv_y}{dt} &= -g - \frac{k \cdot v_{total} \cdot v_y}{m}
    \end{align}
    """)

with col2:
    st.subheader("Parámetros del Modelo")
    st.write("""
    - **k**: Coeficiente de resistencia del aire
    - **m**: Masa del proyectil
    - **g**: Aceleración gravitacional
    - **v_total**: Velocidad total = √(vₓ² + vᵧ²)
    - La fuerza de resistencia es proporcional a v²
    """)

st.info("""
**Nota:** Este modelo considera la resistencia del aire proporcional al cuadrado de la velocidad, 
que es una aproximación válida para altas velocidades. La resistencia actúa en dirección opuesta 
al movimiento, reduciendo tanto el alcance como la altura máxima del proyectil.
""")