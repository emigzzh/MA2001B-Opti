import pandas as pd
import math
from pulp import *

time_limit=10
gap=0.05

# -------------------------------
# CARGA DE DATOS
# -------------------------------
df = pd.read_csv("dataset.csv", encoding="utf-8-sig")  # utf-8-sig elimina BOM si existe
df.columns = df.columns.str.strip()                  # quita espacios en nombres de columna
df["Nodo"] = df["Nodo"].astype(str)
df = df.set_index("Nodo")

N = list(df.index)

# -------------------------------
# DISTANCIAS (Haversine, en km)
# -------------------------------
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0  # radio de la Tierra en km
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi       = math.radians(lat2 - lat1)
    dlambda    = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

dist = {
    (i, j): haversine(
        df.loc[i, "Latitud"],  df.loc[i, "Longitud"],
        df.loc[j, "Latitud"],  df.loc[j, "Longitud"]
    )
    for i in N for j in N
}

# Costo: distancia / velocidad_promedio * horas  (igual que antes: km / 10.3 * 24)
c = {(i, j): dist[(i, j)] / 10.3 * 24 for i in N for j in N}

# -------------------------------
# DEMANDA
# -------------------------------
d = df["Demanda"].astype(float).to_dict()

# -------------------------------
# PRIORIDAD (score de vulnerabilidad ponderado)
#
#   Peso por grupo:
#     Población no vulnerable          : 0.5
#     Población de 0 a 14 años (niños) : 2.5
#     Población femenina 15-49 (jefas) : 3.0
#     Población de 60 años y más       : 5.0
#     Población con discapacidad       : 5.5
# -------------------------------
W_NO_VULNERABLE  = 0.5
W_NINOS          = 2.5
W_JEFAS          = 3.0
W_ADULTOS_MAYORES = 5.0
W_DISCAPACIDAD   = 5.5


p = (
    df["Poblacion no vulnerable"]            * W_NO_VULNERABLE
    + df["Poblacion de 0 a 14 anos"]         * W_NINOS
    + df["Poblacion femenina de 15 a 49 anos"] * W_JEFAS
    + df["Poblacion de 60 anos y mas"]       * W_ADULTOS_MAYORES
    + df["Poblacion con discapacidad"]       * W_DISCAPACIDAD
).to_dict()

# Asegurar que todos los nodos tienen valores (por si hay NaN)
for i in N:
    if i not in p or math.isnan(p[i]): p[i] = 0.0
    if i not in d or math.isnan(d[i]): d[i] = 0.0

# Filtrar 0s
N_orig = N[:]
N = [i for i in N if i == '1' or d[i] > 0]
eliminados = len(N_orig) - len(N)
print(f"[INFO] Nodos eliminados por demanda=0: {eliminados}, quedan {len(N)}")

# ---------------------------------------------------------------
# NODO 1: punto de inicio de rutas Y nodo con demanda propia
# ---------------------------------------------------------------

# -------------------------------
# PARÁMETROS
# -------------------------------
B            = 3000   # presupuesto total de transporte
Q            = 40     # capacidad del camión (cajas por ruta)
alpha        = 0.8    # peso del score de vulnerabilidad en beneficio
beta         = 0.05   # peso de la distancia al depósito en beneficio
gamma_w      = 0.15   # peso de la demanda en beneficio
lambda_cost  = 1      # peso del costo en el objetivo

# Distancia de cada nodo al depósito '1'
delta = {i: dist[('1', i)] for i in N}
delta['1'] = 0

# Beneficio de visitar cada nodo
w = {i: alpha * p[i] + beta * delta[i] + gamma_w * d[i] for i in N}

# Beneficio fijo del nodo 1 (siempre atendido como depósito)
beneficio_nodo1 = w['1']

N_clientes    = [i for i in N if i != '1']
total_demanda = sum(d[i] for i in N_clientes)
max_viajes    = math.ceil(total_demanda / Q)

print(f"[INFO] Nodo 1: d={d['1']:.1f}, p={p['1']:.3f}, w={w['1']:.3f} (beneficio fijo)")
print(f"[INFO] Clientes a rutear: {len(N_clientes)}, demanda total={total_demanda:.1f}")
print(f"[INFO] Q={Q}, max_viajes posibles={max_viajes}, B={B}")

# -------------------------------
# MODELO
# -------------------------------
model = LpProblem("VRP_MierNoriega", LpMaximize)

arcos = [(i, j) for i in N for j in N if i != j]

x = LpVariable.dicts("x", arcos,      0, 1, LpBinary)
f = LpVariable.dicts("f", arcos,      0)
y = LpVariable.dicts("y", N_clientes, 0, 1, LpBinary)

# -------------------------------
# OBJETIVO
# -------------------------------
model += (
    beneficio_nodo1
    + lpSum(w[i] * y[i] for i in N_clientes)
    - lambda_cost * lpSum(c[(i, j)] * x[(i, j)] for i, j in arcos)
)

# -------------------------------
# RESTRICCIONES DE VISITA
# -------------------------------
for i in N_clientes:
    model += lpSum(x[(j, i)] for j in N if j != i) == y[i], f"entrada_{i}"
    model += lpSum(x[(i, j)] for j in N if j != i) == y[i], f"salida_{i}"

# -------------------------------
# BALANCE DEL DEPÓSITO
# -------------------------------
model += (
    lpSum(x[('1', j)] for j in N_clientes)
    ==
    lpSum(x[(i, '1')] for i in N_clientes)
), "balance_deposito"

model += (
    lpSum(x[('1', j)] for j in N_clientes) <= max_viajes
), "max_rutas"

for i in N_clientes:
    model += (
        lpSum(x[(i, j)] for j in N if j != i)
        ==
        lpSum(x[(j, i)] for j in N if j != i)
    ), f"balance_vehiculo_{i}"

# -------------------------------
# CAPACIDAD DE CARGA
# f[(i,j)] = cajas acumuladas entregadas HASTA llegar a j en esa ruta
# El flujo CRECE conforme se hacen entregas
# Al llegar al depósito f[(i,'1')] = total entregado en esa ruta <= Q
for i in N_clientes:
    model += (
        lpSum(f[(i, j)] for j in N if j != i)   # sale de i
        - lpSum(f[(j, i)] for j in N if j != i)  # entra a i
        == d[i] * y[i]                            # se suma d[i] al salir
    ), f"flujo_carga_{i}"

for i, j in arcos:
    model += f[(i, j)] <= Q * x[(i, j)], f"cap_{i}_{j}"

# Al salir del depósito el camión va con 0 cajas entregadas
for j in N_clientes:
    model += f[('1', j)] == 0, f"vacio_salida_{j}"

# Al regresar al depósito trae el total entregado <= Q  ← AQUÍ está la clave
for i in N_clientes:
    model += f[(i, '1')] <= Q * x[(i, '1')], f"cap_regreso_{i}"


# -------------------------------
# PRESUPUESTO
# -------------------------------
model += (
    lpSum(c[(i, j)] * x[(i, j)] for i, j in arcos) <= B
), "presupuesto"


# -----------------------------------------------
# WARM START 
# -----------------------------------------------
# Construye una solución inicial usando:
#   1) Nearest Neighbor (NN):
#      - Inicia cada ruta desde el depósito eligiendo un cliente atractivo
#      - Extiende la ruta agregando el nodo más cercano factible (respeta capacidad Q)
#   2) Mejora local con 2-opt:
#      - Reordena cada ruta para reducir el costo de recorrido
# 
# Luego asigna:
#   - x[(i,j)] como arcos activos
#   - y[i] como nodos visitados
#   - f[(i,j)] como flujo acumulado (carga entregada)
# 
# Esto sirve como "hint" para acelerar el solver (no es solución obligatoria)
import copy

def nearest_neighbor_vrp(N_clientes, c, d, w, Q, deposito='1'):
    restantes = set(N_clientes)
    rutas = []
    # selecciona el nodo inicial de la ruta con mejor beneficio neto
    # (w[i] - costo ida y vuelta al depósito)
    while restantes:
        # elige el mejor nodo de inicio (mayor w neto) que quede
        inicio = max(restantes, key=lambda i: w[i] - c[(deposito,i)] - c[(i,deposito)])
        ruta, carga, actual = [inicio], d[inicio], inicio
        restantes.remove(inicio)
        while True:
            # candidatos: caben en capacidad y no visitados
            candidatos = [n for n in restantes if carga + d[n] <= Q]
            if not candidatos:
                break
            # elige el más cercano al nodo actual
            siguiente = min(candidatos, key=lambda n: c[(actual, n)])
            ruta.append(siguiente)
            carga += d[siguiente]
            restantes.remove(siguiente)
            actual = siguiente
        rutas.append(ruta)
    return rutas

rutas_nn = nearest_neighbor_vrp(N_clientes, c, d, w, Q)

def two_opt(ruta, c, deposito='1'):
    seq = [deposito] + ruta + [deposito]
    mejor = seq[:]
    mejorado = True
    while mejorado:
        mejorado = False
        for i in range(1, len(mejor)-2):
            for j in range(i+1, len(mejor)-1):
                nueva = mejor[:i] + mejor[i:j+1][::-1] + mejor[j+1:]
                costo_actual = sum(c[(mejor[k], mejor[k+1])] for k in range(len(mejor)-1))
                costo_nueva  = sum(c[(nueva[k],  nueva[k+1])]  for k in range(len(nueva)-1))
                if costo_nueva < costo_actual:
                    mejor = nueva
                    mejorado = True
    return mejor[1:-1]

rutas_nn = [two_opt(r, c) for r in rutas_nn]


for v in x.values(): v.setInitialValue(0)
for v in y.values(): v.setInitialValue(0)
for v in f.values(): v.setInitialValue(0)

for ruta in rutas_nn:
    secuencia = ['1'] + ruta + ['1']
    carga_acum = 0
    for k in range(len(secuencia) - 1):
        i, j = secuencia[k], secuencia[k+1]
        x[(i,j)].setInitialValue(1)
        if i == '1':
            f[(i,j)].setInitialValue(0)    
        else:
            carga_acum += d[i]
            f[(i,j)].setInitialValue(carga_acum)
    for nodo in ruta:
        y[nodo].setInitialValue(1)
    carga_acum = 0  

print(f"[NN+2OPT] {len(rutas_nn)} rutas, {sum(len(r) for r in rutas_nn)}/{len(N_clientes)} nodos cubiertos")
for k, ruta in enumerate(rutas_nn):
    print(f"  Ruta {k+1}: {['1']+ruta+['1']}  |  Cajas: {sum(d[i] for i in ruta):.1f}/{Q}")

# -------------------------------
# SOLVER
# -------------------------------
print("\n[INFO] Resolviendo...")
model.solve(PULP_CBC_CMD(timeLimit=time_limit, gapRel=gap, msg=1, warmStart=True, keepFiles=True))

# -------------------------------
# RESULTADOS
# -------------------------------
def val(v):
    return value(v) if value(v) is not None else 0.0

print("\n" + "=" * 60)
print("RESULTADOS")
print("=" * 60)
print(f"Status:            {LpStatus[model.status]}")
print(f"Z (objetivo):      {val(model.objective):.4f}")

visitados    = [i for i in N_clientes if val(y[i]) > 0.5]
ben_clientes = sum(w[i] * val(y[i]) for i in N_clientes)
costo_total  = sum(c[(i, j)] * val(x[(i, j)]) for i, j in arcos)
n_rutas      = int(round(sum(val(x[('1', j)]) for j in N_clientes)))
dem_cubierta = sum(d[i] for i in visitados)

print(f"Beneficio nodo 1:  {beneficio_nodo1:.4f}  (atendido fijo)")
print(f"Beneficio clientes:{ben_clientes:.4f}")
print(f"Costo transporte:  {costo_total:.4f}  (presupuesto B={B})")
print(f"Clientes visitados:{len(visitados)} de {len(N_clientes)}")
print(f"Demanda cubierta:  {dem_cubierta:.1f} de {total_demanda:.1f} cajas")
print(f"Rutas abiertas:    {n_rutas}  (máximo permitido={max_viajes})")

# Reconstruir rutas
print("\nRUTAS DETALLADAS:")
arcos_on = {(i, j) for i, j in arcos if val(x[(i, j)]) > 0.5}
salidas  = [j for j in N_clientes if ('1', j) in arcos_on]

if not salidas:
    print("  (ninguna ruta generada)")
    print(f"\n  Diagnóstico — top 5 nodos más rentables:")
    ranking = sorted(N_clientes, key=lambda i: -w[i])[:5]
    for i in ranking:
        cv = c[('1', i)] + c[(i, '1')]
        print(f"    Nodo {i:>4}: w={w[i]:.2f}, costo_ida_vuelta={cv:.3f}, neto={w[i]-cv:.2f}")
else:
    for inicio in salidas:
        ruta = ['1', inicio]
        actual = inicio
        visitados_ruta = {'1', inicio}
        for _ in range(len(N)):
            sig = next(
                (j for j in N if (actual, j) in arcos_on and j not in visitados_ruta),
                None
            )
            if sig is None:
                break
            ruta.append(sig)
            visitados_ruta.add(sig)
            actual = sig
            if actual == '1':
                break
        if ruta[-1] != '1':
            ruta.append('1')

        dem_r  = sum(d[i] for i in ruta if i != '1')
        cost_r = sum(c[(ruta[k], ruta[k + 1])] for k in range(len(ruta) - 1))
        w_r    = sum(w[i] for i in ruta if i != '1')
        print(f"\n  Ruta: {'->'.join(ruta)}")
        print(f"    Cajas entregadas: {dem_r:.1f}/{Q}  |  Costo: {cost_r:.3f}  |  Beneficio: {w_r:.2f}")






# =================================================================
# Animacion
# =================================================================

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches

def animar_rutas(rutas, df_coords, deposito='1'):
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.set_facecolor('#0f0f1a')
    fig.patch.set_facecolor('#0f0f1a')

    lons = df_coords["Longitud"].astype(float)
    lats = df_coords["Latitud"].astype(float)

    # todos los nodos de fondo
    ax.scatter(lons, lats, c='#444466', s=20, zorder=2)
    dep_lon = df_coords.loc[deposito, "Longitud"]
    dep_lat = df_coords.loc[deposito, "Latitud"]
    ax.scatter([dep_lon], [dep_lat], c='white', s=120, zorder=5, marker='*')
    ax.set_title("VRP — Rutas óptimas", color='white', fontsize=13)
    ax.tick_params(colors='#888888')
    for spine in ax.spines.values():
        spine.set_edgecolor('#333355')

    colores = plt.cm.tab10.colors
    artistas = []

    def get_coord(nodo):
        return (
            float(df_coords.loc[nodo, "Longitud"]),
            float(df_coords.loc[nodo, "Latitud"])
        )

    # pre-calcular frames: cada frame = un arco de una ruta
    frames_data = []  # (ruta_idx, secuencia hasta ese arco)
    for r_idx, ruta in enumerate(rutas):
        seq = [deposito] + ruta + [deposito]
        for k in range(1, len(seq)):
            frames_data.append((r_idx, seq[:k+1]))

    def update(frame):
        for a in artistas:
            a.remove()
        artistas.clear()

        r_idx, seq_parcial = frames_data[frame]
        color = colores[r_idx % len(colores)]

        # dibujar todas las rutas anteriores completas
        for prev_r in range(r_idx):
            prev_seq = [deposito] + rutas[prev_r] + [deposito]
            prev_color = colores[prev_r % len(colores)]
            for k in range(len(prev_seq)-1):
                x0, y0 = get_coord(prev_seq[k])
                x1, y1 = get_coord(prev_seq[k+1])
                ln, = ax.plot([x0, x1], [y0, y1], color=prev_color, lw=1.5, alpha=0.6, zorder=3)
                artistas.append(ln)

        # dibujar ruta actual hasta el arco actual
        for k in range(len(seq_parcial)-1):
            x0, y0 = get_coord(seq_parcial[k])
            x1, y1 = get_coord(seq_parcial[k+1])
            ln, = ax.plot([x0, x1], [y0, y1], color=color, lw=2.2, alpha=0.95, zorder=4)
            artistas.append(ln)

        # nodos visitados en esta ruta hasta ahora
        for nodo in seq_parcial[1:]:
            if nodo != deposito:
                cx, cy = get_coord(nodo)
                sc = ax.scatter([cx], [cy], c=[color], s=45, zorder=5)
                artistas.append(sc)

        # label de ruta actual
        lbl = ax.text(
            0.01, 0.98, f"Ruta {r_idx+1} / {len(rutas)}",
            transform=ax.transAxes, color=color,
            fontsize=11, va='top', fontweight='bold'
        )
        artistas.append(lbl)

        return artistas

    ani = animation.FuncAnimation(
        fig, update,
        frames=len(frames_data),
        interval=120,       # ms por frame — sube si va muy rápido
        blit=False,
        repeat=False
    )

    plt.tight_layout()
    return ani

# ---------------------------------------------
# GIF 2 — Satisfacción de demanda por nodo

def animar_demanda(rutas, df_coords, d, deposito='1'):
    """
    Nodos con demanda pendiente: punto rojo + número de cajas en texto pequeño.
    Al ser visitado: texto desaparece, punto se vuelve verde.
    """
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.set_facecolor('#0f0f1a')
    fig.patch.set_facecolor('#0f0f1a')
    ax.tick_params(colors='#888888')
    for spine in ax.spines.values():
        spine.set_edgecolor('#333355')
 
    lons = df_coords["Longitud"].astype(float)
    lats = df_coords["Latitud"].astype(float)
 
    todos = [n for n in df_coords.index if n != deposito and n in d and d[n] > 0]
    total_demanda = sum(d[i] for i in todos)
 
    # Nodos sin demanda al fondo
    nodos_fondo = [n for n in df_coords.index if n not in todos and n != deposito]
    if nodos_fondo:
        fx = [float(df_coords.loc[n, "Longitud"]) for n in nodos_fondo]
        fy = [float(df_coords.loc[n, "Latitud"])  for n in nodos_fondo]
        ax.scatter(fx, fy, c='#2a2a44', s=12, zorder=1)
 
    dep_lon = float(df_coords.loc[deposito, "Longitud"])
    dep_lat = float(df_coords.loc[deposito, "Latitud"])
    ax.scatter([dep_lon], [dep_lat], c='white', s=160, zorder=6, marker='*')
 
    colores_ruta = plt.cm.tab10.colors
 
    # Offset en grados para el texto (pequeño, justo encima del punto)
    # Se ajusta automáticamente al rango del mapa
    lat_range = lats.max() - lats.min()
    TEXT_OFFSET = lat_range * 0.012   # ~1.2% del rango total
 
    def get_coord(nodo):
        return (
            float(df_coords.loc[nodo, "Longitud"]),
            float(df_coords.loc[nodo, "Latitud"])
        )
 
    # Construir secuencia de frames
    frames_events = []
    satisfechos   = set()
    satisfecha    = 0.0
 
    for r_idx, ruta in enumerate(rutas):
        seq = [deposito] + ruta + [deposito]
        for k in range(len(seq)):
            nodo = seq[k]
            if nodo != deposito and nodo not in satisfechos:
                satisfechos.add(nodo)
                satisfecha += d[nodo]
            frames_events.append({
                'ruta_idx'   : r_idx,
                'seq_hasta'  : seq[:k+1],
                'satisfechos': satisfechos.copy(),
                'satisfecha' : satisfecha,
            })
 
    artistas = []
 
    def update(frame):
        for a in artistas:
            a.remove()
        artistas.clear()
 
        ev        = frames_events[frame]
        sat_set   = ev['satisfechos']
        seq_p     = ev['seq_hasta']
        r_idx     = ev['ruta_idx']
        sat_total = ev['satisfecha']
 
        # Arcos rutas anteriores
        for prev_r in range(r_idx):
            prev_seq   = [deposito] + rutas[prev_r] + [deposito]
            prev_color = colores_ruta[prev_r % len(colores_ruta)]
            for k in range(len(prev_seq)-1):
                x0, y0 = get_coord(prev_seq[k])
                x1, y1 = get_coord(prev_seq[k+1])
                ln, = ax.plot([x0,x1],[y0,y1], color=prev_color, lw=1.4, alpha=0.5, zorder=3)
                artistas.append(ln)
 
        # Arcos ruta actual
        cur_color = colores_ruta[r_idx % len(colores_ruta)]
        for k in range(len(seq_p)-1):
            x0, y0 = get_coord(seq_p[k])
            x1, y1 = get_coord(seq_p[k+1])
            ln, = ax.plot([x0,x1],[y0,y1], color=cur_color, lw=2.0, alpha=0.9, zorder=4)
            artistas.append(ln)
 
        # Nodos clientes
        for nodo in todos:
            cx, cy = get_coord(nodo)
 
            if nodo in sat_set:
                # Atendido: verde, sin etiqueta
                sc = ax.scatter([cx], [cy], c='#00e676', s=40, zorder=5)
                artistas.append(sc)
            else:
                # Pendiente: rojo + texto con número de cajas
                sc = ax.scatter([cx], [cy], c='#ff3333', s=35, zorder=5)
                artistas.append(sc)
                txt = ax.text(
                    cx, cy + TEXT_OFFSET,
                    str(int(d[nodo])),
                    color='white', fontsize=5.5,
                    ha='center', va='bottom',
                    zorder=6, fontweight='bold'
                )
                artistas.append(txt)
 
        # Contador global
        pct = (sat_total / total_demanda * 100) if total_demanda > 0 else 0
        titulo = ax.text(
            0.5, 1.005,
            f"Demanda satisfecha: {int(sat_total)} / {int(total_demanda)} cajas  ({pct:.1f}%)"
            f"   — Ruta {r_idx+1}/{len(rutas)}",
            transform=ax.transAxes, color='white', fontsize=10,
            ha='center', va='bottom', fontweight='bold'
        )
        artistas.append(titulo)
 
        # Leyenda
        leg_items = [
            mpatches.Patch(color='#ff3333', label='Demanda pendiente'),
            mpatches.Patch(color='#00e676', label='Demanda satisfecha'),
        ]
        leg = ax.legend(handles=leg_items, loc='lower left',
                        facecolor='#1a1a2e', edgecolor='#444466',
                        labelcolor='white', fontsize=9)
        artistas.append(leg)
 
        return artistas
 
    ani = animation.FuncAnimation(
        fig, update,
        frames=len(frames_events),
        interval=200,
        blit=False,
        repeat=False
    )
    plt.tight_layout()
    return ani

    
# -------------------------------
# LLAMADA
# -------------------------------
if salidas:
    rutas_finales = []
    for inicio in salidas:
        ruta = [inicio]
        actual = inicio
        visitados_ruta = {'1', inicio}
        for _ in range(len(N)):
            sig = next(
                (j for j in N if (actual, j) in arcos_on and j not in visitados_ruta),
                None
            )
            if sig is None:
                break
            ruta.append(sig)
            visitados_ruta.add(sig)
            actual = sig
            if actual == '1':
                break
        if ruta and ruta[-1] == '1':
            ruta = ruta[:-1]
        rutas_finales.append(ruta)
 
    # GIF 1
    ani1 = animar_rutas(rutas_finales, df)
    print("[INFO] Guardando rutas_vrp.gif ...")
    ani1.save("rutas_vrp.gif", writer="pillow", fps=8, dpi=120)
    print("[INFO] rutas_vrp.gif guardado.")
    plt.close('all')
 
    # GIF 2
    ani2 = animar_demanda(rutas_finales, df, d)
    print("[INFO] Guardando demanda_vrp.gif ...")
    ani2.save("demanda_vrp.gif", writer="pillow", fps=5, dpi=120)
    print("[INFO] demanda_vrp.gif guardado.")
    plt.close('all')