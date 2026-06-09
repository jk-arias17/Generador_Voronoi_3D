# Generador de mezcla de concreto con Voronoi 3D

import matplotlib
matplotlib.use('TkAgg')      # sin esto la ventana 3D no se puede rotar interactivamente en Windows
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import Voronoi, ConvexHull
import math
import os
import random
import time
import functools

# Forzar que cada print se muestre al instante (en IDEs como Spyder el texto
# queda en buffer y no se ve el avance hasta el final).
print = functools.partial(print, flush=True)


CARPETA = os.path.dirname(os.path.abspath(__file__))


# ==============================================================
# Parámetros a modificar
# ==============================================================

# Dimensiones del dominio [mm]
B = 150
H = 150
D = 200

# Fracción volumétrica de agregados sobre el total del dominio
frac_agregado = 0.7

# Tipos de agregado: (nombre, d_min [mm], d_max [mm], fracción del total de agregado)
# Las fracciones deben sumar 1.0. Para agregar otro tipo añadir una fila más.
tipos_agregado = [
    ('3/4"', 13.0, 19.0, 0.10),
    ('1/2"',  9.5, 12.5, 0.35),
    ('3/8"',  4.75, 9.5, 0.40),
    ('#4',   2.36, 4.75, 0.15),
]

# Semilla aleatoria: None = diferente cada ejecución, entero = reproducible
semilla = None

# Elongación: factor > 1, fracción de partículas afectadas (0.0 a 1.0)
elongacion     = 3
porc_elongadas = 0

# Aplanamiento: factor < 1, fracción de partículas afectadas (0.0 a 1.0)
aplanamiento   = 0.8
porc_aplanadas = 0

# Tolerancia de volumen en asignación y número máximo de semillas Voronoi
tol          = 0.02
max_semillas = 10000

# Celdas Voronoi objetivo por partícula de agregado.
# Más alto, menor tamaño de celda, más celdas por partícula, mayor fracción real.
N_cel_por_part = 30

usar_paralelo = False
n_procesos    = 0

# Colores de visualización: índice 0 = matriz, 1..N = tipos de agregado
_COLORES_BASE = [
    (0.62, 0.62, 0.62),
    (1.00, 0.28, 0.28),
    (0.20, 0.45, 1.00),
    (0.20, 0.78, 0.20),
    (1.00, 0.85, 0.10),
    (0.80, 0.20, 0.80),
    (0.10, 0.80, 0.80),
    (1.00, 0.50, 0.00),
    (0.50, 0.25, 0.00),
]

def obtener_color(k):
    if k < len(_COLORES_BASE):
        return _COLORES_BASE[k]
    import matplotlib.cm as cm
    return tuple(cm.tab20((k - len(_COLORES_BASE)) % 20)[:3])


def vol_esfera(d):
    return (math.pi / 6.0) * d**3


vol_dominio   = B * H * D
vol_agregado_total = frac_agregado * vol_dominio

fracciones    = np.array([t[3] for t in tipos_agregado])
vols_min      = np.array([vol_esfera(t[1]) for t in tipos_agregado])
vols_max      = np.array([vol_esfera(t[2]) for t in tipos_agregado])
vols_med      = 0.5 * (vols_min + vols_max)
vol_por_clase = fracciones * vol_agregado_total


def volumen_celda(verts):
    try:
        return float(ConvexHull(np.asarray(verts, dtype=float)).volume)
    except Exception:
        return 0.0


def centroide_celda(verts):
    pts = np.asarray(verts, dtype=float)
    try:
        h = ConvexHull(pts)
        return np.mean(pts[h.vertices], axis=0)
    except Exception:
        return np.mean(pts, axis=0)


def escalar_celda(verts, vol_objetivo, elong=1.0, aplana=1.0):
    # deforma la celda visualmente para los gráficos; no modifica los archivos de texto exportados
    pts = np.asarray(verts, dtype=float)
    v   = volumen_celda(pts)
    if v <= 0:
        return pts
    c    = centroide_celda(pts)
    desplaz = pts - c

    if elong > 1.0 + 1e-9:
        theta = random.uniform(0.0, 2.0 * math.pi)
        phi   = math.acos(random.uniform(-1.0, 1.0))
        u = np.array([math.sin(phi) * math.cos(theta),
                      math.sin(phi) * math.sin(theta),
                      math.cos(phi)])
        proy = desplaz @ u
        perp = desplaz - np.outer(proy, u)
        desplaz = np.outer(proy * elong, u) + perp / math.sqrt(elong)

    if abs(aplana - 1.0) > 1e-9:
        desplaz[:, 2]  *= aplana
        desplaz[:, :2] /= math.sqrt(aplana)

    v2 = volumen_celda(c + desplaz)
    if v2 <= 0:
        v2 = v
    factor = (vol_objetivo / v2) ** (1.0 / 3.0)
    pts_deform = c + factor * desplaz
    recortado = recortar_caja(pts_deform)
    return recortado if recortado is not None and len(recortado) >= 4 else pts_deform


def recortar_semiespacio(pts, normal, offset):
    d = pts @ normal - offset
    adentro = d >= -1e-9
    if not np.any(adentro):
        return None
    if np.all(adentro):
        return pts

    nuevos = list(pts[adentro])

    try:
        hull = ConvexHull(pts)
        vistas = set()
        for tri in hull.simplices:
            for ia in range(len(tri)):
                for ib in range(ia + 1, len(tri)):
                    a, b = int(tri[ia]), int(tri[ib])
                    k = (min(a, b), max(a, b))
                    if k in vistas:
                        continue
                    vistas.add(k)
                    da, db = float(d[a]), float(d[b])
                    if (da < -1e-9 and db > 1e-9) or (da > 1e-9 and db < -1e-9):
                        t = da / (da - db)
                        nuevos.append(pts[a] + t * (pts[b] - pts[a]))
    except Exception:
        pass

    if len(nuevos) < 4:
        return None

    arr = np.array(nuevos, dtype=float)
    try:
        h = ConvexHull(arr)
        return arr[h.vertices]
    except Exception:
        return None


def recortar_caja(verts):
    pts = np.array(verts, dtype=float)
    planos = [
        (np.array([ 1., 0., 0.]),  0.),
        (np.array([-1., 0., 0.]), -B),
        (np.array([ 0., 1., 0.]),  0.),
        (np.array([ 0.,-1., 0.]), -H),
        (np.array([ 0., 0., 1.]),  0.),
        (np.array([ 0., 0.,-1.]), -D),
    ]
    for n, off in planos:
        if pts is None or len(pts) < 4:
            return None
        pts = recortar_semiespacio(pts, n, off)
    return pts



def ordenar_poligono_planar(verts):
    verts = np.asarray(verts, dtype=float)
    if len(verts) < 3:
        return verts

    centroide = verts.mean(axis=0)

    n = None
    for k in range(2, len(verts)):
        cand = np.cross(verts[1] - verts[0], verts[k] - verts[0])
        if np.linalg.norm(cand) > 1e-12:
            n = cand / np.linalg.norm(cand)
            break
    if n is None:
        return verts

    if abs(n[0]) < 0.9:
        u = np.array([1., 0., 0.])
    else:
        u = np.array([0., 1., 0.])
    u = u - (u @ n) * n
    u = u / np.linalg.norm(u)
    v = np.cross(n, u)

    diferencias  = verts - centroide
    xs     = diferencias @ u
    ys     = diferencias @ v
    angulos = np.arctan2(ys, xs)
    return verts[np.argsort(angulos)]


def recortar_poligono_caja(poly):
    if len(poly) < 3:
        return []
    p = [np.asarray(x, dtype=float) for x in poly]

    planos = [
        (np.array([ 1., 0., 0.]),  0.),
        (np.array([-1., 0., 0.]), -B),
        (np.array([ 0., 1., 0.]),  0.),
        (np.array([ 0.,-1., 0.]), -H),
        (np.array([ 0., 0., 1.]),  0.),
        (np.array([ 0., 0.,-1.]), -D),
    ]

    for normal, off in planos:
        if len(p) < 3:
            return []
        out = []
        m = len(p)
        for k in range(m):
            a = p[k]
            b = p[(k + 1) % m]
            da = float(normal @ a) - off
            db = float(normal @ b) - off
            adentro_a = da >= -1e-9
            adentro_b = db >= -1e-9
            if adentro_a:
                out.append(a)
                if not adentro_b:
                    t = da / (da - db)
                    out.append(a + t * (b - a))
            else:
                if adentro_b:
                    t = da / (da - db)
                    out.append(a + t * (b - a))
        p = out

    return p


def ordenar_idx_planar(coords):
    coords = np.asarray(coords, dtype=float)
    if len(coords) < 3:
        return list(range(len(coords)))

    centroide = coords.mean(axis=0)
    n = None
    for k in range(2, len(coords)):
        cand = np.cross(coords[1] - coords[0], coords[k] - coords[0])
        if np.linalg.norm(cand) > 1e-12:
            n = cand / np.linalg.norm(cand)
            break
    if n is None:
        return list(range(len(coords)))

    if abs(n[0]) < 0.9:
        u = np.array([1., 0., 0.])
    else:
        u = np.array([0., 1., 0.])
    u = u - (u @ n) * n
    u = u / np.linalg.norm(u)
    v = np.cross(n, u)

    diferencias  = coords - centroide
    xs     = diferencias @ u
    ys     = diferencias @ v
    angulos = np.arctan2(ys, xs)
    return list(np.argsort(angulos))


def es_borde_caja(poly, eps=1e-6):
    if len(poly) == 0:
        return False
    poly_arr = np.asarray(poly)
    for eje, dim in enumerate([B, H, D]):
        for valor in (0.0, dim):
            if np.all(np.abs(poly_arr[:, eje] - valor) < eps):
                return True
    return False



def _dir_aleatoria():
    theta = random.uniform(0.0, 2.0 * math.pi)
    phi   = math.acos(random.uniform(-1.0, 1.0))
    return np.array([math.sin(phi) * math.cos(theta),
                     math.sin(phi) * math.sin(theta),
                     math.cos(phi)])


def _perp_a(u):
    """Devuelve dos vectores unitarios ortogonales entre sí y a u."""
    ref = np.array([1., 0., 0.]) if abs(u[0]) < 0.9 else np.array([0., 1., 0.])
    p1 = np.cross(u, ref);  p1 /= np.linalg.norm(p1)
    p2 = np.cross(u, p1);   p2 /= np.linalg.norm(p2)
    return p1, p2


import concurrent.futures as _cf

_EXEC = None

# Crea (una sola vez) el pool de procesos para las tareas paralelas. Usa n_procesos
# si se fijó, o todos los núcleos disponibles.
def _get_exec():
    global _EXEC
    if _EXEC is None:
        nproc = n_procesos if n_procesos and n_procesos > 0 else (os.cpu_count() or 1)
        _EXEC = _cf.ProcessPoolExecutor(max_workers=max(1, nproc))
    return _EXEC


# Worker: ordena y recorta una cara interna a partir de sus vértices crudos. Se
# ejecuta en los procesos del pool, por eso no toca estructuras globales.
def _geom_cara_interna(arista_vor):
    a = np.asarray(arista_vor, dtype=float)
    a = a[~np.isnan(a).any(axis=1)]
    if len(a) < 3:
        return None
    loop = recortar_poligono_caja(ordenar_poligono_planar(a))
    if len(loop) < 3:
        return None
    return loop


# Calcula en paralelo la geometría de muchas caras internas. Con pocas caras o sin
# paralelismo lo resuelve en serie.
def _geom_caras_paralelo(lista_aristas):
    if not usar_paralelo or len(lista_aristas) < 1500:
        return [_geom_cara_interna(a) for a in lista_aristas]
    try:
        ex = _get_exec()
        return list(ex.map(_geom_cara_interna, lista_aristas, chunksize=128))
    except Exception:
        return [_geom_cara_interna(a) for a in lista_aristas]


def voronoi_base(todas_semillas=None):
    if todas_semillas is None:
        vol_prom = float((fracciones * vols_med).sum())
        N = int(round(vol_agregado_total / max(vol_prom, 1e-9)))
        N = max(400, min(max_semillas, N))
        todas_semillas = np.column_stack([
            np.random.rand(N) * B,
            np.random.rand(N) * H,
            np.random.rand(N) * D,
        ])

    N_total = len(todas_semillas)

    # puntos ficticios fuera del dominio para que las celdas del borde queden completamente cerradas
    cx, cy, cz = B / 2.0, H / 2.0, D / 2.0
    L = max(B, H, D) * 10.0
    fantasmas = np.array([
        [cx + dx * L, cy + dy * L, cz + dz * L]
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
        if not (dx == 0 and dy == 0 and dz == 0)
    ])

    vor = Voronoi(np.vstack([todas_semillas, fantasmas]))

    celdas       = []
    semilla_celda  = []
    perdidas_inf  = 0
    perdidas_recorte = 0
    for i in range(N_total):
        region = vor.regions[vor.point_region[i]]
        idx_v  = [v for v in region if v >= 0]
        if len(idx_v) < 4:
            perdidas_inf += 1
            continue
        rec = recortar_caja(vor.vertices[idx_v])
        if rec is not None and len(rec) >= 4:
            celdas.append(rec)
            semilla_celda.append(i)
        else:
            perdidas_recorte += 1

    if perdidas_inf or perdidas_recorte:
        print(f"  Aviso: se perdieron {perdidas_inf} celdas no acotadas + "
              f"{perdidas_recorte} por recorte (de {N_total})")

    return celdas, semilla_celda, vor



def asignar_clusters_voronoi(celdas, semilla_celda, vor, vol_por_clase, pos_borde):
    # Agrupa las celdas en partículas de agregado haciéndolas crecer por regiones.
    # Permite el contacto entre agregados (no deja matriz de separación); solo el
    # borde del dominio queda como matriz.
    n_tipos = len(tipos_agregado)

    seed_to_pos = {s: pos for pos, s in enumerate(semilla_celda)}
    vecinos = {}
    for rp in vor.ridge_points:
        s1, s2 = int(rp[0]), int(rp[1])
        p1 = seed_to_pos.get(s1)
        p2 = seed_to_pos.get(s2)
        if p1 is None or p2 is None:
            continue
        vecinos.setdefault(p1, set()).add(p2)
        vecinos.setdefault(p2, set()).add(p1)

    # Centroides rápidos (media de vértices, sin ConvexHull)
    centroides = [np.mean(np.asarray(c, dtype=float), axis=0) for c in celdas]
    # Volumen uniforme por celda: suficiente para el criterio de parada del crecimiento.
    # Voronoi particiona exactamente el dominio, así que la suma es vol_dominio.
    vol_cel_uniforme = vol_dominio / max(1, len(celdas))
    vols_celdas = [vol_cel_uniforme] * len(celdas)

    # Lista de partículas por tipo (mayor volumen primero). Después se entrelazan
    # los tipos para que cada uno reciba espacio en proporción a su número de
    # partículas y los tipos grandes no acaparen todo.
    specs_por_tipo = [[] for _ in range(n_tipos)]
    for k, (_, d_min, d_max, _) in enumerate(tipos_agregado):
        v_min = vol_esfera(d_min)
        v_max = vol_esfera(d_max)
        v_med = 0.5 * (v_min + v_max)
        n_part = max(1, round(vol_por_clase[k] / v_med))
        for _ in range(n_part):
            v_obj = random.uniform(v_min, v_max)
            r = random.random()
            if r < porc_elongadas:
                specs_por_tipo[k].append((k, v_obj, 'elong', _dir_aleatoria()))
            elif r < porc_elongadas + porc_aplanadas:
                specs_por_tipo[k].append((k, v_obj, 'aplana', _dir_aleatoria()))
            else:
                specs_por_tipo[k].append((k, v_obj, 'normal', None))
        specs_por_tipo[k].sort(key=lambda x: -x[1])  # mayor primero dentro del tipo

    # Entrelaza los tipos: la partícula i del tipo k se ubica en la posición
    # (i + 0.5) * n_total / n_k de la secuencia global.
    n_parts_tipo = [len(l) for l in specs_por_tipo]
    n_total_specs = sum(n_parts_tipo)
    orden = []
    for k, lista in enumerate(specs_por_tipo):
        nk = max(n_parts_tipo[k], 1)
        for i in range(len(lista)):
            pos_global = (i + 0.5) * n_total_specs / nk
            orden.append((pos_global, k, i))
    orden.sort()
    specs = [specs_por_tipo[k][i] for _, k, i in orden]

    libres    = set(range(len(celdas))) - pos_borde
    reservadas = set(pos_borde)

    cascaras_por_tipo  = [{} for _ in range(n_tipos)]
    vols_objetivo_tipo = [{} for _ in range(n_tipos)]
    acumulados         = [0.0] * n_tipos
    cluster_de_celda   = {}
    forma_de_cluster   = {}
    cid_counter        = 0

    for k, v_obj, forma, axis in specs:
        if acumulados[k] >= vol_por_clase[k] * (1.0 + tol):
            continue
        candidatos = list(libres - reservadas)
        if not candidatos:
            break

        # Elegir semilla: subsample + máximo vecinos libres disponibles
        muestra = candidatos if len(candidatos) <= 200 else random.sample(candidatos, 200)
        pos0 = max(muestra, key=lambda p: sum(
            1 for n in vecinos.get(p, set()) if n in libres and n not in reservadas))
        libres.discard(pos0)

        cluster     = {pos0}
        n_cluster   = 1
        vol_cluster = vols_celdas[pos0]
        centro      = np.array(centroides[pos0], dtype=float)
        frontera    = {n for n in vecinos.get(pos0, set())
                       if n in libres and n not in reservadas}

        while vol_cluster < v_obj and frontera:
            if forma == 'elong' and axis is not None:
                mejor = max(frontera, key=lambda p: abs(
                    float(np.dot(centroides[p] - centro, axis))))
            elif forma == 'aplana' and axis is not None:
                mejor = min(frontera, key=lambda p: abs(
                    float(np.dot(centroides[p] - centro, axis))))
            else:
                mejor = min(frontera, key=lambda p:
                    float(np.dot(centroides[p] - centro,
                                 centroides[p] - centro)))

            cluster.add(mejor)
            libres.discard(mejor)
            frontera.discard(mejor)
            vol_cluster += vols_celdas[mejor]
            # Media incremental O(1) en lugar de O(n) cada paso
            centro = (centro * n_cluster + centroides[mejor]) / (n_cluster + 1)
            n_cluster += 1
            for n in vecinos.get(mejor, set()):
                if n in libres and n not in reservadas and n not in cluster:
                    frontera.add(n)

        for pos in cluster:
            cascaras_por_tipo[k][pos]  = celdas[pos]
            vols_objetivo_tipo[k][pos] = v_obj
            cluster_de_celda[pos]      = cid_counter
        acumulados[k]  += vol_cluster
        forma_de_cluster[cid_counter] = forma
        cid_counter += 1
        # Contacto permitido: no se reserva matriz alrededor del cluster, así dos
        # agregados pueden compartir cara. La única matriz forzada es el borde.

    # ── Relleno por contacto ─────────────────────────────────────────────────
    # Se permite que los agregados se toquen, así que las celdas interiores que
    # quedaron libres se reparten entre los agregados (cada una se une al vecino
    # más cercano) en vez de dejarse como matriz de separación. El relleno crece
    # desde los agregados hacia afuera y se detiene al alcanzar la fracción
    # objetivo (frac_agregado); el resto queda como matriz interior.
    tipo_de_cluster = {}
    for k_r, casc_r in enumerate(cascaras_por_tipo):
        for p_r in casc_r:
            tipo_de_cluster[cluster_de_celda[p_r]] = k_r

    vol_agg     = len(cluster_de_celda) * vol_cel_uniforme
    objetivo_agg = vol_agregado_total

    frontera_rec = set()
    for p_r in list(cluster_de_celda):
        for n_r in vecinos.get(p_r, set()):
            if n_r not in cluster_de_celda and n_r not in pos_borde:
                frontera_rec.add(n_r)

    # El relleno avanza por niveles (BFS) desde los agregados hacia afuera, así que
    # las celdas más cercanas se toman primero sin necesidad de ordenar.
    while frontera_rec and vol_agg < objetivo_agg:
        siguiente_rec = set()
        for cell in frontera_rec:
            if vol_agg >= objetivo_agg:
                break
            if cell in cluster_de_celda or cell in pos_borde:
                continue
            vec_asig = [nb for nb in vecinos.get(cell, set())
                        if nb in cluster_de_celda]
            if not vec_asig:
                continue
            cc = centroides[cell]
            nb_mejor = min(vec_asig, key=lambda nb: float(
                np.dot(centroides[nb] - cc, centroides[nb] - cc)))
            cid = cluster_de_celda[nb_mejor]
            k_r = tipo_de_cluster[cid]
            cascaras_por_tipo[k_r][cell] = celdas[cell]
            cluster_de_celda[cell] = cid
            vol_agg += vol_cel_uniforme
            for nb in vecinos.get(cell, set()):
                if nb not in cluster_de_celda and nb not in pos_borde:
                    siguiente_rec.add(nb)
        frontera_rec = siguiente_rec
    # ─────────────────────────────────────────────────────────────────────────

    agg_poss  = set(cluster_de_celda)
    matriz_d  = {pos: celdas[pos] for pos in range(len(celdas)) if pos not in agg_poss}

    clase_de_semilla = {}
    for k, casc in enumerate(cascaras_por_tipo):
        for pos in casc:
            clase_de_semilla[semilla_celda[pos]] = k + 1
    for pos in matriz_d:
        clase_de_semilla[semilla_celda[pos]] = 0

    n_clusters_tipo  = [len({cluster_de_celda[p] for p in casc})
                        for casc in cascaras_por_tipo]
    n_elong  = sum(1 for f in forma_de_cluster.values() if f == 'elong')
    n_aplana = sum(1 for f in forma_de_cluster.values() if f == 'aplana')

    return (cascaras_por_tipo, vols_objetivo_tipo, matriz_d, clase_de_semilla,
            cluster_de_celda, forma_de_cluster, n_clusters_tipo, n_elong, n_aplana)



def caras_triangulares(verts):
    try:
        pts = np.asarray(verts, dtype=float)
        h   = ConvexHull(pts)
        return [pts[s] for s in h.simplices]
    except Exception:
        return []


def pintar_ejes_3d(ax, caras_mat, caras_clase, alpha_mat, alpha_agg, titulo):
    if caras_mat:
        ax.add_collection3d(Poly3DCollection(
            caras_mat,
            alpha=alpha_mat,
            facecolor=obtener_color(0),
            edgecolor=(0.45, 0.45, 0.45),
            linewidth=0.05,
        ))
    for clase, caras in caras_clase:
        if caras:
            ax.add_collection3d(Poly3DCollection(
                caras,
                alpha=alpha_agg,
                facecolor=obtener_color(clase),
                edgecolor=(0.10, 0.10, 0.10),
                linewidth=0.08,
            ))

    for x in [0, B]:
        for y in [0, H]:
            ax.plot([x, x], [y, y], [0, D], 'k-', lw=0.5)
    for x in [0, B]:
        for z in [0, D]:
            ax.plot([x, x], [0, H], [z, z], 'k-', lw=0.5)
    for y in [0, H]:
        for z in [0, D]:
            ax.plot([0, B], [y, y], [z, z], 'k-', lw=0.5)

    ax.set_xlim(0, B); ax.set_ylim(0, H); ax.set_zlim(0, D)
    ax.set_xlabel('x [mm]'); ax.set_ylabel('y [mm]'); ax.set_zlabel('z [mm]')
    ax.set_title(titulo, fontsize=11)
    ax.view_init(elev=28, azim=-50)


def graficar_3d(matriz_verts, agregados_grafico):
    from matplotlib.patches import Patch
    leyenda = [Patch(facecolor=obtener_color(k + 1), label=tipos_agregado[k][0])
               for k in range(len(tipos_agregado))]
    leyenda.append(Patch(facecolor=obtener_color(0), label='Matriz'))

    caras_mat = []
    for v in matriz_verts:
        caras_mat.extend(caras_triangulares(v))

    caras_clase = []
    for clase_id, verts_list in agregados_grafico:
        cs = []
        for v in verts_list:
            cs.extend(caras_triangulares(v))
        caras_clase.append((clase_id, cs))

    fig1 = plt.figure(figsize=(9, 8))
    ax1  = fig1.add_subplot(111, projection='3d')
    pintar_ejes_3d(ax1, caras_mat, caras_clase,
                   alpha_mat=0.05, alpha_agg=0.90,
                   titulo='Agregados (matriz transparente)')
    ax1.legend(handles=leyenda, loc='upper right')
    fig1.tight_layout()

    fig2 = plt.figure(figsize=(9, 8))
    ax2  = fig2.add_subplot(111, projection='3d')
    pintar_ejes_3d(ax2, caras_mat, caras_clase,
                   alpha_mat=0.55, alpha_agg=0.90,
                   titulo='Mezcla completa (matriz visible)')
    ax2.legend(handles=leyenda, loc='upper right')
    fig2.tight_layout()

    plt.show()



def resumen(cascaras_por_tipo, matriz_verts):
    # Usa conteo de celdas × volumen uniforme para evitar N_fino llamadas a ConvexHull.
    n_total_celdas = sum(len(c) for c in cascaras_por_tipo) + len(matriz_verts)
    vc = vol_dominio / max(1, n_total_celdas)

    n_tipo   = [len(casc) for casc in cascaras_por_tipo]
    vols_tipo = [n * vc for n in n_tipo]
    Vt = sum(vols_tipo)
    Vm = len(matriz_verts) * vc

    print("\nResumen:")
    print(f"Volumen dominio    = {vol_dominio:12.1f} mm3")
    print(f"Objetivo agregados = {vol_agregado_total:12.1f} mm3  "
          f"({100*vol_agregado_total/vol_dominio:.1f}%)")
    print(f"Agregados logrados = {Vt:12.1f} mm3  "
          f"({100*Vt/vol_dominio:.1f}%)")
    print(f"Matriz             = {Vm:12.1f} mm3  "
          f"({100*Vm/vol_dominio:.1f}%)")
    if Vt > 0:
        for k, (nombre, _, _, _) in enumerate(tipos_agregado):
            V = vols_tipo[k]
            n = n_tipo[k]
            print(f'  {nombre:6s}  {n:5d} celdas  {V:10.1f} mm3  ({100*V/Vt:.1f}%)')



def caras_del_casco(hull):
    def canonico(eq):
        n, d = eq[:3].copy(), float(eq[3])
        for i in range(3):
            if abs(n[i]) > 1e-8:
                if n[i] < 0:
                    n, d = -n, -d
                break
        return tuple(np.round(np.append(n, d), 5))

    grupos = {}
    for i, eq in enumerate(hull.equations):
        grupos.setdefault(canonico(eq), []).append(i)

    caras = []
    for ids in grupos.values():
        tris = [hull.simplices[i] for i in ids]

        cuenta = {}
        for tri in tris:
            for j in range(3):
                a, b = int(tri[j]), int(tri[(j + 1) % 3])
                k = (min(a, b), max(a, b))
                cuenta[k] = cuenta.get(k, 0) + 1
        borde = [k for k, c in cuenta.items() if c == 1]
        if not borde:
            continue

        ady = {}
        for a, b in borde:
            ady.setdefault(a, []).append(b)
            ady.setdefault(b, []).append(a)

        aristas_pend = set(borde)
        while aristas_pend:
            primero  = next(iter(aristas_pend))
            inicio   = primero[0]
            poligono = [inicio]
            prev, curr = None, inicio

            while True:
                sig = None
                for v in ady.get(curr, []):
                    if v == prev:
                        continue
                    e = (min(curr, v), max(curr, v))
                    if e in aristas_pend:
                        sig = v
                        break

                if sig is None:
                    break

                e = (min(curr, sig), max(curr, sig))
                aristas_pend.discard(e)

                if sig == inicio:
                    break

                poligono.append(sig)
                prev, curr = curr, sig

            if len(poligono) >= 3:
                caras.append(poligono)

    return caras




# Limpia un ciclo de nodos: quita repetidos consecutivos, el cierre duplicado y
# cualquier nodo que aparezca más de una vez.
def _dedup_loop(ids):
    out = []
    for v in ids:
        if not out or out[-1] != v:
            out.append(v)
    if len(out) > 1 and out[0] == out[-1]:
        out.pop()
    if len(set(out)) != len(out):
        visto = set(); uniq = []
        for v in out:
            if v not in visto:
                visto.add(v); uniq.append(v)
        out = uniq
    return out


# Corrige las uniones en T sobre las caras del borde: si un nodo de otra cara cae
# sobre una arista de borde, lo inserta en el ciclo para que las caras casen. La
# búsqueda de nodos colineales se hace por bloque (numpy) para no ser O(N^2).
def _corregir_t_borde(nodo_coords, faces):
    eps = 1e-9
    planos = [(0, 0.0), (0, float(B)), (1, 0.0), (1, float(H)), (2, 0.0), (2, float(D))]
    coords = np.asarray(nodo_coords, dtype=float)
    # nodos sobre cada plano del cubo, en arrays, para buscar de a bloque
    plano_ids = {}
    plano_xyz = {}
    for (eje, val) in planos:
        sel = np.where(np.abs(coords[:, eje] - val) < eps)[0]
        plano_ids[(eje, val)] = sel
        plano_xyz[(eje, val)] = coords[sel]

    for fid in range(len(faces)):
        loop = faces[fid]
        if len(loop) < 3:
            continue
        pl = None
        for (eje, val) in planos:
            if all(abs(coords[n][eje] - val) < eps for n in loop):
                pl = (eje, val); break
        if pl is None:
            continue
        cand_ids = plano_ids[pl]
        cand_xyz = plano_xyz[pl]
        if len(cand_ids) == 0:
            faces[fid] = _dedup_loop(loop)
            continue
        nuevo = []
        m = len(loop)
        for k in range(m):
            a = loop[k]; b = loop[(k + 1) % m]
            nuevo.append(a)
            pa = coords[a]; pb = coords[b]
            ab = pb - pa; L2 = float(ab @ ab)
            if L2 < 1e-18:
                continue
            L = math.sqrt(L2)
            # tolerancia relativa al largo de la arista (sirve a cualquier escala)
            tol_perp = max(1e-7, 1e-4 * L)
            tol_t    = tol_perp / L
            t = (cand_xyz - pa) @ ab / L2
            proj = pa + t[:, None] * ab
            d = np.linalg.norm(cand_xyz - proj, axis=1)
            mask = ((t > tol_t) & (t < 1.0 - tol_t) & (d < tol_perp) &
                    (cand_ids != a) & (cand_ids != b))
            if mask.any():
                tt = t[mask]; ii = cand_ids[mask]
                for j in np.argsort(tt, kind="stable"):
                    nuevo.append(int(ii[j]))
        faces[fid] = _dedup_loop(nuevo)


# Red de seguridad: si la superficie de un elemento queda con aristas usadas un
# número impar de veces (un hueco residual por degeneración), las agrupa en ciclos
# y los tapa con un abanico de triángulos hasta un nodo central nuevo.
def _cerrar_huecos(faces_fin, owners_fin, nodo_coords):
    elem_fids = {}
    for fid, owners in enumerate(owners_fin):
        for e in owners:
            elem_fids.setdefault(e, []).append(fid)

    def _find(parent, x):
        r = x
        while parent[r] != r:
            r = parent[r]
        while parent[x] != r:
            parent[x], x = r, parent[x]
        return r

    cap_req = {}
    for e, fids in elem_fids.items():
        cnt = {}
        for fid in fids:
            loop = faces_fin[fid]
            if loop is None:
                continue
            m = len(loop)
            for k in range(m):
                a = loop[k]; b = loop[(k + 1) % m]
                if a == b:
                    continue
                key = (a, b) if a < b else (b, a)
                cnt[key] = cnt.get(key, 0) + 1
        bedges = [k for k, c in cnt.items() if c % 2 == 1]
        if not bedges:
            continue
        parent = {}
        for a, b in bedges:
            parent.setdefault(a, a); parent.setdefault(b, b)
        for a, b in bedges:
            ra, rb = _find(parent, a), _find(parent, b)
            if ra != rb:
                parent[ra] = rb
        comp = {}
        for a, b in bedges:
            comp.setdefault(_find(parent, a), []).append((a, b))
        for edges in comp.values():
            if len(edges) < 3:
                continue
            cap_req.setdefault(frozenset(edges), []).append((e, edges))

    n_tapados = 0
    for _, reqs in cap_req.items():
        owners = [e for e, _ in reqs]
        edges = reqs[0][1]
        nodos = sorted({n for ed in edges for n in ed})
        cc = np.mean([nodo_coords[n] for n in nodos], axis=0)
        cid = len(nodo_coords)
        nodo_coords.append(np.asarray(cc, dtype=float))
        for a, b in edges:
            faces_fin.append([cid, a, b])
            owners_fin.append(list(owners))
        n_tapados += 1
    return n_tapados


# Cuando dos vértices de Voronoi caen casi en el mismo punto, la celda no cierra.
# En cada celda con aristas sueltas se funde el par de nodos más cercano y, si eso
# daña a un vecino, se revierte. Las celdas que ya estaban bien no se tocan.
def _reparar_near_dups(faces_fin, owners_fin, nodo_coords, tol_par=1e-2):
    def elem_fids_map():
        m = {}
        for fid, ow in enumerate(owners_fin):
            if faces_fin[fid] is None:
                continue
            for e in ow:
                m.setdefault(e, []).append(fid)
        return m

    def aristas_impares(fids):
        cnt = {}
        for fid in fids:
            loop = faces_fin[fid]
            if loop is None:
                continue
            m = len(loop)
            for k in range(m):
                a = loop[k]; b = loop[(k + 1) % m]
                if a == b:
                    continue
                key = (a, b) if a < b else (b, a)
                cnt[key] = cnt.get(key, 0) + 1
        return sum(1 for c in cnt.values() if c % 2 == 1)

    def nodo_fids():
        m = {}
        for fid, loop in enumerate(faces_fin):
            if loop is None:
                continue
            for n in loop:
                m.setdefault(n, set()).add(fid)
        return m

    elem = elem_fids_map()
    candidatos = [e for e, f in elem.items() if aristas_impares(f) > 0]
    if not candidatos:
        return 0
    n2f = nodo_fids()
    reparados = 0
    for e in candidatos:
        for _ in range(8):
            fids = elem_fids_map().get(e, [])
            if aristas_impares(fids) == 0:
                break
            alln = sorted({n for fid in fids for n in faces_fin[fid]})
            P = np.array([nodo_coords[n] for n in alln])
            best = None; bd = 1e18
            for i in range(len(alln)):
                dif = P[i + 1:] - P[i]
                if len(dif) == 0:
                    continue
                d2 = np.einsum('ij,ij->i', dif, dif)
                jr = int(d2.argmin())
                if d2[jr] < bd:
                    bd = d2[jr]; best = (alln[i], alln[i + 1 + jr])
            if best is None or bd ** 0.5 > tol_par:
                break
            a, b = best
            fb = list(n2f.get(b, set()))
            snap = {fid: (list(faces_fin[fid]) if faces_fin[fid] is not None
                          else None) for fid in fb}
            afect = set()
            for fid in fb:
                if faces_fin[fid] is not None:
                    afect.update(owners_fin[fid])
            imp_antes = {ae: aristas_impares(elem.get(ae, [])) for ae in afect}
            for fid in fb:
                loop = faces_fin[fid]
                if loop is None:
                    continue
                nl = [a if x == b else x for x in loop]
                cl = []
                for x in nl:
                    if not cl or cl[-1] != x:
                        cl.append(x)
                while len(cl) > 1 and cl[0] == cl[-1]:
                    cl.pop()
                faces_fin[fid] = cl if len(cl) >= 3 else None
            n2f[a] = n2f.get(a, set()) | n2f.get(b, set())
            n2f[b] = set()
            elem = elem_fids_map()
            malo = False
            for ae in afect:
                if ae == e:
                    continue
                if aristas_impares(elem.get(ae, [])) > imp_antes.get(ae, 0):
                    malo = True; break
            if malo:
                for fid, old in snap.items():
                    faces_fin[fid] = old
                n2f = nodo_fids()
                elem = elem_fids_map()
                break
        if aristas_impares(elem_fids_map().get(e, [])) == 0:
            reparados += 1
    return reparados


# El VEM necesita caras planas. La fusión de nodos deja unas pocas caras levemente
# alabeadas: se parten en triángulos (un triángulo siempre es plano), reemplazando
# la cara en sus dos dueños para no romper la conformidad.
def _aplanar_caras(faces_fin, owners_fin, nodo_coords, tol=1e-9):
    n_tri = 0
    for fid in range(len(faces_fin)):
        loop = faces_fin[fid]
        if loop is None or len(loop) < 4:
            continue
        pts = np.array([nodo_coords[n] for n in loop])
        c = pts.mean(axis=0)
        try:
            _, _, Vt = np.linalg.svd(pts - c, full_matrices=False)
        except Exception:
            continue
        nrm = Vt[-1]
        desv = float(np.abs((pts - c) @ nrm).max())
        diam = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))
        if diam <= 1e-12 or desv / diam <= tol:
            continue
        cid = len(nodo_coords)
        nodo_coords.append(np.asarray(c, dtype=float))
        ow = owners_fin[fid]
        m = len(loop)
        faces_fin[fid] = [cid, loop[0], loop[1]]
        for k in range(1, m):
            faces_fin.append([cid, loop[k], loop[(k + 1) % m]])
            owners_fin.append(list(ow))
        n_tri += 1
    return n_tri








# Estira/aplana vectores a lo largo de un eje conservando el volumen.
def _transformar(d, eje, factor):
    u = np.asarray(eje, dtype=float); u = u / np.linalg.norm(u)
    proy = np.outer(d @ u, u)
    perp = d - proy
    return proy * factor + perp / math.sqrt(factor)


# Construye la malla conforme para el VEM y la escribe en los .txt: nodos, aristas,
# caras y elementos. Arma las caras internas (compartidas entre dos celdas) y las
# de borde, corrige uniones en T, repara celdas con vértices casi coincidentes,
# aplana las caras alabeadas, y finalmente exporta y valida.
def generar_malla_fem(celdas, clases_de_celda, semillas_de_celda, vor, deform_specs=None):
    if deform_specs is None:
        deform_specs = {}
    eps_b = 1e-9
    seed_to_pos = {s: pos for pos, s in enumerate(semillas_de_celda)}
    semillas_validas = set(semillas_de_celda)

    # Asigna un id único a cada nodo, pegando al plano del borde los que están casi
    # encima y fusionando coordenadas iguales (redondeo) para que las caras casen.
    REDONDEO = 8
    nodo_id = {}
    nodo_coords = []

    def id_nodo(p):
        v = np.asarray(p, dtype=float).copy()
        for eje, val in ((0, 0.0), (0, float(B)), (1, 0.0), (1, float(H)),
                         (2, 0.0), (2, float(D))):
            if abs(v[eje] - val) < 1e-9:
                v[eje] = val
        k = tuple(np.round(v, REDONDEO))
        if k not in nodo_id:
            nodo_id[k] = len(nodo_coords)
            nodo_coords.append(v)
        return nodo_id[k]

    faces        = []
    face_owners  = []
    elem_faces   = {}

    # Caras internas: primero se juntan los ridges válidos y se calcula su geometría
    # en paralelo (lo pesado); luego se registran los nodos en serie (rápido).
    ridges_validos = []
    for r_idx, vert_idxs in enumerate(vor.ridge_vertices):
        s1, s2 = int(vor.ridge_points[r_idx][0]), int(vor.ridge_points[r_idx][1])
        if s1 not in semillas_validas or s2 not in semillas_validas:
            continue
        if -1 in vert_idxs:
            continue
        p1 = seed_to_pos.get(s1); p2 = seed_to_pos.get(s2)
        if p1 is None or p2 is None:
            continue
        av = vor.vertices[vert_idxs]
        av = av[~np.isnan(av).any(axis=1)]
        if len(av) < 3:
            continue
        ridges_validos.append((p1, p2, av))

    loops = _geom_caras_paralelo([rv[2] for rv in ridges_validos])

    # Registra cada cara interna: da id a sus nodos y la asocia a las dos celdas
    # que la comparten.
    for (p1, p2, _av), loop in zip(ridges_validos, loops):
        if loop is None or len(loop) < 3:
            continue
        ids = _dedup_loop([id_nodo(p) for p in loop])
        if len(ids) < 3:
            continue
        fid = len(faces)
        faces.append(ids); face_owners.append([p1, p2])
        elem_faces.setdefault(p1, []).append(fid)
        elem_faces.setdefault(p2, []).append(fid)

    # Caras de borde del dominio
    planos_caja = [(0, 0.0), (0, float(B)), (1, 0.0), (1, float(H)),
                   (2, 0.0), (2, float(D))]
    for pos, verts in enumerate(celdas):
        pts = np.asarray(verts, dtype=float)
        for eje, val in planos_caja:
            en = pts[np.abs(pts[:, eje] - val) < eps_b]
            if len(en) < 3:
                continue
            en = en.copy(); en[:, eje] = val
            loop = ordenar_poligono_planar(en)
            if len(loop) < 3:
                continue
            ids = _dedup_loop([id_nodo(p) for p in loop])
            if len(ids) < 3:
                continue
            fid = len(faces)
            faces.append(ids); face_owners.append([pos])
            elem_faces.setdefault(pos, []).append(fid)

    _corregir_t_borde(nodo_coords, faces)

    # Nodos que NO se pueden mover sin romper conformidad: los del borde y los
    # compartidos por dos agregados distintos.
    eps_brd = 1e-7
    nodo_en_borde = set()
    for nid, c in enumerate(nodo_coords):
        if (abs(c[0]) < eps_brd or abs(c[0] - B) < eps_brd or
            abs(c[1]) < eps_brd or abs(c[1] - H) < eps_brd or
            abs(c[2]) < eps_brd or abs(c[2] - D) < eps_brd):
            nodo_en_borde.add(nid)

    aggs_de_nodo = {}
    for pos, fids in elem_faces.items():
        if clases_de_celda[pos] <= 0:
            continue
        for fid in fids:
            for n in faces[fid]:
                aggs_de_nodo.setdefault(n, set()).add(pos)

    # Elongación / aplanamiento: solo se mueven nodos libres (no borde y de un solo
    # agregado), para mantener la conformidad.
    nodos_movidos = set()
    for pos, (factor, axis) in deform_specs.items():
        fids = elem_faces.get(pos, [])
        node_ids = list({n for fid in fids for n in faces[fid]})
        if len(node_ids) < 4:
            continue
        if any(n in nodo_en_borde for n in node_ids):
            continue
        movibles = [n for n in node_ids
                    if n not in nodo_en_borde and len(aggs_de_nodo.get(n, ())) <= 1]
        if len(movibles) < 4:
            continue
        coords = np.array([nodo_coords[n] for n in node_ids], dtype=float)
        c = coords.mean(axis=0)
        try:
            v_old = float(ConvexHull(coords).volume)
        except Exception:
            continue
        d = _transformar(coords - c, axis, factor)
        try:
            v_new = float(ConvexHull(c + d).volume)
        except Exception:
            v_new = v_old
        f_vol = (v_old / v_new) ** (1.0 / 3.0) if v_new > 1e-12 else 1.0
        movset = set(movibles)
        for n, nc in zip(node_ids, c + d * f_vol):
            if n in movset:
                nodo_coords[n] = nc
                nodos_movidos.add(n)

    # Triangular caras tocadas por la deformación; las demás quedan como polígono.
    faces_fin   = []
    owners_fin  = []

    def add_face(loop, owners):
        faces_fin.append(loop); owners_fin.append(owners)
        return len(faces_fin) - 1

    for fid in range(len(faces)):
        loop = faces[fid]
        owners = face_owners[fid]
        if len(loop) < 3:
            continue
        tocada = any(n in nodos_movidos for n in loop)
        if not tocada or len(loop) == 3:
            add_face(loop, owners)
        else:
            cc = np.mean([nodo_coords[n] for n in loop], axis=0)
            cid = len(nodo_coords)
            nodo_coords.append(np.asarray(cc, dtype=float))
            mloop = len(loop)
            for k in range(mloop):
                a = loop[k]; b = loop[(k + 1) % mloop]
                if a == b:
                    continue
                add_face([cid, a, b], owners)

    n_reparados = _reparar_near_dups(faces_fin, owners_fin, nodo_coords)
    if n_reparados:
        print(f"  (se repararon {n_reparados} celdas con vértices casi coincidentes)")

    n_tapados = _cerrar_huecos(faces_fin, owners_fin, nodo_coords)
    if n_tapados:
        print(f"  (se cerraron {n_tapados} huecos residuales por degeneración)")

    _aplanar_caras(faces_fin, owners_fin, nodo_coords)

    # Construir aristas, caras y elementos
    arista_id = {}
    aristas   = []

    def id_arista(a, b):
        ka = frozenset((a, b))
        if ka not in arista_id:
            arista_id[ka] = len(aristas)
            aristas.append((min(a, b), max(a, b)))
        return arista_id[ka]

    caras = []
    elem_faces_fin = {}
    for fid, loop in enumerate(faces_fin):
        if loop is None or len(loop) < 3:
            continue
        m = len(loop)
        eids = [id_arista(loop[k], loop[(k + 1) % m]) for k in range(m)]
        new_fid = len(caras)
        caras.append(eids)
        for pos in owners_fin[fid]:
            elem_faces_fin.setdefault(pos, []).append(new_fid)

    # Cada elemento es la lista de caras de una celda; guarda también su clase.
    elementos = []
    elem_clase = []
    for pos in range(len(celdas)):
        fids = elem_faces_fin.get(pos)
        if not fids:
            continue
        elementos.append(list(dict.fromkeys(fids)))
        elem_clase.append(clases_de_celda[pos])

    # Escribe los cuatro .txt de entrada del solver. Las coordenadas pasan a metros.
    N = len(nodo_coords)
    ruta_nodos = os.path.join(CARPETA, "input_nodos.txt")
    ruta_aris  = os.path.join(CARPETA, "input_aristas.txt")
    ruta_caras = os.path.join(CARPETA, "input_caras.txt")
    ruta_elem  = os.path.join(CARPETA, "input_elementos.txt")

    with open(ruta_nodos, 'w') as f:
        for i, c in enumerate(nodo_coords):
            f.write(f"{i} {N + i} {2 * N + i} "
                    f"{c[0]/1000:.18e} {c[1]/1000:.18e} {c[2]/1000:.18e} "
                    f"0 0 0 0 0 0\n")
    with open(ruta_aris, 'w') as f:
        for i, (a, b) in enumerate(aristas):
            f.write(f"{i} {a} {b}\n")
    with open(ruta_caras, 'w') as f:
        for i, ars in enumerate(caras):
            f.write(f"{i} " + " ".join(map(str, ars)) + "\n")
    with open(ruta_elem, 'w') as f:
        for i, (cs, cl) in enumerate(zip(elementos, elem_clase)):
            f.write(f"{i} {cl} {cl} " + " ".join(map(str, cs)) + "\n")

    ruta_props = os.path.join(CARPETA, "input_propiedades.txt")
    with open(ruta_props, 'w') as f:
        f.write("0 0 0\n")
        for k in range(len(tipos_agregado)):
            f.write(f"{k + 1} 0 0\n")

    print(f"\nMalla: {len(elementos)} elementos  {N} nodos  {len(caras)} caras")
    validar_malla(aristas, caras, elementos, nodo_coords)


# Revisa la malla y avisa de cualquier defecto: caras que no cierran o con aristas
# repetidas, caras no planas, caras huérfanas o compartidas por más de dos
# elementos, elementos que no son poliedros cerrados (V-E+F != 2) y agujeros en el
# borde del cubo. Imprime "OK" solo si todo está bien.
def validar_malla(aristas, caras, elementos, nodo_coords):
    caras_rotas        = []
    caras_con_dup_aris = []
    caras_no_planares  = []

    for cid, ars in enumerate(caras):
        if len(ars) < 3:
            caras_rotas.append(cid); continue
        if len(set(ars)) != len(ars):
            caras_con_dup_aris.append(cid)
        ok = True
        for k in range(len(ars)):
            a = aristas[ars[k]]; b = aristas[ars[(k + 1) % len(ars)]]
            if not (set(a) & set(b)):
                ok = False; break
        if not ok:
            caras_rotas.append(cid); continue

        nodos_cara = []
        for arid in ars:
            for nid in aristas[arid]:
                if nid not in nodos_cara: nodos_cara.append(nid)
        if len(nodos_cara) >= 4:
            coords_c  = np.array([nodo_coords[n] for n in nodos_cara])
            centroide = coords_c.mean(axis=0)
            centrada  = coords_c - centroide
            try:
                _, S, Vt = np.linalg.svd(centrada, full_matrices=False)
                n_pl = Vt[-1]
                max_abs = float(np.abs(centrada @ n_pl).max())
                diam = max(float(np.linalg.norm(coords_c[i] - coords_c[j]))
                           for i in range(len(coords_c))
                           for j in range(i + 1, len(coords_c)))
                if diam > 1e-10 and max_abs / diam > 1e-9:
                    caras_no_planares.append((cid, max_abs, max_abs / diam))
            except Exception:
                pass

    uso = [0] * len(caras)
    for elem in elementos:
        for cid in elem: uso[cid] += 1
    n_huerfanas    = sum(1 for u in uso if u == 0)
    n_sobre_usadas = sum(1 for u in uso if u > 2)

    abiertos = []
    elems_dup_cara = []
    for eid, elem in enumerate(elementos):
        if len(set(elem)) != len(elem): elems_dup_cara.append(eid)
        nodos = set(); ars = set()
        for cid in elem:
            for arid in caras[cid]:
                ars.add(arid); nodos.update(aristas[arid])
        V, E, F = len(nodos), len(ars), len(elem)
        if V - E + F != 2: abiertos.append((eid, V, E, F, V - E + F))

    caras_borde_ids = [cid for cid, u in enumerate(uso) if u == 1]
    v_b = set(); e_b = set()
    for cid in caras_borde_ids:
        for arid in caras[cid]:
            e_b.add(arid); v_b.update(aristas[arid])
    chi_borde = len(v_b) - len(e_b) + len(caras_borde_ids)
    if chi_borde != 2:
        print(f"  CRITICO: borde del cubo chi={chi_borde} (esperado 2)")

    planos_cubo = [(0, 0., 'x=0'), (0, B, f'x={B}'),
                   (1, 0., 'y=0'), (1, H, f'y={H}'),
                   (2, 0., 'z=0'), (2, D, f'z={D}')]
    eps_pl = 1e-6
    for eje, val, nombre in planos_cubo:
        cids_pl = []
        for cid in caras_borde_ids:
            nodos_c = set()
            for arid in caras[cid]: nodos_c.update(aristas[arid])
            if all(abs(nodo_coords[n][eje] - val) < eps_pl for n in nodos_c):
                cids_pl.append(cid)
        v_pl, e_pl = set(), set()
        for cid in cids_pl:
            for arid in caras[cid]: e_pl.add(arid); v_pl.update(aristas[arid])
        chi_pl = len(v_pl) - len(e_pl) + len(cids_pl)
        if chi_pl != 1:
            print(f"  CRITICO: cara {nombre} chi={chi_pl} (hueco en el borde)")

    hay_error = (caras_rotas or caras_con_dup_aris or caras_no_planares
                 or n_huerfanas or n_sobre_usadas or abiertos
                 or elems_dup_cara or chi_borde != 2)

    if caras_no_planares:
        dists_rel = [r for _, _, r in caras_no_planares]
        dists_abs = [d for _, d, _ in caras_no_planares]
        print(f"  CRITICO: {len(caras_no_planares)} caras no-planares "
              f"(desv_rel max={max(dists_rel):.2e}, desv_abs max={max(dists_abs):.2e} mm)")
    if not hay_error:
        print("  OK: malla conforme y cerrada")
        return
    if caras_rotas:
        print(f"  CRITICO: {len(caras_rotas)} caras no cierran")
    if caras_con_dup_aris:
        print(f"  CRITICO: {len(caras_con_dup_aris)} caras con aristas repetidas")
    if n_huerfanas:
        print(f"  CRITICO: {n_huerfanas} caras huerfanas")
    if n_sobre_usadas:
        print(f"  CRITICO: {n_sobre_usadas} caras compartidas por >2 elementos")
    if elems_dup_cara:
        print(f"  CRITICO: {len(elems_dup_cara)} elementos con caras repetidas")
    if abiertos:
        print(f"  CRITICO: {len(abiertos)} elementos no son poliedros cerrados (V-E+F != 2)")
        for eid, V, E, F, chi in abiertos[:3]:
            print(f"           ej elem {eid}: V={V} E={E} F={F} chi={chi}")


def main():
    if semilla is None:
        s = int(time.time() * 1000) % (2**32 - 1)
    else:
        s = int(semilla)
    np.random.seed(s)
    random.seed(s)
    print(f"Semilla: {s}")

    t0 = time.time()
    n_tipos = len(tipos_agregado)

    vol_prom = float((fracciones * vols_med).sum())
    N_part   = int(round(vol_agregado_total / max(vol_prom, 1e-9)))
    N_part   = max(100, min(max_semillas, N_part))
    N_fino   = max(1000, N_part * N_cel_por_part)

    semillas_finas = np.column_stack([
        np.random.rand(N_fino) * B,
        np.random.rand(N_fino) * H,
        np.random.rand(N_fino) * D,
    ])

    print(f"\n{N_part} partículas objetivo | {N_fino} seeds Voronoi "
          f"({N_cel_por_part} celdas/partícula)")
    print("Generando Voronoi 3D...")
    celdas, semilla_celda, vor = voronoi_base(semillas_finas)
    _eps_b = 1e-9
    _limites_b = [(0, 0.0), (0, float(B)),
                  (1, 0.0), (1, float(H)),
                  (2, 0.0), (2, float(D))]
    pos_borde = set()
    for pos in range(len(celdas)):
        pts = np.asarray(celdas[pos], dtype=float)
        for eje, val in _limites_b:
            if np.sum(np.abs(pts[:, eje] - val) < _eps_b) >= 3:
                pos_borde.add(pos)
                break

    print("Asignando clusters de agregado...")
    (cascaras_por_tipo, vols_objetivo_tipo, matriz_d, clase_de_semilla,
     cluster_de_celda, forma_de_cluster,
     n_clusters_tipo, n_elong_clusters, n_aplana_clusters) = \
        asignar_clusters_voronoi(celdas, semilla_celda, vor,
                                  vol_por_clase, pos_borde)

    _vol_cel_r = vol_dominio / max(1, len(celdas))
    for k, (nombre, _, _, _) in enumerate(tipos_agregado):
        n_cl   = n_clusters_tipo[k]
        v_real = len(cascaras_por_tipo[k]) * _vol_cel_r

    n_total_clusters = sum(n_clusters_tipo)
    if n_elong_clusters > 0:
        print(f"\n  → {n_elong_clusters}/{n_total_clusters} partículas elongadas")
    if n_aplana_clusters > 0:
        print(f"  → {n_aplana_clusters}/{n_total_clusters} partículas aplanadas")

    celdas_de_cluster = {}
    for pos, cid in cluster_de_celda.items():
        celdas_de_cluster.setdefault(cid, []).append(celdas[pos])

    casco_de_cluster = {}
    for cid, lista_v in celdas_de_cluster.items():
        pts = np.vstack(lista_v)
        try:
            h = ConvexHull(pts)
            casco_de_cluster[cid] = pts[h.vertices]
        except Exception:
            casco_de_cluster[cid] = pts

    agregados_por_tipo = []
    for k, casc in enumerate(cascaras_por_tipo):
        tipo_k = {}
        vistos = set()
        for pos in casc:
            cid = cluster_de_celda[pos]
            if cid not in vistos:
                tipo_k[pos] = casco_de_cluster[cid]
                vistos.add(cid)
        agregados_por_tipo.append(tipo_k)

    print(f"\nTiempo: {time.time() - t0:.1f} s")

    resumen(cascaras_por_tipo, list(matriz_d.values()))

    verts_matriz = list(matriz_d.values())
    for casc in cascaras_por_tipo:
        verts_matriz.extend(casc.values())
    for pos in pos_borde:
        verts_matriz.append(celdas[pos])

    agregados_grafico = [(k + 1, list(agg.values()))
                   for k, agg in enumerate(agregados_por_tipo)]

    vertices_fem, clases_fem, semillas_fem = [], [], []
    for pos, s_idx in enumerate(semilla_celda):
        vertices_fem.append(celdas[pos])
        clases_fem.append(clase_de_semilla.get(s_idx, 0))
        semillas_fem.append(s_idx)

    # Fracción real: usando volumen de celda uniforme para evitar N_fino llamadas a ConvexHull.
    # Voronoi particiona exactamente el dominio → vol_total ≈ vol_dominio.
    vol_cel_aprox = vol_dominio / max(1, len(vertices_fem))
    n_agg_cells   = sum(1 for c in clases_fem if c != 0)
    vol_agregado_fem = n_agg_cells * vol_cel_aprox
    vol_total_fem    = vol_dominio
    fraccion_real = vol_agregado_fem / vol_total_fem
    print(f"\nFracción real de agregado en la malla VEM:")
    print(f"  Objetivo   : {100 * frac_agregado:.2f}%  ({vol_agregado_total:.1f} mm3)")
    print(f"  Real : {100 * fraccion_real:.2f}%  ({vol_agregado_fem:.1f} mm3)")

    generar_malla_fem(vertices_fem, clases_fem, semillas_fem, vor)

    if _EXEC is not None:
        _EXEC.shutdown(wait=False)

    #graficar_3d(verts_matriz, agregados_grafico)


if __name__ == "__main__":
    main()
