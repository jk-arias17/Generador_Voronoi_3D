# Generador de mezcla de concreto con Voronoi 3D

import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import Voronoi, ConvexHull
import math
import os
import random
import time


CARPETA = os.path.dirname(os.path.abspath(__file__))


# ==============================================================
# Parámetros a modificar
# ==============================================================

# Dimensiones del dominio [mm]
B = 150
H = 150
D = 200

# Fracción volumétrica de agregados sobre el total del dominio
frac_agregado = 0.5

# Tipos de agregado: (nombre, d_min [mm], d_max [mm], fracción del total de agregado)
# Las fracciones deben sumar 1.0. Se pueden agregar o quitar filas libremente.
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
porc_elongadas = 0.3

# Aplanamiento: factor < 1, fracción de partículas afectadas (0.0 a 1.0)
aplanamiento   = 0.8
porc_aplanadas = 0.3

# Tolerancia de volumen en asignación
tol = 0.02

n_matriz_obj          = 1000000
sep_agregados         = 0.4
intentos_siembra      = 80
factor_excluir_matriz = 0.8
factor_sobresiembra   = 1.25
n_iter_pesos          = 15
damping_fit           = 0.8
marg_borde_factor     = 1.4
max_rondas_insercion  = 6
usar_paralelo = True
n_procesos    = 0

# índice 0 = matriz, 1..N = tipos de agregado
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


vol_dominio        = B * H * D
vol_agregado_total = frac_agregado * vol_dominio

fracciones    = np.array([t[3] for t in tipos_agregado])
vols_min      = np.array([vol_esfera(t[1]) for t in tipos_agregado])
vols_max      = np.array([vol_esfera(t[2]) for t in tipos_agregado])
vols_med      = 0.5 * (vols_min + vols_max)
vol_por_clase = fracciones * vol_agregado_total


# ==============================================================
# Utilidades geométricas
# ==============================================================

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
            a = p[k]; b = p[(k + 1) % m]
            da = float(normal @ a) - off
            db = float(normal @ b) - off
            adentro_a = da >= -1e-9
            adentro_b = db >= -1e-9
            if adentro_a:
                out.append(a)
                if not adentro_b:
                    t = da / (da - db); out.append(a + t * (b - a))
            else:
                if adentro_b:
                    t = da / (da - db); out.append(a + t * (b - a))
        p = out
    return p


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
    u = u - (u @ n) * n; u /= np.linalg.norm(u)
    v = np.cross(n, u)
    diferencias = verts - centroide
    angulos = np.arctan2(diferencias @ v, diferencias @ u)
    return verts[np.argsort(angulos)]


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
    u = u - (u @ n) * n; u /= np.linalg.norm(u)
    v = np.cross(n, u)
    diferencias = coords - centroide
    angulos = np.arctan2(diferencias @ v, diferencias @ u)
    return list(np.argsort(angulos))


def _dir_aleatoria():
    theta = random.uniform(0.0, 2.0 * math.pi)
    phi   = math.acos(random.uniform(-1.0, 1.0))
    return np.array([math.sin(phi) * math.cos(theta),
                     math.sin(phi) * math.sin(theta),
                     math.cos(phi)])


import concurrent.futures as _cf

_EXEC = None

def _get_exec():
    global _EXEC
    if _EXEC is None:
        nproc = n_procesos if n_procesos and n_procesos > 0 else (os.cpu_count() or 1)
        _EXEC = _cf.ProcessPoolExecutor(max_workers=max(1, nproc))
    return _EXEC


def _vol_de_puntos(pts):
    try:
        return float(ConvexHull(np.asarray(pts, dtype=float)).volume)
    except Exception:
        return 0.0


def _vols_paralelo(lista_pts):
    if not usar_paralelo or len(lista_pts) < 1500:
        return [_vol_de_puntos(p) for p in lista_pts]
    try:
        ex = _get_exec()
        return list(ex.map(_vol_de_puntos, lista_pts, chunksize=128))
    except Exception:
        return [_vol_de_puntos(p) for p in lista_pts]


class _VorPotencia:
    __slots__ = ("vertices", "ridge_points", "ridge_vertices")

    def __init__(self, vertices, ridge_points, ridge_vertices):
        self.vertices = vertices
        self.ridge_points = ridge_points
        self.ridge_vertices = ridge_vertices


def power_diagram(sites, weights):
    sites   = np.asarray(sites, dtype=float)
    weights = np.asarray(weights, dtype=float)
    N = len(sites)
    g = (sites ** 2).sum(axis=1) - weights
    lifted = np.column_stack([sites, g])
    hull = ConvexHull(lifted)
    simplices = hull.simplices[hull.equations[:, 3] < -1e-12]
    M = len(simplices)

    a = simplices[:, 0]
    b = simplices[:, 1:]
    A   = 2.0 * (sites[b] - sites[a][:, None, :])
    rhs = g[b] - g[a][:, None]
    vertices = np.full((M, 3), np.nan)
    try:
        vertices = np.linalg.solve(A, rhs[:, :, None])[:, :, 0]
    except np.linalg.LinAlgError:
        for m in range(M):
            try:
                vertices[m] = np.linalg.solve(A[m], rhs[m])
            except np.linalg.LinAlgError:
                pass

    regions = [[] for _ in range(N)]
    for m in range(M):
        s0, s1, s2, s3 = simplices[m]
        regions[s0].append(m); regions[s1].append(m)
        regions[s2].append(m); regions[s3].append(m)
    return vertices, regions, simplices


def construir_adaptador_vor(vertices, simplices, n_real):
    from collections import defaultdict
    ridge = defaultdict(list)
    sm = np.asarray(simplices)
    for m in range(len(sm)):
        tet = sm[m]
        for ii in range(4):
            pi = int(tet[ii])
            for jj in range(ii + 1, 4):
                pj = int(tet[jj])
                p, q = (pi, pj) if pi < pj else (pj, pi)
                if q < n_real:
                    ridge[(p, q)].append(m)
    return _VorPotencia(np.asarray(vertices, dtype=float),
                        list(ridge.keys()), list(ridge.values()))


def _fantasmas():
    cx, cy, cz = B / 2.0, H / 2.0, D / 2.0
    L = max(B, H, D) * 10.0
    return np.array([[cx + dx * L, cy + dy * L, cz + dz * L]
                     for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
                     if not (dx == 0 and dy == 0 and dz == 0)])


def radio_eq(v):
    return (3.0 * v / (4.0 * math.pi)) ** (1.0 / 3.0)


def sembrar_agregados(rng):
    specs = []
    for k, (_, d_min, d_max, _) in enumerate(tipos_agregado):
        v_min = vol_esfera(d_min); v_max = vol_esfera(d_max)
        v_med = 0.5 * (v_min + v_max)
        n_part = max(1, int(round(factor_sobresiembra * vol_por_clase[k] / v_med)))
        for _ in range(n_part):
            v_obj = rng.uniform(v_min, v_max)
            specs.append((k, v_obj, radio_eq(v_obj)))
    specs.sort(key=lambda x: -x[1])

    sitios = []; pesos = []; clases = []; radios = []; vobjs = []
    placed = np.empty((0, 3)); placed_r = np.empty((0,))
    vol_colocado = [0.0] * len(tipos_agregado)
    for k, v_obj, r in specs:
        if vol_colocado[k] >= vol_por_clase[k]:
            continue
        m = r * marg_borde_factor
        ok = False
        for _ in range(intentos_siembra):
            p = np.array([rng.uniform(m, B - m),
                          rng.uniform(m, H - m),
                          rng.uniform(m, D - m)])
            if len(placed) == 0:
                ok = True; break
            d = np.linalg.norm(placed - p, axis=1)
            if np.all(d > placed_r + r + sep_agregados):
                ok = True; break
        if ok:
            sitios.append(p); pesos.append(r * r); clases.append(k)
            radios.append(r); vobjs.append(v_obj)
            vol_colocado[k] += (4.0 / 3.0) * math.pi * r ** 3
            placed = np.vstack([placed, p]); placed_r = np.append(placed_r, r)
    return (np.array(sitios, dtype=float), np.array(pesos), np.array(clases),
            np.array(radios), np.array(vobjs))


def sembrar_matriz(rng, agg_sites, agg_r):
    """Matriz densa uniforme; descarta semillas demasiado cerca de un agregado."""
    mat = rng.random((n_matriz_obj, 3)) * np.array([B, H, D])
    if len(agg_sites):
        from scipy.spatial import cKDTree
        tree = cKDTree(agg_sites)
        d, idx = tree.query(mat, k=1)
        mat = mat[d > agg_r[idx] * factor_excluir_matriz]
    return mat


def ajustar_pesos(sites, weights, n_agg, v_target, n_real):
    s = (vol_dominio / max(n_real, 1)) ** (1.0 / 3.0)
    r_floor = 0.25 * s
    dw_max  = 0.5 * s * s
    w_min   = -(1.5 * s) ** 2
    w_max   = (float(np.max(np.sqrt(np.maximum(weights[:n_agg], 0.0)))) + 2.0 * s) ** 2

    for it in range(n_iter_pesos):
        vertices, regions, _ = power_diagram(sites, weights)
        nubes = [vertices[regions[i]] if regions[i] else np.empty((0, 3))
                 for i in range(n_agg)]
        vols = np.array(_vols_paralelo(nubes))
        err = v_target - vols
        r_cur = np.maximum((3.0 * np.maximum(vols, 1e-9) / (4.0 * math.pi)) ** (1.0 / 3.0), r_floor)
        dw = damping_fit * err / (math.pi * r_cur)
        dw = np.clip(dw, -dw_max, dw_max)
        weights[:n_agg] = np.clip(weights[:n_agg] + dw, w_min, w_max)
        frac = float(vols.sum()) / vol_dominio
        print(f"    iteracion {it}: fraccion {100*frac:.1f}%")
    return weights


def _dentro_caja(pts, eps=1e-9):
    return (pts[:, 0].min() >= -eps and pts[:, 0].max() <= B + eps and
            pts[:, 1].min() >= -eps and pts[:, 1].max() <= H + eps and
            pts[:, 2].min() >= -eps and pts[:, 2].max() <= D + eps)


def construir_celdas(vertices, regions, n_real):
    celdas = []; semilla_celda = []; toca_borde = []
    for i in range(n_real):
        idx = regions[i]
        if not idx:
            continue
        raw = vertices[idx]
        raw = raw[~np.isnan(raw).any(axis=1)]
        if len(raw) < 4:
            continue
        if _dentro_caja(raw):
            cell = raw; borde = False
        else:
            cell = recortar_caja(raw)
            if cell is None or len(cell) < 4:
                continue
            borde = True
        celdas.append(np.asarray(cell, dtype=float))
        semilla_celda.append(i)
        toca_borde.append(borde)
    return celdas, semilla_celda, toca_borde


def separar_por_insercion(sites, weights, n_agg, n_mat):
    for ronda in range(max_rondas_insercion):
        vertices, regions, simplices = power_diagram(sites, weights)
        n_real = n_agg + n_mat
        vor = construir_adaptador_vor(vertices, simplices, n_real)

        nuevas = []; nuevas_w = []
        for (i, j), ms in zip(vor.ridge_points, vor.ridge_vertices):
            if i < n_agg and j < n_agg:
                fp = vertices[ms]
                fp = fp[~np.isnan(fp).any(axis=1)]
                if len(fp) < 1:
                    continue
                p = fp.mean(axis=0)
                val = float(np.dot(p - sites[i], p - sites[i]) - weights[i])
                s_loc = (vol_dominio / n_real) ** (1.0 / 3.0)
                p = p + np.random.normal(0.0, 1e-4, size=3)
                nuevas.append(p)
                nuevas_w.append(-val + (0.7 * s_loc) ** 2)

        if not nuevas:
            print(f"  inserción de matriz: 0 contactos en la ronda {ronda} -> separados")
            return sites, weights, n_mat, vertices, regions, simplices, vor

        print(f"  inserción de matriz: ronda {ronda}, {len(nuevas)} contactos -> "
              f"se insertan {len(nuevas)} celdas de matriz")
        agg_s = sites[:n_agg]; mat_s = sites[n_agg:n_real]; gh_s = sites[n_real:]
        agg_wv = weights[:n_agg]; mat_wv = weights[n_agg:n_real]; gh_wv = weights[n_real:]
        sites = np.vstack([agg_s, mat_s, np.array(nuevas), gh_s])
        weights = np.concatenate([agg_wv, mat_wv, np.array(nuevas_w), gh_wv])
        n_mat += len(nuevas)

    vertices, regions, simplices = power_diagram(sites, weights)
    vor = construir_adaptador_vor(vertices, simplices, n_agg + n_mat)
    return sites, weights, n_mat, vertices, regions, simplices, vor





def _transformar(d, eje, factor):
    u = np.asarray(eje, dtype=float); u = u / np.linalg.norm(u)
    proy = np.outer(d @ u, u)
    perp = d - proy
    return proy * factor + perp / math.sqrt(factor)


def _dir_aleatoria_rng(rng):
    theta = rng.uniform(0.0, 2.0 * math.pi)
    phi   = math.acos(rng.uniform(-1.0, 1.0))
    return np.array([math.sin(phi) * math.cos(theta),
                     math.sin(phi) * math.sin(theta),
                     math.cos(phi)])


# ==============================================================
# Visualización
# ==============================================================

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
            caras_mat, alpha=alpha_mat,
            facecolor=obtener_color(0),
            edgecolor=(0.45, 0.45, 0.45), linewidth=0.05))
    for clase_id, caras in caras_clase:
        if caras:
            ax.add_collection3d(Poly3DCollection(
                caras, alpha=alpha_agg,
                facecolor=obtener_color(clase_id),
                edgecolor=(0.10, 0.10, 0.10), linewidth=0.08))
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


def graficar_3d(verts_matriz, agregados_grafico):
    from matplotlib.patches import Patch
    leyenda = [Patch(facecolor=obtener_color(k + 1), label=tipos_agregado[k][0])
               for k in range(len(tipos_agregado))]
    leyenda.append(Patch(facecolor=obtener_color(0), label='Matriz'))

    caras_mat = []
    for v in verts_matriz:
        caras_mat.extend(caras_triangulares(v))

    caras_clase = []
    for clase_id, verts_list in agregados_grafico:
        cs = [t for v in verts_list for t in caras_triangulares(v)]
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


def resumen(celdas, clase):
    n_tipos   = len(tipos_agregado)
    vols_tipo = [0.0] * n_tipos
    n_tipo    = [0]   * n_tipos

    for pos in range(len(celdas)):
        k = clase[pos]
        if k > 0:
            vols_tipo[k - 1] += volumen_celda(celdas[pos])
            n_tipo[k - 1]    += 1

    Vt = sum(vols_tipo)
    Vm = vol_dominio - Vt

    print("\nResumen:")
    print(f"Volumen dominio    = {vol_dominio:12.1f} mm3")
    print(f"Objetivo agregados = {vol_agregado_total:12.1f} mm3  "
          f"({100*vol_agregado_total/vol_dominio:.1f}%)")
    print(f"Agregados logrados = {Vt:12.1f} mm3  ({100*Vt/vol_dominio:.1f}%)")
    print(f"Matriz             = {Vm:12.1f} mm3  ({100*Vm/vol_dominio:.1f}%)")
    if Vt > 0:
        for k, (nombre, _, _, _) in enumerate(tipos_agregado):
            V = vols_tipo[k]; n = n_tipo[k]
            obj = 100 * vol_por_clase[k] / vol_dominio
            print(f'  {nombre:6s}  {n:5d} part.  {V:10.1f} mm3  '
                  f'(real {100*V/vol_dominio:.1f}% | objetivo {obj:.1f}%)')


# ==============================================================
# Exportación FEM (malla conforme)
# ==============================================================

def caras_del_casco(hull):
    def canonico(eq):
        n, d = eq[:3].copy(), float(eq[3])
        for i in range(3):
            if abs(n[i]) > 1e-8:
                if n[i] < 0: n, d = -n, -d
                break
        return tuple(np.round(np.append(n, d), 5))

    grupos = {}
    for i, eq in enumerate(hull.equations):
        grupos.setdefault(canonico(eq), []).append(i)

    caras = []
    for ids in grupos.values():
        tris   = [hull.simplices[i] for i in ids]
        cuenta = {}
        for tri in tris:
            for j in range(3):
                a, b = int(tri[j]), int(tri[(j + 1) % 3])
                k    = (min(a, b), max(a, b))
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
                    if v == prev: continue
                    e = (min(curr, v), max(curr, v))
                    if e in aristas_pend: sig = v; break
                if sig is None: break
                e = (min(curr, sig), max(curr, sig))
                aristas_pend.discard(e)
                if sig == inicio: break
                poligono.append(sig)
                prev, curr = curr, sig
            if len(poligono) >= 3:
                caras.append(poligono)
    return caras


def corregir_uniones_t(nodo_coords, aristas, arista_id, caras):
    eps = 1e-11
    plano_nodos = {}
    for nid, c in enumerate(nodo_coords):
        for eje, dim in enumerate([B, H, D]):
            for valor in (0.0, dim):
                if abs(c[eje] - valor) < eps:
                    plano_nodos.setdefault((eje, valor), []).append(nid)

    cortes_arista = {}
    for eid in range(len(aristas)):
        a, b = aristas[eid]
        ca = nodo_coords[a]; cb = nodo_coords[b]
        plano_comun = None
        for eje, dim in enumerate([B, H, D]):
            for valor in (0.0, dim):
                if abs(ca[eje] - valor) < eps and abs(cb[eje] - valor) < eps:
                    plano_comun = (eje, valor); break
            if plano_comun is not None: break
        if plano_comun is None:
            continue
        pa = np.asarray(ca, dtype=float); pb = np.asarray(cb, dtype=float)
        ab = pb - pa; ab_len2 = float(ab @ ab)
        if ab_len2 < 1e-12:
            continue
        tol_dist = 1e-10; tol_t = 1e-6
        for nid in plano_nodos.get(plano_comun, []):
            if nid == a or nid == b: continue
            pn = np.asarray(nodo_coords[nid], dtype=float)
            t  = float((pn - pa) @ ab) / ab_len2
            if tol_t < t < 1 - tol_t:
                proy = pa + t * ab
                if np.linalg.norm(pn - proy) < tol_dist:
                    nodo_coords[nid] = proy
                    cortes_arista.setdefault(eid, []).append((t, nid))

    if not cortes_arista:
        return 0

    cadena_cortes = {}
    for eid, t_nodos in cortes_arista.items():
        t_nodos.sort(key=lambda x: x[0])
        a, b = aristas[eid]
        cadena_nodos = [a] + [nid for _, nid in t_nodos] + [b]
        nuevos_ids = []
        for i in range(len(cadena_nodos) - 1):
            x, y = cadena_nodos[i], cadena_nodos[i + 1]
            ka = frozenset({x, y})
            if ka not in arista_id:
                arista_id[ka] = len(aristas)
                aristas.append((min(x, y), max(x, y)))
            nuevos_ids.append(arista_id[ka])
        cadena_cortes[eid] = nuevos_ids

    for fid in range(len(caras)):
        ars = caras[fid]
        if not any(eid in cadena_cortes for eid in ars):
            continue
        nueva  = []
        n_ars  = len(ars)
        for k, eid in enumerate(ars):
            if eid not in cadena_cortes:
                nueva.append(eid); continue
            a_orig, b_orig = aristas[eid]
            chain = cadena_cortes[eid]
            next_eid = ars[(k + 1) % n_ars]
            n_a, n_b = aristas[next_eid]
            if b_orig == n_a or b_orig == n_b:
                nueva.extend(chain)
            elif a_orig == n_a or a_orig == n_b:
                nueva.extend(chain[::-1])
            else:
                prev_eid = ars[(k - 1) % n_ars]
                p_a, p_b = aristas[prev_eid]
                if a_orig == p_a or a_orig == p_b:
                    nueva.extend(chain)
                else:
                    nueva.extend(chain[::-1])
        caras[fid] = nueva

    return len(cortes_arista)


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


def _corregir_t_borde(nodo_coords, faces):
    eps = 1e-9
    planos = [(0, 0.0), (0, float(B)), (1, 0.0), (1, float(H)), (2, 0.0), (2, float(D))]
    nodos_plano = {pl: [] for pl in planos}
    for nid, c in enumerate(nodo_coords):
        for (eje, val) in planos:
            if abs(c[eje] - val) < eps:
                nodos_plano[(eje, val)].append(nid)

    for fid in range(len(faces)):
        loop = faces[fid]
        if len(loop) < 3:
            continue
        pl = None
        for (eje, val) in planos:
            if all(abs(nodo_coords[n][eje] - val) < eps for n in loop):
                pl = (eje, val); break
        if pl is None:
            continue
        candidatos = nodos_plano[pl]
        nuevo = []
        m = len(loop)
        for k in range(m):
            a = loop[k]; b = loop[(k + 1) % m]
            nuevo.append(a)
            pa = np.asarray(nodo_coords[a], float); pb = np.asarray(nodo_coords[b], float)
            ab = pb - pa; L2 = float(ab @ ab)
            if L2 < 1e-18:
                continue
            inserta = []
            for nid in candidatos:
                if nid == a or nid == b:
                    continue
                pn = np.asarray(nodo_coords[nid], float)
                t = float((pn - pa) @ ab) / L2
                if 1e-6 < t < 1 - 1e-6 and np.linalg.norm(pn - (pa + t * ab)) < 1e-7:
                    inserta.append((t, nid))
            inserta.sort()
            for _, nid in inserta:
                nuevo.append(nid)
        faces[fid] = _dedup_loop(nuevo)


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
            loop = faces_fin[fid]; m = len(loop)
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


def generar_malla_fem(celdas, clases_de_celda, semillas_de_celda, vor, deform_specs=None):
    if deform_specs is None:
        deform_specs = {}
    REDONDEO = 8
    eps_b = 1e-9
    seed_to_pos = {s: pos for pos, s in enumerate(semillas_de_celda)}
    semillas_validas = set(semillas_de_celda)

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

    # Caras internas
    for r_idx, vert_idxs in enumerate(vor.ridge_vertices):
        s1, s2 = int(vor.ridge_points[r_idx][0]), int(vor.ridge_points[r_idx][1])
        if s1 not in semillas_validas or s2 not in semillas_validas:
            continue
        if -1 in vert_idxs:
            continue
        p1 = seed_to_pos.get(s1); p2 = seed_to_pos.get(s2)
        if p1 is None or p2 is None:
            continue
        arista_vor = vor.vertices[vert_idxs]
        arista_vor = arista_vor[~np.isnan(arista_vor).any(axis=1)]
        if len(arista_vor) < 3:
            continue
        loop = recortar_poligono_caja(ordenar_poligono_planar(arista_vor))
        if len(loop) < 3:
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

    # Elongación / aplanamiento moviendo nodos compartidos
    nodos_movidos = set()
    for pos, (factor, axis) in deform_specs.items():
        fids = elem_faces.get(pos, [])
        node_ids = list({n for fid in fids for n in faces[fid]})
        if len(node_ids) < 4:
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
        nuevos = c + d * f_vol
        for n, nc in zip(node_ids, nuevos):
            nodo_coords[n] = nc
            nodos_movidos.add(n)

    # Triangular caras tocadas; las demás quedan como polígono
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

    n_tapados = _cerrar_huecos(faces_fin, owners_fin, nodo_coords)
    if n_tapados:
        print(f"  (se cerraron {n_tapados} huecos residuales por degeneración)")

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
        m = len(loop)
        eids = [id_arista(loop[k], loop[(k + 1) % m]) for k in range(m)]
        caras.append(eids)
        for pos in owners_fin[fid]:
            elem_faces_fin.setdefault(pos, []).append(fid)

    elementos = []
    elem_clase = []
    for pos in range(len(celdas)):
        fids = elem_faces_fin.get(pos)
        if not fids:
            continue
        elementos.append(list(dict.fromkeys(fids)))
        elem_clase.append(clases_de_celda[pos])

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


# ==============================================================
# Main
# ==============================================================

def main():
    if semilla is None:
        s = int(time.time() * 1000) % (2**32 - 1)
    else:
        s = int(semilla)
    rng = np.random.default_rng(s)
    np.random.seed(s)
    random.seed(s)
    print(f"Semilla: {s}")

    t0 = time.time()

    print("\nSembrando partículas...")
    agg_sites, agg_w, agg_k, agg_r, agg_vobj = sembrar_agregados(rng)
    n_agg = len(agg_sites)
    mat = sembrar_matriz(rng, agg_sites, agg_r)
    n_mat = len(mat)
    ghosts = _fantasmas()
    if n_agg:
        sites = np.vstack([agg_sites, mat, ghosts])
    else:
        sites = np.vstack([mat, ghosts])
    weights = np.concatenate([agg_w, np.zeros(n_mat), np.zeros(len(ghosts))])
    n_real = n_agg + n_mat
    sites[:n_real] += rng.normal(0.0, 1e-4, size=(n_real, 3))

    print(f"  {n_agg} partículas de agregado | {n_mat} celdas de matriz "
          f"| {n_real} celdas totales")

    print("\nAjustando pesos (Laguerre)...")
    weights = ajustar_pesos(sites, weights, n_agg, agg_vobj, n_real)

    print("\nInsertando matriz entre agregados pegados...")
    sites, weights, n_mat, vertices, regions, simplices, vor = \
        separar_por_insercion(sites, weights, n_agg, n_mat)
    n_real = n_agg + n_mat

    celdas, semilla_celda, toca_borde = construir_celdas(vertices, regions, n_real)

    clase = []
    for pos, i in enumerate(semilla_celda):
        if i < n_agg and not toca_borde[pos]:
            clase.append(int(agg_k[i]) + 1)
        else:
            clase.append(0)

    sitio_a_pos = {ss: p for p, ss in enumerate(semilla_celda)}
    vecinos_pos = {}
    for (s1, s2) in vor.ridge_points:
        p1 = sitio_a_pos.get(s1); p2 = sitio_a_pos.get(s2)
        if p1 is None or p2 is None:
            continue
        vecinos_pos.setdefault(p1, set()).add(p2)
        vecinos_pos.setdefault(p2, set()).add(p1)

    agg_posiciones = [p for p in range(len(clase)) if clase[p] > 0]
    agg_posiciones.sort(key=lambda p: -agg_vobj[semilla_celda[p]])
    aceptados = set()
    n_degradados = 0
    for p in agg_posiciones:
        vecino_agg = any(q in aceptados for q in vecinos_pos.get(p, ()))
        if vecino_agg:
            clase[p] = 0
            n_degradados += 1
        else:
            aceptados.add(p)
    if n_degradados:
        print(f"  {n_degradados} agregados aún pegados degradados a matriz "
              f"(red de seguridad)")

    deform_specs = {}
    n_el = 0; n_ap = 0
    for pos in range(len(semilla_celda)):
        if clase[pos] == 0:
            continue
        rr = rng.random()
        eje = _dir_aleatoria_rng(rng)
        if rr < porc_elongadas:
            deform_specs[pos] = (elongacion, eje); n_el += 1
        elif rr < porc_elongadas + porc_aplanadas:
            deform_specs[pos] = (aplanamiento, eje); n_ap += 1
    print(f"\nElongación/aplanamiento: {n_el} elongadas | {n_ap} aplanadas")

    print(f"\nTiempo total (geometría): {time.time() - t0:.1f} s")

    resumen(celdas, clase)

    generar_malla_fem(list(celdas), list(clase), list(semilla_celda), vor, deform_specs)

    if _EXEC is not None:
        _EXEC.shutdown(wait=False)

    # Visualización (descomentar para ver el 3D)
    # verts_matriz = [celdas[p] for p in range(len(celdas)) if clase[p] == 0]
    # agregados_grafico = [(k + 1, [celdas[p] for p in range(len(celdas)) if clase[p] == k + 1])
    #                      for k in range(len(tipos_agregado))]
    # graficar_3d(verts_matriz, agregados_grafico)


if __name__ == "__main__":
    main()
