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
elongacion     = 1.5
porc_elongadas = 0.3

# Aplanamiento: factor < 1, fracción de partículas afectadas (0.0 a 1.0)
aplanamiento   = 0.5
porc_aplanadas = 0.3

# Tolerancia de volumen en asignación y número máximo de semillas Voronoi
tol          = 0.02
max_semillas = 10000

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



def asignar_granu(celdas, libres, vol_min, vol_max, vol_obj, vol_efectivo=None):
    vols = {i: volumen_celda(celdas[i]) for i in libres}

    candidatos = sorted([i for i in libres if vols[i] >= vol_min],
                        key=lambda i: -vols[i])

    cascara  = {}
    vols_objetivo = {}
    acumulado     = 0.0
    for i in candidatos:
        if acumulado >= vol_obj * (1.0 + tol):
            break
        v_objetivo    = random.uniform(vol_min, vol_max)
        cascara[i] = celdas[i]
        vols_objetivo[i] = v_objetivo
        v_efectivo = vol_efectivo[i] if (vol_efectivo and i in vol_efectivo) else vols[i]
        acumulado  += v_efectivo

    return cascara, set(cascara.keys()), acumulado, vols_objetivo


def asignar_fino(celdas_libres_dict, vol_min, vol_max, vol_obj, vol_efectivo=None):
    vols = {i: volumen_celda(c) for i, c in celdas_libres_dict.items()}
    orden = sorted(celdas_libres_dict.keys(), key=lambda i: -vols[i])

    cascara  = {}
    vols_objetivo = {}
    matriz   = {}
    acumulado     = 0.0

    for i in orden:
        c = celdas_libres_dict[i]
        v = vols[i]
        v_efectivo = vol_efectivo[i] if (vol_efectivo and i in vol_efectivo) else v
        if acumulado < vol_obj * (1.0 + tol) and v >= vol_min:
            v_objetivo    = random.uniform(vol_min, vol_max)
            cascara[i] = c
            vols_objetivo[i] = v_objetivo
            acumulado       += v_efectivo
        else:
            matriz[i] = c

    return cascara, matriz, acumulado, vols_objetivo



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
    vols_tipo = [sum(volumen_celda(v) for v in casc.values()) for casc in cascaras_por_tipo]
    Vt = sum(vols_tipo)
    Vm = sum(volumen_celda(v) for v in matriz_verts) if matriz_verts else 0.0

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
            n = len(cascaras_por_tipo[k])
            print(f'  {nombre:6s}  {n:5d} part.  {V:10.1f} mm3  ({100*V/Vt:.1f}%)')



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
        ca = nodo_coords[a]
        cb = nodo_coords[b]
        plano_comun = None
        for eje, dim in enumerate([B, H, D]):
            for valor in (0.0, dim):
                if abs(ca[eje] - valor) < eps and abs(cb[eje] - valor) < eps:
                    plano_comun = (eje, valor)
                    break
            if plano_comun is not None:
                break
        if plano_comun is None:
            continue

        pa = np.asarray(ca, dtype=float)
        pb = np.asarray(cb, dtype=float)
        ab = pb - pa
        ab_len2 = float(ab @ ab)
        if ab_len2 < 1e-12:
            continue

        tol_dist = 1e-10
        tol_t    = 1e-6
        for nid in plano_nodos.get(plano_comun, []):
            if nid == a or nid == b:
                continue
            pn = np.asarray(nodo_coords[nid], dtype=float)
            t = float((pn - pa) @ ab) / ab_len2
            if tol_t < t < 1 - tol_t:
                proy = pa + t * ab
                if np.linalg.norm(pn - proy) < tol_dist:
                    nodo_coords[nid] = proy  # anclar exactamente sobre la arista de borde
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

        nueva = []
        n_ars = len(ars)
        for k, eid in enumerate(ars):
            if eid not in cadena_cortes:
                nueva.append(eid)
                continue

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
                elif b_orig == p_a or b_orig == p_b:
                    nueva.extend(chain[::-1])
                else:
                    nueva.extend(chain)

        caras[fid] = nueva

    return len(cortes_arista)


def generar_malla_fem(vertices_de_celda, clases_de_celda, semillas_de_celda, vor):
    REDONDEO = 8

    nodo_id     = {}
    nodo_coords = []
    arista_id   = {}
    aristas     = []
    cara_id     = {}
    caras       = []
    elementos   = []
    elem_clase  = []

    def id_nodo(p):
        k = tuple(np.round(p, REDONDEO))
        if k not in nodo_id:
            nodo_id[k] = len(nodo_coords)
            nodo_coords.append(np.asarray(p, dtype=float))
        return nodo_id[k]

    def registrar_cara_de_poligono(poligono_3d):
        _snap_glob = [(0, 0.0), (0, float(B)),
                      (1, 0.0), (1, float(H)),
                      (2, 0.0), (2, float(D))]
        poligono_ajustado = []
        for _p in poligono_3d:
            _v = np.asarray(_p, dtype=float).copy()
            for _ej, _vl in _snap_glob:
                if abs(_v[_ej] - _vl) < 1e-9:
                    _v[_ej] = _vl
            poligono_ajustado.append(_v)
        ids_globales = [id_nodo(p) for p in poligono_ajustado]

        limpio = []
        for v in ids_globales:
            if not limpio or limpio[-1] != v:
                limpio.append(v)
        if len(limpio) > 1 and limpio[0] == limpio[-1]:
            limpio.pop()

        if len(set(limpio)) != len(limpio):
            visto = set()
            unico = []
            for v in limpio:
                if v not in visto:
                    visto.add(v)
                    unico.append(v)
            limpio = unico

        if len(limpio) < 3:
            return None

        kc = frozenset(limpio)
        if kc in cara_id:
            return cara_id[kc]

        ids_aristas = []
        n = len(limpio)
        for k in range(n):
            a, b = limpio[k], limpio[(k + 1) % n]
            ka = frozenset({a, b})
            if ka not in arista_id:
                arista_id[ka] = len(aristas)
                aristas.append((min(a, b), max(a, b)))
            ids_aristas.append(arista_id[ka])

        cara_id[kc] = len(caras)
        caras.append(ids_aristas)
        return cara_id[kc]

    semillas_validas = set(semillas_de_celda)

    semilla_caras_int = {s: [] for s in semillas_validas}

    n_aristas_vor      = 0
    n_aristas_inf   = 0
    n_aristas_fuera = 0
    n_aristas_degen = 0
    n_aristas_ok         = 0
    for r_idx, vert_idxs in enumerate(vor.ridge_vertices):
        s1, s2 = vor.ridge_points[r_idx]
        if s1 not in semillas_validas or s2 not in semillas_validas:
            continue
        n_aristas_vor += 1

        if -1 in vert_idxs:
            n_aristas_inf += 1
            continue

        arista_vor = vor.vertices[vert_idxs]
        arista_vor = ordenar_poligono_planar(arista_vor)
        recortada  = recortar_poligono_caja(arista_vor)
        if len(recortada) < 3:
            n_aristas_fuera += 1
            continue

        cid = registrar_cara_de_poligono(recortada)
        if cid is None:
            n_aristas_degen += 1
            continue

        semilla_caras_int[s1].append(cid)
        semilla_caras_int[s2].append(cid)
        n_aristas_ok += 1

    semilla_caras_borde = {}
    n_caras_borde_total = 0

    planos_caja = [
        (0, 0.0),     (0, float(B)),
        (1, 0.0),     (1, float(H)),
        (2, 0.0),     (2, float(D)),
    ]
    eps_borde = 1e-9

    for verts, seed_idx in zip(vertices_de_celda, semillas_de_celda):
        pts = np.asarray(verts, dtype=float)
        for eje, val in planos_caja:
            mascara = np.abs(pts[:, eje] - val) < eps_borde
            en_cara = pts[mascara]
            if len(en_cara) < 3:
                continue
            en_cara = en_cara.copy()
            en_cara[:, eje] = val
            poligono_ord = ordenar_poligono_planar(en_cara)
            if len(poligono_ord) < 3:
                continue
            cid = registrar_cara_de_poligono(poligono_ord)
            if cid is not None:
                semilla_caras_borde.setdefault(seed_idx, []).append(cid)
                n_caras_borde_total += 1

    for verts, clase, seed_idx in zip(vertices_de_celda, clases_de_celda, semillas_de_celda):
        ids_caras_elem = []
        caras_elem_set = set()
        for cid in semilla_caras_int.get(seed_idx, []):
            if cid not in caras_elem_set:
                caras_elem_set.add(cid)
                ids_caras_elem.append(cid)
        for cid in semilla_caras_borde.get(seed_idx, []):
            if cid not in caras_elem_set:
                caras_elem_set.add(cid)
                ids_caras_elem.append(cid)
        if ids_caras_elem:
            elementos.append(ids_caras_elem)
            elem_clase.append(clase)

    corregir_uniones_t(nodo_coords, aristas, arista_id, caras)

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
        f.write(f"0 0 0\n")  # matriz
        for k in range(len(tipos_agregado)):
            f.write(f"{k + 1} 0 0\n")

    print(f"\nMalla: {len(elementos)} elementos  {N} nodos  {len(caras)} caras")
    if n_aristas_inf:
        print(f"  AVISO: {n_aristas_inf} ridges al infinito -> caras internas faltantes")
    validar_malla(aristas, caras, elementos, nodo_coords)



def validar_malla(aristas, caras, elementos, nodo_coords):
    caras_rotas         = []
    caras_con_dup_aris  = []
    caras_no_planares   = []
    for cid, ars in enumerate(caras):
        if len(ars) < 3:
            caras_rotas.append(cid)
            continue
        if len(set(ars)) != len(ars):
            caras_con_dup_aris.append(cid)
        ok = True
        for k in range(len(ars)):
            a = aristas[ars[k]]
            b = aristas[ars[(k + 1) % len(ars)]]
            if not (set(a) & set(b)):
                ok = False
                break
        if not ok:
            caras_rotas.append(cid)
            continue

        nodos_cara = []
        for arid in ars:
            for nid in aristas[arid]:
                if nid not in nodos_cara:
                    nodos_cara.append(nid)
        if len(nodos_cara) >= 4:
            coords_c = np.array([nodo_coords[n] for n in nodos_cara])
            centroide = coords_c.mean(axis=0)
            centrada = coords_c - centroide
            try:
                _, S, Vt = np.linalg.svd(centrada, full_matrices=False)
                n_pl    = Vt[-1]
                max_abs = float(np.abs(centrada @ n_pl).max())
                diam    = 0.0
                for _i in range(len(coords_c)):
                    for _j in range(_i + 1, len(coords_c)):
                        d = float(np.linalg.norm(coords_c[_i] - coords_c[_j]))
                        if d > diam:
                            diam = d
                if diam > 1e-10:
                    desv_rel = max_abs / diam
                    if desv_rel > 1e-9:
                        caras_no_planares.append((cid, max_abs, desv_rel))
            except Exception:
                pass

    uso = [0] * len(caras)
    for elem in elementos:
        for cid in elem:
            uso[cid] += 1
    n_huerfanas    = sum(1 for u in uso if u == 0)
    n_sobre_usadas = sum(1 for u in uso if u > 2)
    n_borde        = sum(1 for u in uso if u == 1)
    n_internas     = sum(1 for u in uso if u == 2)

    abiertos       = []
    elems_dup_cara = []
    for eid, elem in enumerate(elementos):
        if len(set(elem)) != len(elem):
            elems_dup_cara.append(eid)
        nodos = set()
        ars   = set()
        for cid in elem:
            for arid in caras[cid]:
                ars.add(arid)
                nodos.update(aristas[arid])
        V, E, F = len(nodos), len(ars), len(elem)
        if V - E + F != 2:
            abiertos.append((eid, V, E, F, V - E + F))

    caras_borde_ids = [cid for cid, u in enumerate(uso) if u == 1]
    v_borde = set()
    e_borde = set()
    for cid in caras_borde_ids:
        for arid in caras[cid]:
            e_borde.add(arid)
            v_borde.update(aristas[arid])
    chi_borde = len(v_borde) - len(e_borde) + len(caras_borde_ids)
    if chi_borde != 2:
        print(f"  CRITICO: borde del cubo chi={chi_borde} (esperado 2)")

    planos_cubo = [
        (0,  0.0, 'x=0'),  (0,  B, f'x={B}'),
        (1,  0.0, 'y=0'),  (1,  H, f'y={H}'),
        (2,  0.0, 'z=0'),  (2,  D, f'z={D}'),
    ]
    eps_pl = 1e-6
    for eje, val, nombre in planos_cubo:
        cids_pl = []
        for cid in caras_borde_ids:
            nodos_c = set()
            for arid in caras[cid]:
                nodos_c.update(aristas[arid])
            coords_eje = [nodo_coords[n][eje] for n in nodos_c]
            if all(abs(co - val) < eps_pl for co in coords_eje):
                cids_pl.append(cid)
        v_pl, e_pl = set(), set()
        for cid in cids_pl:
            for arid in caras[cid]:
                e_pl.add(arid)
                v_pl.update(aristas[arid])
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
              f"(desv_rel max={max(dists_rel):.2e}, "
              f"desv_abs max={max(dists_abs):.2e} mm) "
              f"-> VEM fallara patch test")
        planos_caja_diag = [(0,0.,f'x=0'),(0,float(B),f'x={B}'),
                            (1,0.,f'y=0'),(1,float(H),f'y={H}'),
                            (2,0.,f'z=0'),(2,float(D),f'z={D}')]
        peores = sorted(caras_no_planares, key=lambda t: -t[2])[:5]
        for cid, abs_mm, rel in peores:
            ars = caras[cid]
            nodos_c = []
            for arid in ars:
                for nid in aristas[arid]:
                    if nid not in nodos_c:
                        nodos_c.append(nid)
            coords_c = np.array([nodo_coords[n] for n in nodos_c])
            centroide = coords_c.mean(axis=0)
            centrada = coords_c - centroide
            try:
                _, S, Vt = np.linalg.svd(centrada, full_matrices=False)
                n_pl = Vt[-1]
                dists_v = np.abs(centrada @ n_pl)
                peor_idx = int(np.argmax(dists_v))
            except Exception:
                peor_idx = 0
            tipo = 'interna'
            for eje, val, nom in planos_caja_diag:
                if all(abs(coords_c[j,eje] - val) < 1e-6 for j in range(len(coords_c))):
                    tipo = f'borde {nom}'
                    break
            uso_c = sum(1 for elem in elementos if cid in elem)
            print(f"    cara {cid}: desv_rel={rel:.2e} desv_abs={abs_mm:.2e}mm "
                  f"tipo={tipo} uso={uso_c} nverts={len(nodos_c)}")
            print(f"      vertice peor: {coords_c[peor_idx]}")
            print(f"      centroide:    {centroide}")
            print(f"      normal SVD:   {Vt[-1] if 'Vt' in dir() else 'n/a'}")
            print(f"      coords de todos los vertices:")
            for j, coord in enumerate(coords_c):
                marca = ' <-- PEOR' if j == peor_idx else ''
                print(f"        [{j}] {coord}{marca}")

    if not hay_error:
        print("  OK: malla conforme y cerrada")
        return

    if caras_rotas:
        print(f"  CRITICO: {len(caras_rotas)} caras no cierran "
              f"(aristas no forman ciclo)")
    if caras_con_dup_aris:
        print(f"  CRITICO: {len(caras_con_dup_aris)} caras con aristas "
              f"repetidas dentro de la misma cara")
    if n_huerfanas:
        print(f"  CRITICO: {n_huerfanas} caras huerfanas (no las usa nadie)")
    if n_sobre_usadas:
        print(f"  CRITICO: {n_sobre_usadas} caras compartidas por >2 elementos "
              f"(geometricamente imposible)")
    if elems_dup_cara:
        print(f"  CRITICO: {len(elems_dup_cara)} elementos con caras repetidas "
              f"en su lista")
    if abiertos:
        print(f"  CRITICO: {len(abiertos)} elementos no son poliedros cerrados "
              f"(V-E+F != 2)")
        print(f"           -> tipicamente nodos colgantes: la celda vecina")
        print(f"              mete un vertice extra en la cara compartida")
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
    N_base   = int(round(vol_agregado_total / max(vol_prom, 1e-9)))
    N_base   = max(400, min(max_semillas, N_base))
    r_est    = (3.0 * vol_dominio / (4.0 * math.pi * N_base)) ** (1.0 / 3.0)

    semillas_base = np.column_stack([
        np.random.rand(N_base) * B,
        np.random.rand(N_base) * H,
        np.random.rand(N_base) * D,
    ])

    n_elongadas_pre  = int(round(N_base * max(0.0, min(1.0, porc_elongadas))))
    n_aplanadas_pre = int(round(N_base * max(0.0, min(1.0, porc_aplanadas))))
    semillas_elongadas  = set(random.sample(range(N_base), min(n_elongadas_pre,  N_base)))
    semillas_aplanadas = set(random.sample(range(N_base), min(n_aplanadas_pre, N_base))) - semillas_elongadas

    grupo_de_semilla    = {i: i for i in range(N_base)}
    info_deform_grupo = {}
    semillas_extras           = []

    d_elong = r_est * 0.9          # separación entre sub-semillas en elongación
    d_aplana = r_est * 0.9         # radio de los satélites en aplanamiento

    for i in range(N_base):
        if i in semillas_elongadas:
            u = _dir_aleatoria()
            info_deform_grupo[i] = ('elong', u, elongacion)
            n_sub = max(2, round(elongacion))   # nº total de semillas en el grupo
            # posiciones: centradas en semillas_base[i], separadas d_elong
            offsets = [(k - (n_sub - 1) / 2.0) * d_elong for k in range(n_sub)]
            for t in offsets:
                if abs(t) < 1e-9:
                    continue   # la semilla central ya existe (índice i)
                pos = np.clip(semillas_base[i] + t * u,
                              [0.0, 0.0, 0.0], [float(B), float(H), float(D)])
                abs_idx = N_base + len(semillas_extras)
                semillas_extras.append(pos)
                grupo_de_semilla[abs_idx] = i

        elif i in semillas_aplanadas:
            u = _dir_aleatoria()
            info_deform_grupo[i] = ('aplana', u, aplanamiento)
            p1, p2 = _perp_a(u)
            # 4 satélites en el plano perpendicular a u → forma de disco
            for dv in [p1, -p1, p2, -p2]:
                pos = np.clip(semillas_base[i] + d_aplana * dv,
                              [0.0, 0.0, 0.0], [float(B), float(H), float(D)])
                abs_idx = N_base + len(semillas_extras)
                semillas_extras.append(pos)
                grupo_de_semilla[abs_idx] = i

    if semillas_extras:
        todas_semillas = np.vstack([semillas_base, np.array(semillas_extras)])
    else:
        todas_semillas = semillas_base

    n_grupos_elongados  = len(semillas_elongadas)
    n_grupos_aplanados = len(semillas_aplanadas)
    n_extras   = len(semillas_extras)
    if n_grupos_elongados > 0:
        print(f"\n{n_grupos_elongados}/{N_base} semillas elongadas "
              f"({round(elongacion)} sub-semillas c/u, factor {elongacion})")
    if n_grupos_aplanados > 0:
        print(f"{n_grupos_aplanados}/{N_base} semillas aplanadas "
              f"(5 sub-semillas c/u, factor {aplanamiento})")
    if n_extras > 0:
        print(f"  → {n_extras} semillas extra generadas antes del Voronoi")

    print("Generando Voronoi 3D...")
    celdas, semilla_celda, vor = voronoi_base(todas_semillas)

    pos_base = [pos for pos, s in enumerate(semilla_celda) if s < N_base]
    print(f"  {len(pos_base)} celdas base | {len(semilla_celda) - len(pos_base)} "
          f"celdas extra | vol. medio base ~ "
          f"{vol_dominio / max(1, len(pos_base)):.1f} mm3")

    _eps_b = 1e-9
    _limites_b = [(0, 0.0), (0, float(B)),
                  (1, 0.0), (1, float(H)),
                  (2, 0.0), (2, float(D))]
    pos_borde_base = set()
    for pos in pos_base:
        pts = np.asarray(celdas[pos], dtype=float)
        for eje, val in _limites_b:
            if np.sum(np.abs(pts[:, eje] - val) < _eps_b) >= 3:
                pos_borde_base.add(pos)
                break
    print(f"  {len(pos_borde_base)} celdas base en el borde (forzadas a matriz)")

    libres  = [pos for pos in pos_base if pos not in pos_borde_base]
    _n_sub_elong = max(2, round(elongacion))

    vol_efectivo = {}
    for pos in libres:
        s_idx = semilla_celda[pos]
        if s_idx in semillas_elongadas:
            factor_pos = _n_sub_elong
        elif s_idx in semillas_aplanadas:
            factor_pos = 5
        else:
            factor_pos = 1
        vol_efectivo[pos] = volumen_celda(celdas[pos]) * factor_pos

    cascaras_por_tipo = []
    vols_objetivo_tipo = []
    matriz_d = {}

    for k, (nombre, d_min, d_max, _) in enumerate(tipos_agregado):
        vol_min = vol_esfera(d_min)
        vol_max = vol_esfera(d_max)
        print(f'Asignando {nombre}...')
        if k < n_tipos - 1:
            casc, usados, acumulado, vols_obj_k = asignar_granu(
                celdas, libres, vol_min, vol_max, vol_por_clase[k], vol_efectivo)
            libres = [i for i in libres if i not in usados]
        else:
            libres_mapa = {i: celdas[i] for i in libres}
            casc, matriz_d, acumulado, vols_obj_k = asignar_fino(
                libres_mapa, vol_min, vol_max, vol_por_clase[k], vol_efectivo)
        cascaras_por_tipo.append(casc)
        vols_objetivo_tipo.append(vols_obj_k)
        print(f'  {len(casc)} particulas | vol. efectivo acumulado. {acumulado:.1f} mm3')

    print(f"  Matriz: {len(matriz_d)} celdas")

    clase_de_seed = {}
    for k, casc in enumerate(cascaras_por_tipo):
        for pos in casc:
            clase_de_seed[semilla_celda[pos]] = k + 1
    for pos in matriz_d:
        clase_de_seed[semilla_celda[pos]] = 0
    for pos in pos_borde_base:
        clase_de_seed[semilla_celda[pos]] = 0
    for abs_idx, g_id in grupo_de_semilla.items():
        if abs_idx >= N_base:
            clase_de_seed[abs_idx] = clase_de_seed.get(g_id, 0)

    n_extra_borde = 0
    for pos, s_idx in enumerate(semilla_celda):
        if clase_de_seed.get(s_idx, 0) != 0:
            pts = np.asarray(celdas[pos], dtype=float)
            for eje, val in _limites_b:
                if np.sum(np.abs(pts[:, eje] - val) < _eps_b) >= 3:
                    clase_de_seed[s_idx] = 0
                    n_extra_borde += 1
                    break
    if n_extra_borde:
        print(f"  → {n_extra_borde} sub-celda(s) extra en el borde forzadas a matriz")

    semillas_de_agregado = set()
    for casc in cascaras_por_tipo:
        for pos in casc:
            semillas_de_agregado.add(semilla_celda[pos])

    n_elongados_agg  = sum(1 for s in semillas_elongadas  if s in semillas_de_agregado)
    n_aplanados_agg = sum(1 for s in semillas_aplanadas if s in semillas_de_agregado)
    n_total_agregados  = len(semillas_de_agregado)
    if n_elongados_agg > 0:
        print(f"\n  → {n_elongados_agg}/{n_total_agregados} agregados elongados en la malla")
    if n_aplanados_agg > 0:
        print(f"  → {n_aplanados_agg}/{n_total_agregados} agregados aplanados en la malla")

    verts_por_grupo = {}
    for pos, s_idx in enumerate(semilla_celda):
        g_id = grupo_de_semilla.get(s_idx, s_idx)
        verts_por_grupo.setdefault(g_id, []).append(celdas[pos])

    casco_por_grupo = {}
    for g_id, lista_v in verts_por_grupo.items():
        pts = np.vstack(lista_v)
        try:
            h = ConvexHull(pts)
            casco_por_grupo[g_id] = pts[h.vertices]
        except Exception:
            casco_por_grupo[g_id] = pts

    grupos_multisemilla = {g_id for abs_idx, g_id in grupo_de_semilla.items()
                        if abs_idx >= N_base}

    agregados_por_tipo = []
    for k, (casc, vols_obj_k) in enumerate(zip(cascaras_por_tipo, vols_objetivo_tipo)):
        tipo_k = {}
        for pos in casc:
            s_idx = semilla_celda[pos]
            g_id  = grupo_de_semilla.get(s_idx, s_idx)
            if g_id in grupos_multisemilla:
                tipo_k[pos] = casco_por_grupo[g_id]
            else:
                tipo_k[pos] = escalar_celda(casc[pos], vols_obj_k[pos])
        agregados_por_tipo.append(tipo_k)

    print(f"\nTiempo: {time.time() - t0:.1f} s")

    resumen(cascaras_por_tipo, list(matriz_d.values()))

    verts_matriz = list(matriz_d.values())
    for casc in cascaras_por_tipo:
        verts_matriz.extend(casc.values())
    for pos in pos_borde_base:
        verts_matriz.append(celdas[pos])
    set_semillas_extras = {idx for idx in grupo_de_semilla if idx >= N_base}
    for pos, s_idx in enumerate(semilla_celda):
        if s_idx in set_semillas_extras:
            verts_matriz.append(celdas[pos])

    agregados_grafico = [(k + 1, list(agg.values()))
                   for k, agg in enumerate(agregados_por_tipo)]

    vertices_fem, clases_fem, semillas_fem = [], [], []
    for pos, s_idx in enumerate(semilla_celda):
        vertices_fem.append(celdas[pos])
        clases_fem.append(clase_de_seed.get(s_idx, 0))
        semillas_fem.append(s_idx)

    vols_fem = [volumen_celda(v) for v in vertices_fem]
    vol_total_fem = sum(vols_fem)
    vol_agregado_fem   = sum(v for v, c in zip(vols_fem, clases_fem) if c != 0)
    fraccion_real = vol_agregado_fem / vol_total_fem if vol_total_fem > 0 else 0.0
    print(f"\nFracción real de agregado en la malla FEM:")
    print(f"  Objetivo   : {100 * frac_agregado:.2f}%  ({vol_agregado_total:.1f} mm³)")
    print(f"  Real (FEM) : {100 * fraccion_real:.2f}%  ({vol_agregado_fem:.1f} mm³)")

    generar_malla_fem(vertices_fem, clases_fem, semillas_fem, vor)

    graficar_3d(verts_matriz, agregados_grafico)


if __name__ == "__main__":
    main()
