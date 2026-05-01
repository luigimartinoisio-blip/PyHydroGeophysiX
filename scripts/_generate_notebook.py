"""
_generate_notebook.py
Genera Coupled_2D_Hydrogeophysical_Modeling.ipynb nel progetto PyHydroGeophysiX.
Eseguire UNA VOLTA con il venv attivo:
    python _generate_notebook.py
"""
import json
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# CELLA 1 – Caricamento Dati e Coordinate Locali
# ─────────────────────────────────────────────────────────────────────────────
CELL1 = r"""
# ═══════════════════════════════════════════════════════════════════════════
# CELLA 1 – Caricamento Dati e Coordinate Locali (Data Parsing)
# Input : electr_coord.csv (GPS WGS84) + 08topog_TL.txt (profilo locale X,Z)
# Output: el_pos = [(x [m], z [m a.s.l.]), ...]  – 48 elettrodi
# ═══════════════════════════════════════════════════════════════════════════
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ── Directory di input ────────────────────────────────────────────────────
INPUT_DIR = Path(
    'C:/Users/luigi/git/github.com/luigimartinoisio-blip/PyHydroGeophysiX/Position'
)

# ── 1. Lettura 08topog_TL.txt ─────────────────────────────────────────────
# Formato: #,X,Y,Z,Depth  (la prima riga è l'header commentata con '#')
# X = distanza progressiva [m] in sistema locale (0–56.4 m, passo 1.2 m)
# Z = quota altimetrica assoluta [m a.s.l.]
topog = pd.read_csv(
    INPUT_DIR / '08topog_TL.txt',
    comment='#',
    header=None,
    names=['id', 'X_local', 'Y', 'Z_abs', 'Depth'],
    sep=',',
    skipinitialspace=True,
)
topog = topog.dropna(how='all').reset_index(drop=True)
topog = topog.astype({'X_local': float, 'Z_abs': float})

# ── 2. Lettura electr_coord.csv ───────────────────────────────────────────
# Formato: Nome punto, Latitudine, Longitudine, Altezza ellisoidica
# Ruolo  : verifica incrociata della quota (h ellissoidica ≈ Z topografica)
elec_gps = pd.read_csv(INPUT_DIR / 'electr_coord.csv', skipinitialspace=True)
elec_gps.columns = elec_gps.columns.str.strip()

# ── 3. Cross-check quote (soglia 0.5 m) ──────────────────────────────────
z_topog = topog['Z_abs'].values.astype(float)
z_gps   = elec_gps['Altezza ellisoidica'].values.astype(float)
delta_z = np.abs(z_topog - z_gps)
print('Cross-check quote: max Dz = {:.4f} m  (soglia < 0.5 m)'.format(delta_z.max()))
if delta_z.max() >= 0.5:
    raise ValueError('Incoerenza topografia/GPS: max Dz = {:.3f} m'.format(delta_z.max()))
print('OK  Quote coerenti tra i due file.')

# ── 4. Costruzione el_pos in coordinate locale 2D ─────────────────────────
# X = distanza progressiva dal primo elettrodo (Pt1 → X = 0)
# Z = quota altimetrica assoluta [m a.s.l.] → LIMITE SUPERIORE del dominio
x_profile = topog['X_local'].values.astype(float)
z_abs     = topog['Z_abs'].values.astype(float)

if len(x_profile) != 48:
    raise ValueError('Attesi 48 elettrodi, trovati {}'.format(len(x_profile)))

el_pos = list(zip(x_profile.tolist(), z_abs.tolist()))   # [(x1,z1), ..., (x48,z48)]

# ── 5. Riepilogo ──────────────────────────────────────────────────────────
spacing = float(np.diff(x_profile).mean())
rows = [
    ('Numero elettrodi',  str(len(el_pos))),
    ('Lunghezza profilo', '{:.2f} m'.format(x_profile[-1] - x_profile[0])),
    ('Spaziatura media',  '{:.2f} m'.format(spacing)),
    ('Quota min (Pt48)',  '{:.3f} m a.s.l.'.format(z_abs.min())),
    ('Quota max (Pt1)',   '{:.3f} m a.s.l.'.format(z_abs.max())),
    ('Dislivello totale', '{:.3f} m'.format(z_abs.max() - z_abs.min())),
]
print()
print('  {:<32} {}'.format('Parametro', 'Valore'))
print('  ' + '-' * 52)
for lbl, val in rows:
    print('  {:<32} {}'.format(lbl, val))
print()
print('Prime 5  :', el_pos[:5])
print('Ultime 5 :', el_pos[-5:])

# ── 6. Plot profilo topografico ───────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 4), dpi=120)
xs, zs  = zip(*el_pos)
ax.plot(xs, zs, 'o-', ms=4, lw=1.5, color='#2E86AB', label='Topografia / Elettrodi')
ax.fill_between(xs, min(zs) - 2, zs, alpha=0.12, color='#2E86AB')
for i in range(0, len(xs), 8):
    ax.annotate('Pt{}'.format(i + 1), (xs[i], zs[i]),
                textcoords='offset points', xytext=(0, 6), ha='center', fontsize=7)
ax.set_xlabel('Distanza progressiva X [m]', fontsize=11)
ax.set_ylabel('Quota Z [m a.s.l.]', fontsize=11)
ax.set_title('Profilo ERT - Sistema di Coordinate Locale 2D',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()
plt.show()
print('\nOK  el_pos costruito: {} elettrodi - pronto per Cella 2'.format(len(el_pos)))
"""

# ─────────────────────────────────────────────────────────────────────────────
# CELLA 2 – Mesh Geofisica PyGIMLi  (architettura a due regioni)
# ─────────────────────────────────────────────────────────────────────────────
CELL2 = r"""
# ═══════════════════════════════════════════════════════════════════════════
# CELLA 2 – Generazione della Mesh Geofisica (PyGIMLi)
#
# Architettura a due regioni:
#   marker=1  Zona Vadosa  (X=0-56.4 m, profondita' 0-4 m)  area<=0.005 m^2
#   marker=2  Substrato + Padding laterale                    area<=2.0   m^2
#
# Triangle crea smooth grading automatico tra le due regioni.
# quality=34 evita triangoli distorti nella zona di transizione.
# =══════════════════════════════════════════════════════════════════════════
import numpy as np
import pygimli as pg
import pygimli.meshtools as mt
import matplotlib.pyplot as plt

# ── Parametri geometrici ──────────────────────────────────────────────────
x_arr = np.array([p[0] for p in el_pos])
z_arr = np.array([p[1] for p in el_pos])

LATERAL_EXT    = 15.0   # [m] padding laterale oltre il profilo
DEPTH_TOTAL    = 20.0   # [m] profondita' max sotto z_min degli elettrodi
VADOSA_DEPTH   =  4.0   # [m] spessore zona vadosa idrogeologicamente attiva
# Area max vadosa: sqrt(2 * 0.005) = 0.10 m → dim. caratteristica <= 0.10 m
VADOSA_AREA    = 0.005  # [m^2]
SUBSTRATE_AREA = 2.0    # [m^2] substrato + padding (computazionalmente leggero)

z_min_surf = z_arr.min()
z_max_surf = z_arr.max()
z_bottom   = z_min_surf - DEPTH_TOTAL
x_min      = x_arr[0]  - LATERAL_EXT
x_max      = x_arr[-1] + LATERAL_EXT

print('Dominio  : X=[{:.1f}, {:.1f}] m | Z=[{:.2f}, {:.2f}] m a.s.l.'.format(
    x_min, x_max, z_bottom, z_max_surf))
print('Zona vadosa: X=[{:.1f}, {:.1f}] m | spessore {:.1f} m'.format(
    x_arr[0], x_arr[-1], VADOSA_DEPTH))

# ── 1. PLC Mondo intero (marker=2 – substrato + padding laterale) ─────────
# Confine superiore: topografia + padding piatto
# Confine inferiore: piano orizzontale a z_bottom
boundary_pts = (
    [[x_min, float(z_arr[0])]] +
    [[float(x), float(z)] for x, z in zip(x_arr, z_arr)] +
    [[x_max, float(z_arr[-1])],
     [x_max, z_bottom],
     [x_min, z_bottom]]
)
world_plc = mt.createPolygon(
    boundary_pts, isClosed=True, marker=2, area=SUBSTRATE_AREA
)

# ── 2. PLC Zona Vadosa (marker=1 – X=0..56.4 m, depth 0..4 m) ────────────
# Top: topografia esatta degli elettrodi (senza padding laterale)
# Base: topografia - VADOSA_DEPTH (parallela punta per punta alla topografia)
# Lati: segmenti verticali in X=x_arr[0] e X=x_arr[-1]
vadosa_top = [[float(x), float(z)] for x, z in zip(x_arr, z_arr)]
vadosa_bot = [[float(x), float(z) - VADOSA_DEPTH]
              for x, z in zip(x_arr[::-1], z_arr[::-1])]
vadosa_plc = mt.createPolygon(
    vadosa_top + vadosa_bot, isClosed=True, marker=1, area=VADOSA_AREA
)

# ── 3. Fusione PLCs e aggiunta nodi elettrodo ─────────────────────────────
full_plc = mt.mergePLC([world_plc, vadosa_plc])

# Elettrodi come nodi espliciti di superficie + sub-surface anchor
for xi, zi in el_pos:
    full_plc.createNode([float(xi), float(zi)])
    full_plc.createNode([float(xi), float(zi) - 0.01])   # stabilita' FEM

# Nodi ausiliari alla base della zona vadosa ogni 3 elettrodi:
# guidano il smooth grading dalla regione densa a quella sparsa
for xi, zi in list(zip(x_arr, z_arr))[::3]:
    full_plc.createNode([float(xi), float(zi) - VADOSA_DEPTH])

# ── 4. Generazione mesh ───────────────────────────────────────────────────
# quality=34 → angolo minimo piu' stringente per ridurre triangoli distorti
# nella zona di transizione tra le due regioni (smooth grading automatico)
print('\nGenerazione mesh in corso (quality=34, smooth=True) ...')
mesh = mt.createMesh(full_plc, quality=34, smooth=True)

# ── 5. Statistiche per regione (marker) ──────────────────────────────────
cell_areas   = np.array([c.size() for c in mesh.cells()])
cell_markers = np.array([c.marker() for c in mesh.cells()])
char_len     = np.sqrt(2.0 * cell_areas)

vadosa_mask = cell_markers == 1
substr_mask = cell_markers == 2
cl_vadosa   = char_len[vadosa_mask]
cl_substr   = char_len[substr_mask]

print('\nOK  Mesh generata:')
print('   Nodi totali        : {}'.format(mesh.nodeCount()))
print('   Celle totali       : {}'.format(mesh.cellCount()))
print()
print('   [Zona Vadosa  marker=1  (X=0-56.4 m, 0-{:.0f} m depth)]'.format(VADOSA_DEPTH))
print('   Celle              : {}'.format(vadosa_mask.sum()))
print('   Dim. max caratt.   : {:.4f} m  (limite ~0.10 m)'.format(cl_vadosa.max()))
print('   Dim. media caratt. : {:.4f} m'.format(cl_vadosa.mean()))
print()
print('   [Substrato + Padding  marker=2]')
print('   Celle              : {}'.format(substr_mask.sum()))
print('   Dim. max caratt.   : {:.4f} m'.format(cl_substr.max()))
print('   Dim. media caratt. : {:.4f} m'.format(cl_substr.mean()))

if cl_vadosa.max() <= 0.12:
    print('\nOK  Raffinamento zona vadosa CONFORME (dim.max {:.4f} m)'.format(cl_vadosa.max()))
else:
    print('\nATT Raffinamento FUORI LIMITE ({:.4f} m) - ridurre VADOSA_AREA'.format(
        cl_vadosa.max()))

# ── 6. Visualizzazione ────────────────────────────────────────────────────
markers_arr = np.array([c.marker() for c in mesh.cells()], dtype=float)

fig, axes = plt.subplots(1, 2, figsize=(18, 7), dpi=110)

# — Vista completa: celle colorate per marker di regione —
pg.show(mesh, data=markers_arr, ax=axes[0], showMesh=True,
        cMap='Set2', colorBar=True, label='Region marker')
axes[0].plot(x_arr, z_arr, 'rv', ms=5, zorder=10, label='Elettrodi (48)')
axes[0].plot(x_arr, z_arr - VADOSA_DEPTH, 'b--', lw=1.5,
             label='Base zona vadosa (-4 m)')
axes[0].set_title(
    'Mesh Geofisica - Vista Completa\n(marker 1=Vadosa  marker 2=Substrato+Padding)',
    fontsize=11, fontweight='bold')
axes[0].set_xlabel('X [m]')
axes[0].set_ylabel('Z [m a.s.l.]')
axes[0].legend(fontsize=8)

# — Zoom sulla zona vadosa (elettrodi 12-24) —
j0, j1 = 11, 24
pg.show(mesh, data=markers_arr, ax=axes[1], showMesh=True,
        cMap='Set2', colorBar=False)
axes[1].plot(x_arr, z_arr, 'rv', ms=5, zorder=10, label='Elettrodi')
axes[1].plot(x_arr, z_arr - VADOSA_DEPTH, 'b--', lw=1.5,
             label='Base vadosa (-{:.0f} m)'.format(VADOSA_DEPTH))
axes[1].set_xlim(x_arr[j0] - 0.5, x_arr[j1] + 0.5)
axes[1].set_ylim(z_arr[j0:j1+1].min() - VADOSA_DEPTH - 0.5,
                 z_arr[j0:j1+1].max() + 0.3)
axes[1].set_title(
    'Zoom Zona Vadosa (0-{:.0f} m depth)\ndim.max = {:.3f} m | celle = {}'.format(
        VADOSA_DEPTH, cl_vadosa.max(), vadosa_mask.sum()),
    fontsize=11, fontweight='bold')
axes[1].set_xlabel('X [m]')
axes[1].set_ylabel('Z [m a.s.l.]')
axes[1].legend(fontsize=9)
plt.tight_layout()
plt.show()

sensor_arr = np.array([[float(x), float(z)] for x, z in el_pos])
print('\nOK  {} posizioni elettrodo pronte - mesh pronta per forward ERT'.format(len(el_pos)))"""

# ─────────────────────────────────────────────────────────────────────────────
# CELLA 3 – Griglia MODFLOW 2D (flopy.modflow.ModflowDis) con offset datum
# ─────────────────────────────────────────────────────────────────────────────
CELL3 = r"""
# ═══════════════════════════════════════════════════════════════════════════
# CELLA 3 – Griglia MODFLOW 2D (flopy.modflow.ModflowDis)
#           con Z-offset chirurgico e localizzazione sensori tensiometrici
# ═══════════════════════════════════════════════════════════════════════════
#
# GEOMETRIA CELLE:
# ───────────────────────────────────────────────────────────────────────────
# Gli elettrodi (48) sono i BORDI delle colonne.
# ncol = 48 - 1 = 47 celle. Il centro X della cella j è la media di X[j] e X[j+1].
# Il top della cella j è la media delle quote Z degli elettrodi ai suoi bordi.
#
# Z-OFFSET CHIRURGICO (5 layer):
# ───────────────────────────────────────────────────────────────────────────
# Vero Piano Campagna (TGS)           z_TGS
# Elettrodi (piantati -10 cm dal TGS) z_el = z_TGS - 0.10
# TOP modello MODFLOW                 top_elev = z_el  (le 47 medie)
#
# Basi layer (offset dal top_elev di ogni cella):
#   L1 base: top_elev - 0.1   → centro L1 a -0.05 m dal top
#   L2 base: top_elev - 0.3   → centro L2 a -0.20 m dal top
#                                  → -0.20 + (-0.10) = -0.30 m dal TGS  ✓ Tens.1
#   L3 base: top_elev - 0.5   → centro L3 a -0.40 m dal top
#   L4 base: top_elev - 0.9   → centro L4 a -0.70 m dal top
#                                  → -0.70 + (-0.10) = -0.80 m dal TGS  ✓ Tens.2
#   L5 base: top_elev - 4.0   → copre l'intera zona vadosa attiva
# ───────────────────────────────────────────────────────────────────────────
import numpy as np
import flopy
import matplotlib.pyplot as plt
from pathlib import Path

# ── 1. Arrays da Cella 1 ─────────────────────────────────────────────────
x_arr = np.array([p[0] for p in el_pos])   # 48 posizioni X degli elettrodi [m]
z_arr = np.array([p[1] for p in el_pos])   # 48 quote Z degli elettrodi [m a.s.l.]

assert len(x_arr) == 48, 'Attesi 48 elettrodi, trovati {}'.format(len(x_arr))

# ── 2. Griglia orizzontale: ncol = 47 celle (elettrodi = bordi) ───────────
NCOL = len(x_arr) - 1                              # 47
NROW = 1                                            # 2D pseudo-profilo
NLAY = 5                                            # 5 layer chirurgici

delr = np.full(NCOL, 1.2)                          # larghezza cella = 1.2 m uniforme
delc = np.array([1.0])                             # larghezza riga (pseudo-2D)

# Centri X delle celle (per localizzazione sensori)
x_col_center = (x_arr[:-1] + x_arr[1:]) / 2.0     # shape (47,)

# ── 3. Topografia celle: media delle quote Z agli elettrodi bordo ─────────
top_elev = (z_arr[:-1] + z_arr[1:]) / 2.0         # shape (47,)  [m a.s.l.]

print('Griglia 2D: NLAY={}, NROW={}, NCOL={}'.format(NLAY, NROW, NCOL))
print('delr = {:.2f} m (uniforme)'.format(delr[0]))
print('top_elev : min={:.3f} m  max={:.3f} m  range={:.3f} m'.format(
    top_elev.min(), top_elev.max(), top_elev.max() - top_elev.min()))

# ── 4. Basi layer (Z-offset chirurgico) ──────────────────────────────────
# Tutti gli offset sono RELATIVI a top_elev di ogni cella (topografia variabile)
# Shape botm: (NLAY, NROW, NCOL) = (5, 1, 47)
OFFSETS = np.array([0.1, 0.3, 0.5, 0.9, 4.0])   # profondita' basi layer [m]

botm = np.zeros((NLAY, NROW, NCOL))
for k, dz in enumerate(OFFSETS):
    botm[k, 0, :] = top_elev - dz                 # quota assoluta base layer k

# Top array per MODFLOW DIS (shape NROW x NCOL)
top_2d = top_elev.reshape(NROW, NCOL)

# ── 5. Verifica centri cella (quote assolute) ─────────────────────────────
# Centro L1: (top_elev + base_L1) / 2 = top_elev - 0.05  → -0.05 m dal top
# Centro L2: (base_L1 + base_L2) / 2 = top_elev - 0.20  → -0.20 m dal top = -0.30 m TGS
# Centro L4: (base_L3 + base_L4) / 2 = top_elev - 0.70  → -0.70 m dal top = -0.80 m TGS
ctr_abs = np.zeros((NLAY, NCOL))
for k in range(NLAY):
    top_k = top_elev if k == 0 else botm[k-1, 0, :]
    bot_k = botm[k, 0, :]
    ctr_abs[k, :] = (top_k + bot_k) / 2.0

# Offset dal top_elev (dovrebbe essere costante columnwise per verifica)
ctr_offset = top_elev - ctr_abs[1, :]   # Layer 2: atteso ~0.20 m
ctr_offset_L4 = top_elev - ctr_abs[3, :]  # Layer 4: atteso ~0.70 m
print('\nVerifica centri cella (valori medi su 47 colonne):')
print('  Centro L2 a {:.4f} m sotto top_elev  → {:.4f} m sotto TGS (target -0.30 m)'.format(
    ctr_offset.mean(), ctr_offset.mean() + 0.10))
print('  Centro L4 a {:.4f} m sotto top_elev  → {:.4f} m sotto TGS (target -0.80 m)'.format(
    ctr_offset_L4.mean(), ctr_offset_L4.mean() + 0.10))

# ── 6. Costruzione modello flopy.modflow (MODFLOW 2005) ───────────────────
MODEL_WS = Path(
    'C:/Users/luigi/git/github.com/luigimartinoisio-blip/PyHydroGeophysiX/mf2005_coupled'
)
MODEL_WS.mkdir(parents=True, exist_ok=True)
MODEL_NAME = 'coupled_hydrogeo_dis'

model = flopy.modflow.Modflow(
    modelname = MODEL_NAME,
    exe_name  = 'mf2005',        # deve essere nel PATH; non eseguito qui
    model_ws  = str(MODEL_WS),
    version   = 'mf2005',
)

# DIS – discretizzazione spazio-temporale
dis = flopy.modflow.ModflowDis(
    model,
    nlay = NLAY,
    nrow = NROW,
    ncol = NCOL,
    delr = delr,
    delc = delc,
    top  = top_2d,
    botm = botm,
    nper = 1,
    perlen   = [1.0],
    nstp     = [1],
    steady   = [True],
    itmuni   = 4,    # unità tempo: giorni
    lenuni   = 2,    # unità lunghezza: metri
)

print('\nOK  Pacchetto DIS creato:')
print('   NLAY={}, NROW={}, NCOL={}'.format(dis.nlay, dis.nrow, dis.ncol))
print('   Extent X: [{:.2f}, {:.2f}] m'.format(
    float(x_col_center[0]) - 0.6, float(x_col_center[-1]) + 0.6))
print('   Extent Z: [{:.3f}, {:.3f}] m a.s.l.'.format(
    float(botm[-1, 0, :].min()), float(top_elev.max())))

# ── 7. Centri cella (quote assolute) per Layer 2 e Layer 4 ───────────────
# ModflowDis non espone zcellcenters; usiamo ctr_abs calcolato al passo 5.
# ctr_abs[k, j] = (top_k[j] + bot_k[j]) / 2  →  shape (NLAY, NCOL)

# Layer 2 → indice 1;  Layer 4 → indice 3
z_L2 = ctr_abs[1, :]   # shape (47,)   [m a.s.l.]
z_L4 = ctr_abs[3, :]   # shape (47,)   [m a.s.l.]

# ── 8. Localizzazione sensori a X = 8 m, 31 m, 51 m ──────────────────────
SENS_X_TARGETS = [8.0, 31.0, 51.0]   # [m] posizioni nominali sensori

print('\n' + '=' * 72)
print('  REPORT SENSORI TENSIOMETRICI – POSIZIONI E QUOTE NODALI')
print('  (Layer 2 ≈ -30 cm TGS  |  Layer 4 ≈ -80 cm TGS)')
print('=' * 72)
print('  {:<10}{:<10}{:<18}{:<22}{:<22}'.format(
    'X target', 'col_idx', 'X cella [m]', 'Z nodo L2 [m asl]', 'Z nodo L4 [m asl]'))
print('  ' + '-' * 70)

for x_target in SENS_X_TARGETS:
    # Indice di colonna più vicino al target
    col_idx = int(np.argmin(np.abs(x_col_center - x_target)))
    x_real  = float(x_col_center[col_idx])
    z_sens2 = float(z_L2[col_idx])
    z_sens4 = float(z_L4[col_idx])
    print('  {:<10.1f}{:<10d}{:<18.4f}{:<22.4f}{:<22.4f}'.format(
        x_target, col_idx, x_real, z_sens2, z_sens4))

print('=' * 72)
print()
print('Nota: Z nodo = quota assoluta centro-cella [m a.s.l.].')
print('      L2 centro a -0.20 m dal top modello = -0.30 m dal TGS.')
print('      L4 centro a -0.70 m dal top modello = -0.80 m dal TGS.')
print()
print('OK  Cella 3 completata. Modello MODFLOW 2005 pronto (non eseguito).')
print('    Workspace: {}'.format(MODEL_WS))
"""


# ─────────────────────────────────────────────────────────────────────────────
# CELLA 4 – Funzioni Petrofisiche Pure (Van Genuchten + Archie)
# ─────────────────────────────────────────────────────────────────────────────
CELL4 = r"""
# ═══════════════════════════════════════════════════════════════════════════
# CELLA 4 – Funzioni Petrofisiche: Van Genuchten & Legge di Archie
#
# Scopo : tradurre output idrologici (carico di pressione h) in input
#         geofisici (resistività elettrica bulk rho_b) attraverso:
#
#   MODFLOW  →  h [m]  →  vg_theta()  →  θ [-]  →  archie_rho()  →  ρ_b [Ω·m]
#
# Funzioni vettorializzate su array numpy, gestione singolarità inclusa.
# ═══════════════════════════════════════════════════════════════════════════
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── 1. Van Genuchten (1980) ───────────────────────────────────────────────
def vg_theta(h, theta_r, theta_s, alpha, n_vg):
    # Contenuto d'acqua volumetrico theta [-]  (Van Genuchten 1980).
    # h       : carico di pressione [m]   (h < 0 = insaturo)
    # theta_r : contenuto d'acqua residuo [-]
    # theta_s : contenuto d'acqua a saturazione [-]
    # alpha   : param. di scala VG [1/m]
    # n_vg    : param. di forma VG [-]  (n > 1)
    # m_vg    = 1 - 1/n_vg  (Mualem)
    # Per h >= 0 si forza theta = theta_s.
    h    = np.asarray(h, dtype=float)
    m_vg = 1.0 - 1.0 / n_vg

    # Calcolo su tutto l'array (formula matriciale, no loop)
    Se    = 1.0 / (1.0 + np.abs(alpha * h) ** n_vg) ** m_vg
    theta = theta_r + (theta_s - theta_r) * Se

    # Condizione di saturazione: h >= 0 → theta = theta_s
    theta = np.where(h >= 0.0, theta_s, theta)
    return theta


# ── 2. Legge di Archie generalizzata (insaturo) ───────────────────────────
def archie_rho(theta, porosity, rho_w, m, n_arch):
    # Resistivita' bulk rho_b [ohm*m]  (Archie generalizzato, insaturo).
    # theta    : contenuto d'acqua volumetrico [-]
    # porosity : porosita' totale phi          [-]
    # rho_w    : resistivita' acqua di poro    [ohm*m]
    # m        : esponente di cementazione     [-]
    # n_arch   : esponente di saturazione      [-]
    # S_w = theta/phi  clampato in [1e-8, 1.0]  per evitare singolarita'.
    # rho_b = rho_w * phi^(-m) * S_w^(-n)
    theta = np.asarray(theta, dtype=float)
    Sw    = theta / porosity
    Sw    = np.clip(Sw, 1e-8, 1.0)          # evita Sw=0 e Sw>1
    rho_b = rho_w * porosity**(-m) * Sw**(-n_arch)
    return rho_b


print('OK  Funzioni petrofisiche definite:')
print('   vg_theta(h, theta_r, theta_s, alpha, n_vg)   → θ [-]')
print('   archie_rho(theta, porosity, rho_w, m, n_arch) → ρ_b [Ω·m]')

# ── 3. Parametri tipici: limo/argilla ────────────────────────────────────
THETA_S   = 0.45    # contenuto d'acqua a saturazione  [-]
THETA_R   = 0.08    # contenuto d'acqua residuo        [-]
ALPHA_VG  = 1.5     # parametro di scala VG            [1/m]
N_VG      = 1.4     # parametro di forma VG            [-]
POROSITY  = 0.45    # porosità totale                  [-]
RHO_W     = 20.0    # resistività acqua di poro        [Ω·m]
M_ARCH    = 1.8     # esponente di cementazione        [-]
N_ARCH    = 2.0     # esponente di saturazione         [-]

# ── 4. Array di carico di pressione (0 → -10 m) ──────────────────────────
h_test = np.linspace(0.0, -10.0, 500)     # [m]  suzione crescente

# ── 5. Funzioni a cascata ────────────────────────────────────────────────
theta_test = vg_theta(h_test, THETA_R, THETA_S, ALPHA_VG, N_VG)
rho_test   = archie_rho(theta_test, POROSITY, RHO_W, M_ARCH, N_ARCH)

# Statistiche rapide
print('\n  Capo h [m]  │  θ [-]   │  ρ_b [Ω·m]')
print('  ' + '─' * 40)
for h_val in [0.0, -0.1, -0.5, -1.0, -3.0, -10.0]:
    idx = int(np.argmin(np.abs(h_test - h_val)))
    print('  {:>10.1f}  │  {:.4f}  │  {:>9.2f}'.format(
        h_test[idx], theta_test[idx], rho_test[idx]))

# ── 6. Visualizzazione ───────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=115)
fig.suptitle(
    'Funzioni Petrofisiche  –  Limo/Argilla\n'
    r'($\theta_s$={:.2f}, $\theta_r$={:.2f}, $\alpha$={:.1f} m$^{{-1}}$, '
    r'$n_{{VG}}$={:.1f}, $\phi$={:.2f}, $\rho_w$={:.0f} Ω·m, '
    r'$m$={:.1f}, $n_{{arch}}$={:.1f})'.format(
        THETA_S, THETA_R, ALPHA_VG, N_VG,
        POROSITY, RHO_W, M_ARCH, N_ARCH),
    fontsize=10, fontweight='bold')

# ─ Subplot 1: Curva di ritenzione idrica h vs θ ─
ax1.plot(h_test, theta_test, color='#2E86AB', lw=2.0)
ax1.axhline(THETA_S, color='#2E86AB', lw=0.8, ls='--', alpha=0.6,
            label=r'$\theta_s = {:.2f}$'.format(THETA_S))
ax1.axhline(THETA_R, color='#A23B72', lw=0.8, ls='--', alpha=0.6,
            label=r'$\theta_r = {:.2f}$'.format(THETA_R))
# Punti tensiometrici tipici (profondità 30 e 80 cm dal TGS → h indicativo)
for h_mark, lbl, col in [(-0.3, 'Tens.1 (-30 cm)', '#E84855'),
                          (-0.8, 'Tens.2 (-80 cm)', '#FF8C42')]:
    th_m = vg_theta(h_mark, THETA_R, THETA_S, ALPHA_VG, N_VG)
    ax1.scatter([h_mark], [th_m], s=60, zorder=8, color=col, label=lbl)
ax1.set_xlabel('Carico di pressione  $h$  [m]', fontsize=11)
ax1.set_ylabel('Contenuto acqua  $\\theta$  [-]', fontsize=11)
ax1.set_title('Curva di ritenzione idrica (VG)', fontsize=11, fontweight='bold')
ax1.invert_xaxis()          # suzione decrescente verso destra (asse h invertito)
ax1.set_xlim(0, -10)
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3, linestyle='--')

# ─ Subplot 2: Curva petrofisica θ vs ρ_b ─
ax2.semilogy(theta_test, rho_test, color='#8B2FC9', lw=2.0)
# Punti tensiometrici
for h_mark, lbl, col in [(-0.3, 'Tens.1 (-30 cm)', '#E84855'),
                          (-0.8, 'Tens.2 (-80 cm)', '#FF8C42')]:
    th_m  = vg_theta(h_mark, THETA_R, THETA_S, ALPHA_VG, N_VG)
    rho_m = archie_rho(th_m, POROSITY, RHO_W, M_ARCH, N_ARCH)
    ax2.scatter([th_m], [rho_m], s=60, zorder=8, color=col, label=lbl)
ax2.set_xlabel('Contenuto acqua  $\\theta$  [-]', fontsize=11)
ax2.set_ylabel('Resistivita bulk  $\\rho_b$  [ohm*m]', fontsize=11)
ax2.set_title('Curva petrofisica Archie (scala log rho)', fontsize=11, fontweight='bold')
ax2.yaxis.set_major_formatter(ticker.ScalarFormatter())
ax2.legend(fontsize=9)
ax2.grid(True, which='both', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.show()

print('\nOK  Cella 4 completata.')
print('   θ(h = 0.0 m) = {:.4f}  →  ρ_b = {:.2f} Ω·m  (saturo)'.format(
    float(vg_theta(0.0,   THETA_R, THETA_S, ALPHA_VG, N_VG)),
    float(archie_rho(THETA_S, POROSITY, RHO_W, M_ARCH, N_ARCH))))
print('   θ(h = -3.0 m) = {:.4f}  →  ρ_b = {:.2f} Ω·m'.format(
    float(vg_theta(-3.0,  THETA_R, THETA_S, ALPHA_VG, N_VG)),
    float(archie_rho(vg_theta(-3.0, THETA_R, THETA_S, ALPHA_VG, N_VG),
                     POROSITY, RHO_W, M_ARCH, N_ARCH))))
print('   θ(h = -10.0 m) = {:.4f}  →  ρ_b = {:.2f} Ω·m  (quasi-residuo)'.format(
    float(vg_theta(-10.0, THETA_R, THETA_S, ALPHA_VG, N_VG)),
    float(archie_rho(vg_theta(-10.0, THETA_R, THETA_S, ALPHA_VG, N_VG),
                     POROSITY, RHO_W, M_ARCH, N_ARCH))))
"""


# ─────────────────────────────────────────────────────────────────────────────
# Assemblaggio notebook
# ─────────────────────────────────────────────────────────────────────────────
def to_source(code: str) -> list:
    """Converte una stringa di codice in lista-di-righe per Jupyter."""
    lines = code.lstrip('\n').splitlines(keepends=True)
    if lines and lines[-1].endswith('\n'):
        lines[-1] = lines[-1][:-1]   # ultima riga senza \n finale (convenzione Jupyter)
    return lines


def make_code_cell(src: str) -> dict:
    return {
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': to_source(src),
    }


notebook = {
    'nbformat': 4,
    'nbformat_minor': 5,
    'metadata': {
        'kernelspec': {
            'display_name': 'Python 3 (.venv)',
            'language': 'python',
            'name': 'python3',
        },
        'language_info': {
            'codemirror_mode': {'name': 'ipython', 'version': 3},
            'file_extension': '.py',
            'mimetype': 'text/x-python',
            'name': 'python',
            'pygments_lexer': 'ipython3',
            'version': '3.11.0',
        },
    },
    'cells': [
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': (
                '# Coupled 2D Hydrogeophysical Modeling\n'
                '**PyHydroGeophysiX · PyGIMLi · Flopy MODFLOW 6**\n\n'
                'Test sintetico 2D accoppiato:\n'
                '- **Cella 1** – Parsing dati reali → `el_pos` (48 elettrodi)\n'
                '- **Cella 2** – Mesh geofisica PyGIMLi (Zona Vadosa marker=1, Substrato+Padding marker=2)\n'
                '- **Cella 3** – Griglia MODFLOW 2005 con offset datum elettrodi/tensiometri\n'
                '- **Cella 4** – Funzioni petrofisiche: Van Genuchten + Archie (h → θ → ρ_b)'
            ),
        },
        make_code_cell(CELL1),
        make_code_cell(CELL2),
        make_code_cell(CELL3),
        make_code_cell(CELL4),
    ],
}

OUTPUT = Path(
    'C:/Users/luigi/git/github.com/luigimartinoisio-blip/PyHydroGeophysiX'
    '/Coupled_2D_Hydrogeophysical_Modeling.ipynb'
)

with open(OUTPUT, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print('OK  Notebook generato:')
print('    {}'.format(OUTPUT))
print('    -> Aprire con Jupyter nel venv PyHydroGeophysiX')
