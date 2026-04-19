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
# CELLA 3 – Discretizzazione Idrologica Flopy / MODFLOW 6
# ─────────────────────────────────────────────────────────────────────────────
CELL3 = r"""
# ═══════════════════════════════════════════════════════════════════════════
# CELLA 3 – Discretizzazione Idrologica (Flopy / MODFLOW 6)
#            con Offset di Datum tra Elettrodi e Tensiometri
# ═══════════════════════════════════════════════════════════════════════════
#
# FISICA DELL'OFFSET DI DATUM
# ───────────────────────────────────────────────────────────────────────────
# Vero Piano Campagna (TGS)         ━━━━━━━━━━━  z_TGS
# Elettrodi (piantati -10 cm TGS)   ═══════════  z_el = z_TGS - 0.10
# TOP del modello MODFLOW           = z_el
#
# Tens.1 a -30 cm dal TGS  =  z_TGS - 0.30  =  z_el - 0.20  (20 cm dal top)
# Tens.2 a -80 cm dal TGS  =  z_TGS - 0.80  =  z_el - 0.70  (70 cm dal top)
#
# CALCOLO SPESSORI LAYER per centratura esatta dei nodi:
# ───────────────────────────────────────────────────────────────────────────
# Layer 1: centro = Dz1/2 = 0.20 m  -->  Dz1 = 0.40 m  ✓
# Layer 2: centro = Dz1 + Dz2/2 = 0.70 m  -->  Dz2 = 0.60 m  ✓
# Layer 3–6: spessori progressivi per coprire 20 m di dominio
# ───────────────────────────────────────────────────────────────────────────
import numpy as np
import flopy
import matplotlib.pyplot as plt
from pathlib import Path

# ── Coordinate elettrodi (da Cella 1) ────────────────────────────────────
x_arr = np.array([p[0] for p in el_pos])
z_arr = np.array([p[1] for p in el_pos])   # quota elettrodi = TOP del modello

N_ELEC       = len(x_arr)                          # 48
ELEC_SPACING = float(np.diff(x_arr).mean())        # 1.2 m

# ── Offset fisico datum ───────────────────────────────────────────────────
ELEC_OFFSET = 0.10   # [m] elettrodi piantati 10 cm sotto il TGS
SENS1_TGS   = 0.30   # [m] Tensiometro 1 a 30 cm dal TGS
SENS2_TGS   = 0.80   # [m] Tensiometro 2 a 80 cm dal TGS
SENS1_MODEL = SENS1_TGS - ELEC_OFFSET   # = 0.20 m dal top modello
SENS2_MODEL = SENS2_TGS - ELEC_OFFSET   # = 0.70 m dal top modello

# ── Spessori layer [m] ────────────────────────────────────────────────────
# Derivazione:
#   Layer 1: Dz1 = 2 * SENS1_MODEL = 2 * 0.20 = 0.40 m
#   Layer 2: Dz1 + Dz2/2 = 0.70  -->  Dz2 = 2*(0.70 - 0.40) = 0.60 m
DZ = np.array([
     0.40,   # Layer 1 – nodo Tens.1 centrato a -20 cm dal top modello
     0.60,   # Layer 2 – nodo Tens.2 centrato a -70 cm dal top modello
     1.50,   # Layer 3 – zona vadosa profonda
     2.50,   # Layer 4 – zona di transizione
     5.00,   # Layer 5 – zona satura
    10.00,   # Layer 6 – substrato / acquifero profondo  (tot = 20 m)
])
NLAY = len(DZ)   # 6
NROW = 1         # griglia pseudo-2D (profilo)
NCOL = N_ELEC    # 48 colonne, una per elettrodo

# ── Verifica analitica centri cella ──────────────────────────────────────
cumDZ = np.cumsum(DZ)                  # fondo di ogni layer dal top [m]
ctrs  = cumDZ - DZ / 2.0              # centri di cella dal top [m]

if abs(ctrs[0] - SENS1_MODEL) > 1e-9:
    raise AssertionError('Layer 1 centre {:.4f} != target {:.4f}'.format(
        ctrs[0], SENS1_MODEL))
if abs(ctrs[1] - SENS2_MODEL) > 1e-9:
    raise AssertionError('Layer 2 centre {:.4f} != target {:.4f}'.format(
        ctrs[1], SENS2_MODEL))

print('OK  Centri cella verificati analiticamente:')
print('   Layer 1: {:.3f} m dal top  (target {:.3f} m - Tens.1) OK'.format(
      ctrs[0], SENS1_MODEL))
print('   Layer 2: {:.3f} m dal top  (target {:.3f} m - Tens.2) OK'.format(
      ctrs[1], SENS2_MODEL))

# ── Griglia MODFLOW ───────────────────────────────────────────────────────
delr   = np.full(NCOL, ELEC_SPACING)    # larghezze colonne = 1.2 m uniforme
delc   = np.array([1.0])                # larghezza riga (pseudo-2D)
top_2d = z_arr.reshape(NROW, NCOL)     # top variabile = quota elettrodi

botm = np.zeros((NLAY, NROW, NCOL))
for k in range(NLAY):
    botm[k, 0, :] = z_arr - cumDZ[k]   # fondo assoluto layer k [m a.s.l.]

# ── Costruzione simulazione MODFLOW 6 (Flopy) ────────────────────────────
MODEL_WS = Path(
    'C:/Users/luigi/git/github.com/luigimartinoisio-blip/PyHydroGeophysiX/mf6_coupled'
)
MODEL_WS.mkdir(parents=True, exist_ok=True)
MODEL_NAME = 'coupled_hydrogeo'

sim = flopy.mf6.MFSimulation(
    sim_name = MODEL_NAME,
    sim_ws   = str(MODEL_WS),
    exe_name = 'mf6',          # 'mf6' deve essere nel PATH di sistema
)
# TDIS – singolo periodo stazionario (placeholder per regime transiente)
tdis = flopy.mf6.ModflowTdis(
    sim, nper=1, perioddata=[(1.0, 1, 1.0)]   # (perlen[d], nstp, tsmult)
)
# IMS – solutore iterativo
ims = flopy.mf6.ModflowIms(sim, complexity='MODERATE', outer_maximum=100)
# GWF – Groundwater Flow Model
gwf = flopy.mf6.ModflowGwf(sim, modelname=MODEL_NAME, save_flows=True)
# DIS – griglia strutturata con top variabile (quota elettrodi)
dis = flopy.mf6.ModflowGwfdis(
    gwf,
    nlay=NLAY, nrow=NROW, ncol=NCOL,
    delr=delr, delc=delc,
    top=top_2d, botm=botm,
    length_units='METERS',
)
# IC – livello piezometrico iniziale = quota superficie (placeholder)
ic = flopy.mf6.ModflowGwfic(gwf, strt=top_2d)
# NPF – proprieta' idrauliche (placeholder; da calibrare con inversione)
npf = flopy.mf6.ModflowGwfnpf(
    gwf, k=1e-5, icelltype=1   # Kh [m/s]; celle convertibili (unconfinate)
)
# OC – output head e budget
oc = flopy.mf6.ModflowGwfoc(
    gwf,
    head_filerecord   = '{}.hds'.format(MODEL_NAME),
    budget_filerecord = '{}.bud'.format(MODEL_NAME),
    saverecord  = [('HEAD', 'ALL'), ('BUDGET', 'ALL')],
    printrecord = [('HEAD', 'LAST')],
)
sim.write_simulation()
print('\nOK  File MODFLOW 6 scritti in:\n   {}'.format(MODEL_WS))

# ── Tabella struttura layer ───────────────────────────────────────────────
uses = [
    'Tens.1 (TGS-{}cm / modello-{}cm)'.format(int(SENS1_TGS*100), int(SENS1_MODEL*100)),
    'Tens.2 (TGS-{}cm / modello-{}cm)'.format(int(SENS2_TGS*100), int(SENS2_MODEL*100)),
    'Zona vadosa profonda',
    'Zona di transizione',
    'Zona satura',
    'Substrato / acquifero profondo',
]
tops_from_top = np.concatenate([[0.0], cumDZ[:-1]])
print('\n  STRUTTURA LAYER MODFLOW')
print('  {:<8}{:<10}{:<18}{:<22}{}'.format(
      'Layer', 'Dz [m]', 'Top [m top]', 'Centro [m top]', 'Uso'))
print('  ' + '-' * 72)
for k in range(NLAY):
    print('  {:<8}{:<10.2f}{:<18.3f}{:<22.3f}{}'.format(
          k+1, DZ[k], tops_from_top[k], ctrs[k], uses[k]))

# ── Visualizzazione sezione verticale ─────────────────────────────────────
cmap_lay = plt.get_cmap('RdYlBu_r', NLAY)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), dpi=110,
                               gridspec_kw={'height_ratios': [3, 2]})

ctr1_abs = (z_arr + botm[0, 0, :]) / 2.0
ctr2_abs = (botm[0, 0, :] + botm[1, 0, :]) / 2.0

# — Sezione completa: tutti i layer —
for k in range(NLAY):
    top_k = z_arr if k == 0 else botm[k-1, 0, :]
    bot_k = botm[k, 0, :]
    ax1.fill_between(x_arr, bot_k, top_k, alpha=0.65, color=cmap_lay(k),
                     label='Layer {} Dz={:.2f} m'.format(k+1, DZ[k]))
    ax1.plot(x_arr, (top_k + bot_k) / 2.0, '--', lw=0.7, color=cmap_lay(k), alpha=0.8)
ax1.plot(x_arr, z_arr, 'k-', lw=2, label='Top modello (elettrodi)')
ax1.plot(x_arr, z_arr + ELEC_OFFSET, 'k--', lw=1.2, alpha=0.6,
         label='Vero piano campagna (+{}cm)'.format(int(ELEC_OFFSET*100)))
ax1.plot(x_arr, ctr1_abs, 'rs', ms=6, zorder=12,
         label='Nodo Tens.1 (mod.-{}cm / TGS-{}cm)'.format(
             int(SENS1_MODEL*100), int(SENS1_TGS*100)))
ax1.plot(x_arr, ctr2_abs, 'b^', ms=6, zorder=12,
         label='Nodo Tens.2 (mod.-{}cm / TGS-{}cm)'.format(
             int(SENS2_MODEL*100), int(SENS2_TGS*100)))
ax1.set_xlabel('X [m]', fontsize=11)
ax1.set_ylabel('Z [m a.s.l.]', fontsize=11)
ax1.set_title('Griglia MODFLOW - Sezione Verticale con Offset di Datum',
              fontsize=13, fontweight='bold')
ax1.legend(loc='lower left', fontsize=8, ncol=2)
ax1.grid(True, alpha=0.3, linestyle='--')

# — Zoom layer 1-2 (zona sensori) —
for k in range(2):
    top_k = z_arr if k == 0 else botm[k-1, 0, :]
    bot_k = botm[k, 0, :]
    ax2.fill_between(x_arr, bot_k, top_k, alpha=0.70, color=cmap_lay(k),
                     label='Layer {} Dz={:.2f} m - centro -{:.0f}cm'.format(
                         k+1, DZ[k], ctrs[k]*100))
ax2.plot(x_arr, z_arr, 'k-', lw=2, label='Top (elettrodi)')
ax2.plot(x_arr, z_arr + ELEC_OFFSET, 'k--', lw=1.2, alpha=0.6, label='TGS')
ax2.plot(x_arr, ctr1_abs, 'rs', ms=7, zorder=12,
         label='Tens.1 → -{:.0f}cm'.format(SENS1_MODEL*100))
ax2.plot(x_arr, ctr2_abs, 'b^', ms=7, zorder=12,
         label='Tens.2 → -{:.0f}cm'.format(SENS2_MODEL*100))
ax2.set_ylim(z_arr.min() - cumDZ[1] - 0.3, z_arr.max() + 0.3)
ax2.set_xlabel('X [m]', fontsize=11)
ax2.set_ylabel('Z [m a.s.l.]', fontsize=11)
ax2.set_title('Zoom Layer 1-2: Zona Sensori (Vadosa Superficiale)',
              fontsize=11, fontweight='bold')
ax2.legend(loc='lower left', fontsize=9)
ax2.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()
plt.show()

# ── Verifica numerica colonna centrale ────────────────────────────────────
j_ref = NCOL // 2
print('\n  VERIFICA NUMERICA (colonna j={}, X={:.1f} m)'.format(j_ref, x_arr[j_ref]))
print('  {:<6}{:<16}{:<16}{:<18}{:<16}{}'.format(
      'Layer', 'Top[a.s.l.]', 'Bot[a.s.l.]', 'Centro[a.s.l.]', 'Dal top[m]', 'Nota'))
print('  ' + '-' * 82)
for k in range(NLAY):
    tk = float(z_arr[j_ref]) if k == 0 else float(botm[k-1, 0, j_ref])
    bk = float(botm[k, 0, j_ref])
    ck = (tk + bk) / 2.0
    dk = float(z_arr[j_ref]) - ck
    flag = '<- Tens.1 OK' if k == 0 else ('<- Tens.2 OK' if k == 1 else '')
    print('  {:<6}{:<16.3f}{:<16.3f}{:<18.3f}{:<16.3f}{}'.format(
          k+1, tk, bk, ck, dk, flag))
print()
print('OK  MODFLOW 6 pronto per accoppiamento con forward ERT (PyGIMLi mesh)')
print('   Offs.: elettrodi a -{}cm sotto TGS'.format(int(ELEC_OFFSET*100)))
print('   Tens.1 TGS-{}cm → Layer 1 nodo a -{}cm dal top modello'.format(
      int(SENS1_TGS*100), int(SENS1_MODEL*100)))
print('   Tens.2 TGS-{}cm → Layer 2 nodo a -{}cm dal top modello'.format(
      int(SENS2_TGS*100), int(SENS2_MODEL*100)))
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
                '- **Cella 3** – Griglia MODFLOW con offset datum elettrodi/tensiometri'
            ),
        },
        make_code_cell(CELL1),
        make_code_cell(CELL2),
        make_code_cell(CELL3),
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
