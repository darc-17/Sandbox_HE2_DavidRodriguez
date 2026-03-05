"""
Punto 1 – Clasificación para Focalización de Programas Sociales
================================================================
Taller 1 · Consultoría Económica con IA Responsable
Autores : David Rodríguez · Juan Rueda
Fecha   : 2026

Contexto
--------
Cliente  : BID (Banco Interamericano de Desarrollo)
País     : Costa Rica
Programa : Subsidios de asistencia social (IMAS / SINIRUBE)
Problema : El modelo vigente comete errores de focalización:
           • Falsos negativos → hogares vulnerables excluidos del subsidio.
           • Falsos positivos → hogares no necesitados que reciben el subsidio.

Variable objetivo (Target)
--------------------------
  1 = Pobreza extrema   → clase POBRE     (Target_binario = 1)
  2 = Pobreza moderada  → clase POBRE     (Target_binario = 1)
  3 = Vulnerable        → clase NO POBRE  (Target_binario = 0)
  4 = No pobre          → clase NO POBRE  (Target_binario = 0)

La binarización se aplica con la regla de mayoría a nivel de hogar.
En caso de empate se elige 1 (pobre) para minimizar falsos negativos,
acorde al objetivo del programa de no excluir hogares vulnerables.

Pipeline
--------
  1. Carga de datos
  2. Limpieza
     2.1  Eliminación de variables redundantes
     2.2  Imputación de valores faltantes
     2.3  Dummies de varianza casi cero
     2.4  Tratamiento de outliers
     2.5  Estandarización de edjefe / edjefa
     2.6  Reparación de hogares sin jefe registrado
     2.7  Reconstrucción de tasa_dependencia
     2.8  Eliminación de variables adicionales
  3. Binarización del Target (regla de mayoría por hogar)
  4. Agregación al nivel de hogar (idhogar)
  5. Diagnóstico de multicolinealidad (VIF)
  6. Eliminación final
     6a. Resolución de multicolinealidades perfectas post-VIF
     6b. Reducción de dimensionalidad post-agregación
  7. Renombrado de variables a nombres descriptivos
  8. Exportación del dataset limpio

Ajustes respecto al notebook original
--------------------------------------
  • meaneduc: capping al máximo teórico (21 años); el valor 37 es imposible.
  • instlevel1 (sin educación): preservada — es indicador NBI directo y
    tiene correlación confirmada con pobreza (Target medio 2.92 vs 3.34).
  • estadocivil6 (viudo/a) y estadocivil7 (soltero/a): preservadas —
    asociadas a hogares monoparentales vulnerables.
  • pareddes (paredes de desecho) y pisonotiene (piso de tierra): preservadas
    en la reducción final — son mediciones objetivas de material, distintas
    de la evaluación subjetiva epared1/eviv1.
  • dependency: eliminada y reconstruida como tasa_dependencia usando la
    fórmula original del INEC (verificada: correlación = 1.0 con valores
    numéricos del dataset).
  • female: agregada con mean (proporción de mujeres), no con max.
"""

# =============================================================================
# 0. IMPORTACIONES Y CONFIGURACIÓN
# =============================================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor

pd.set_option("display.max_columns", 50)
sns.set_theme(style="whitegrid")

# Carpetas de salida — siempre relativas al directorio del script,
# independientemente del directorio de trabajo (CWD) de Python
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DIR_DATOS  = os.path.join(SCRIPT_DIR, "Datos")
DIR_VIZS   = os.path.join(SCRIPT_DIR, "Visualizaciones")

# URL del dataset (repositorio del taller)
URL_TRAIN = (
    "https://raw.githubusercontent.com/darc-17/Sandbox_HE2_DavidRodriguez"
    "/refs/heads/main/Taller%201/Punto%201.%20Clasificacion%20para%20"
    "Focalizaci%C3%B3n%20de%20Programas%20Sociales/train.csv"
)

# =============================================================================
# 1. CARGA DE DATOS
# =============================================================================

print("=" * 65)
print("1. CARGA DE DATOS")
print("=" * 65)

df = pd.read_csv(URL_TRAIN)

print(f"  Shape inicial : {df.shape[0]:,} individuos · {df.shape[1]} variables")
print(f"  Tipos de dato :\n{df.dtypes.value_counts().rename('columnas').to_string()}")

# =============================================================================
# 2. LIMPIEZA
# =============================================================================

# -----------------------------------------------------------------------------
# 2.1  ELIMINACIÓN DE VARIABLES REDUNDANTES
# -----------------------------------------------------------------------------
# Criterio: variables que son transformaciones matemáticas de otras (SQB),
# duplicados exactos, o sumas de partes ya disponibles por separado.
# -----------------------------------------------------------------------------

print("\n" + "=" * 65)
print("2.1  Eliminación de variables redundantes")
print("=" * 65)

drop_redundantes = [
    # Términos cuadráticos (SQB) — transformaciones de variables ya presentes.
    # Se eliminan porque el modelo puede aprender no-linealidades por su cuenta
    # o generarlas explícitamente si es necesario.
    "SQBescolari", "SQBage", "agesq", "SQBhogar_total",
    "SQBedjefe", "SQBhogar_nin", "SQBovercrowding",
    "SQBdependency", "SQBmeaned",

    # Tamaño del hogar — duplicados entre sí y de hogar_total
    "tamhog", "hhsize", "tamviv",

    # Subtotales por género ya calculados: r4h3 = r4h1+r4h2, etc.
    "r4h3", "r4m3", "r4t3",

    # Género — male + female = 1 siempre; se conserva female
    "male",
]

drop_redundantes = [c for c in drop_redundantes if c in df.columns]
df.drop(columns=drop_redundantes, inplace=True)
print(f"  Eliminadas : {len(drop_redundantes)} columnas")
print(f"  Shape      : {df.shape}")

# -----------------------------------------------------------------------------
# 2.2  IMPUTACIÓN DE VALORES FALTANTES
# -----------------------------------------------------------------------------
# Cada missing tiene una causa estructural; se imputa según su mecanismo
# subyacente. Cuando el missing mismo es informativo, se crea una variable
# indicadora binaria antes de imputar.
# -----------------------------------------------------------------------------

print("\n" + "=" * 65)
print("2.2  Imputación de valores faltantes")
print("=" * 65)

miss_df = (
    df.isnull().sum()
    .pipe(lambda s: s[s > 0])
    .sort_values(ascending=False)
    .rename("n_missing")
    .to_frame()
    .assign(pct=lambda x: (x["n_missing"] / len(df) * 100).round(2))
)
print("  Variables con missing:")
print(miss_df.to_string())

# ── Análisis del mecanismo causal de cada missing ─────────────────────────
# Se verifica empíricamente que los NaN no son aleatorios, sino que
# tienen una causa estructural que justifica la imputación determinística.
# Es decir, están como missings pero son ceros, por ejemplo.

print("\n  [PRUEBA 2.2a] v18q1 — NaN ocurre exclusivamente cuando v18q=0?")
n_nan_con_tablet    = df[df["v18q"] == 1]["v18q1"].isnull().sum()
n_nan_sin_tablet    = df[df["v18q"] == 0]["v18q1"].isnull().sum()
print(f"    NaN en v18q1 cuando v18q=1 (tiene tablet) : {n_nan_con_tablet}")
print(f"    NaN en v18q1 cuando v18q=0 (no tiene)     : {n_nan_sin_tablet:,}")
print(f"    → Todos los NaN se explican por v18q=0: {n_nan_con_tablet == 0}")

print("\n  [PRUEBA 2.2b] v2a1 — NaN coincide con no-arrendatarios (tipovivi3=0)?")
rent_cross = df.groupby("tipovivi3")["v2a1"].agg(
    n_missing =lambda x: x.isnull().sum(),
    n_presente=lambda x: x.notnull().sum()
)
print(rent_cross.rename(index={0: "tipovivi3=0 (no arrienda)",
                                1: "tipovivi3=1 (arrienda)"}).to_string())
print("    → NaN ocurren solo en tipovivi3=0; imputar 0 es equivalente a no pago de arriendo.")

print("\n  [PRUEBA 2.2c] rez_esc — NaN es estructural por rango de edad?")
df["_age_group"] = pd.cut(df["age"], bins=[0, 5, 17, 65, 120],
                           labels=["0-5 años", "6-17 años", "18-65 años", "65+ años"])
rez_miss = df.groupby("_age_group", observed=True)["rez_esc"].apply(
    lambda x: f"{x.isnull().sum():,} NaN  ({x.isnull().mean()*100:.1f}%)"
)
print(rez_miss.to_string())
print("    → 100% NaN fuera del rango escolar (6-17). Imputar 0 = 'no aplica'.")
df.drop(columns=["_age_group"], inplace=True)

# ── v18q1 (número de tablets) ─────────────────────────────────────────────
# NaN aparece exclusivamente cuando v18q = 0 (el hogar no tiene tablet).
# Imputación: 0 (sin tablets).
assert df[df["v18q"] == 1]["v18q1"].isnull().sum() == 0, (
    "ERROR: NaN en v18q1 para hogares que declaran tener tablet."
)
df["v18q1"] = df["v18q1"].fillna(0)

# ── v2a1 (renta mensual) ──────────────────────────────────────────────────
# NaN cuando el hogar no paga arriendo (tipovivi3 = 0).
# Se crea indicador paga_arriendo antes de imputar para no perder esa señal.
df["paga_arriendo"] = df["v2a1"].notna().astype(int)
df["v2a1"] = df["v2a1"].fillna(0)

# ── rez_esc (retraso escolar) ─────────────────────────────────────────────
# NaN para edades no escolares (< 6 y > 17 años, ~83 % del total).
# Se crea indicador tiene_retraso para preservar la señal de niños en edad
# escolar que sí presentan retraso.
df["tiene_retraso"] = df["rez_esc"].notna().astype(int)
df["rez_esc"] = df["rez_esc"].fillna(0)

# ── meaneduc (promedio años educación adultos 18+) ────────────────────────
# Solo 5 filas; se imputa con la mediana del resto de la muestra.
df["meaneduc"] = df["meaneduc"].fillna(df["meaneduc"].median())

print(f"\n  Missing restantes : {df.isnull().sum().sum()}")
print(f"  Shape             : {df.shape}")

# -----------------------------------------------------------------------------
# 2.3  DUMMIES DE VARIANZA CASI CERO
# -----------------------------------------------------------------------------
# Se eliminan dummies con modo dominante ≥ 98 % y sin valor predictivo
# social documentado en la metodología NBI de Costa Rica.
# Las variables con baja varianza que SÍ son indicadores de carencia crítica
# se preservan explícitamente en la lista `preservar_nbi`.
# -----------------------------------------------------------------------------

print("\n" + "=" * 65)
print("2.3  Dummies de varianza casi cero")
print("=" * 65)

# ── Análisis sistemático de varianza en dummies ───────────────────────────
# Se calcula el porcentaje del modo dominante para cada variable binaria.
# El umbral de 98% identifica variables con información casi nula.
print("  [PRUEBA 2.3] Tabla completa de dummies con modo dominante ≥ 98%:")
dummy_cols = [
    c for c in df.columns
    if df[c].nunique() == 2
    and pd.to_numeric(df[c], errors="coerce").notna().all()
    and pd.to_numeric(df[c], errors="coerce").min() == 0
]
low_var_rows = []
for c in dummy_cols:
    pct_mode = df[c].value_counts(normalize=True).iloc[0] * 100
    if pct_mode >= 98:
        low_var_rows.append({"variable": c, "pct_modo_dominante": round(pct_mode, 2)})

low_var_df = pd.DataFrame(low_var_rows).sort_values("pct_modo_dominante", ascending=False)
print(low_var_df.to_string(index=False))
print(f"\n  Total detectadas: {len(low_var_df)} variables con varianza casi cero.")
print("  Las marcadas como NBI se preservan a pesar de la baja varianza (ver lista abajo).")

# Variables a eliminar: baja varianza, sin relevancia social directa
drop_baja_varianza = [
    "planpri",       # Electricidad de planta privada     (99.96 % modo)
    "pisoother",     # Piso: otro material                (99.93 %)
    "pisonatur",     # Piso: material natural             (99.86 %)
    "paredother",    # Pared: otro material               (99.85 %)
    "elimbasu6",     # Basura: otro método                (99.83 %)
    "elimbasu4",     # Basura: espacio vacío              (99.83 %)
    "energcocinar1", # Sin cocina                         (99.83 %)
    "techootro",     # Techo: otro material               (99.82 %)
    "paredfibras",   # Pared: fibras naturales            (99.81 %)
    "parentesco8",   # Suegro/a                           (99.78 %)
    "techocane",     # Techo: caña / fibras               (99.75 %)
    "sanitario6",    # Sanitario: otro sistema            (99.69 %)
    "parentesco10",  # Cuñado/a                           (99.67 %)
    "parentesco12",  # Otro no familiar                   (99.18 %)
    "parentesco5",   # Yerno / nuera                      (99.15 %)
    "paredzinc",     # Pared: zinc                        (99.03 %)
    "parentesco7",   # Madre / padre                      (98.96 %)
    "parentesco4",   # Hijastro/a                         (98.78 %)
    "parentesco11",  # Otro familiar                      (98.71 %)
    "parentesco9",   # Cuñado/a                           (98.62 %)
    "techoentrepiso",# Techo: fibrocemento / entrepiso    (98.34 %)
]

# Variables de baja varianza que SE CONSERVAN por relevancia
preservar_nbi = [
    "noelec",      # Sin electricidad                  → carencia básica
    "abastaguano", # Sin abasto de agua                → carencia básica
    "sanitario1",  # Sin inodoro                       → carencia básica
    "sanitario5",  # Inodoro a letrina / pozo negro    → precariedad rural
    "v14a",        # Tiene baño                        → indicador básico
    "instlevel6",  # Educación técnica incompleta      → capital humano
    "instlevel7",  # Educación técnica completa        → capital humano
    "instlevel9",  # Posgrado                          → indicador no-pobreza
    "tipovivi4",   # Vivienda en precario / tugurio    → precariedad extrema
    "pisonotiene", # Sin piso (tierra)                 → NBI Costa Rica
    "pareddes",    # Paredes de material de desecho    → NBI Costa Rica
]

drop_final_baja_var = [v for v in drop_baja_varianza if v not in preservar_nbi]
df.drop(columns=[c for c in drop_final_baja_var if c in df.columns], inplace=True)

print(f"  Eliminadas : {len(drop_final_baja_var)} variables")
print(f"  Preservadas (NBI): {len(preservar_nbi)} variables críticas")
print(f"  Shape      : {df.shape}")

# -----------------------------------------------------------------------------
# 2.4  TRATAMIENTO DE OUTLIERS
# -----------------------------------------------------------------------------
# Dos variables requieren corrección:
#   • v2a1 (renta mensual): cola derecha extrema (max = 1,000,000 CRC).
#     Se capea al percentil 99 para no distorsionar la frontera de pobreza.
#   • meaneduc (promedio años educación adultos): valor máximo = 37, imposible
#     dado que el máximo de escolari es 21. Es un error de captura. Se capea
#     al máximo teórico (21 años).
# -----------------------------------------------------------------------------

print("\n" + "=" * 65)
print("2.4  Tratamiento de outliers")
print("=" * 65)

cont_vars = ["age", "escolari", "meaneduc", "overcrowding",
             "hogar_total", "rooms", "bedrooms", "v2a1"]
print("  Estadísticas clave (pre-capping):")
print(
    df[cont_vars]
    .describe(percentiles=[.01, .05, .25, .5, .75, .95, .99]).T
    [["min", "1%", "5%", "50%", "95%", "99%", "max"]]
    .round(2)
    .to_string()
)

# ── Evidencia del outlier en meaneduc ────────────────────────────────────
# El máximo individual de escolari es 21 años; un promedio de hogar de 37
# es aritméticamente imposible y evidencia un error de captura de datos.
print(f"\n  [PRUEBA 2.4] meaneduc — valor máximo vs máximo teórico:")
print(f"    Max escolari (individual): {df['escolari'].max()} años")
print(f"    Max meaneduc (promedio)  : {df['meaneduc'].max()} años  ← imposible")
print(f"    Registros con meaneduc > {int(df['escolari'].max())}: "
      f"{(df['meaneduc'] > df['escolari'].max()).sum()}")
print(f"    → Se capea al máximo teórico para corregir el error de captura.")

# Capping v2a1 al percentil 99
p99_renta = df["v2a1"].quantile(0.99)
n_capped_renta = (df["v2a1"] > p99_renta).sum()
df["v2a1"] = df["v2a1"].clip(upper=p99_renta)
print(f"\n  v2a1    → capping al p99 ({p99_renta:,.0f} CRC)."
      f" Registros afectados: {n_capped_renta}")

# Capping meaneduc al máximo teórico de escolari (21 años)
max_teorico_educ = int(df["escolari"].max())  # 21
n_capped_educ = (df["meaneduc"] > max_teorico_educ).sum()
df["meaneduc"] = df["meaneduc"].clip(upper=max_teorico_educ)
print(f"  meaneduc → capping al máximo teórico ({max_teorico_educ} años)."
      f" Registros afectados: {n_capped_educ}")

# -----------------------------------------------------------------------------
# 2.5  ESTANDARIZACIÓN DE edjefe / edjefa
# -----------------------------------------------------------------------------
# Las columnas edjefe y edjefa contienen valores mixtos:
#   • Numérico : años de educación del jefe (p.ej. 6, 11, 17)
#   • "yes"    : tiene educación, años no registrados → se imputa como 1
#   • "no"     : sin educación formal                 → se imputa como 0
#
# Se crea ed_jefe_final = max(edjefe, edjefa) porque solo uno de los dos
# es el jefe de hogar efectivo.
# Se crea sin_jefe_educado como indicador de vulnerabilidad específica.
# -----------------------------------------------------------------------------

print("\n" + "=" * 65)
print("2.5  Estandarización de edjefe / edjefa")
print("=" * 65)

# ── Diagnóstico de datos mixtos ───────────────────────────────────────────
# Se muestra la distribución de valores antes de limpiar para documentar
# que las columnas contienen strings junto con valores numéricos.
print("  [PRUEBA 2.5a] Distribución de valores en edjefe (top 15, antes de limpiar):")
print(df["edjefe"].value_counts().head(15).to_string())

print("\n  [PRUEBA 2.5b] Distribución de valores en edjefa (top 10):")
print(df["edjefa"].value_counts().head(10).to_string())

n_yes_jefe = (df["edjefe"] == "yes").sum()
n_no_jefe  = (df["edjefe"] == "no").sum()
n_yes_jefa = (df["edjefa"] == "yes").sum()
n_no_jefa  = (df["edjefa"] == "no").sum()
print(f"\n  [PRUEBA 2.5c] Strings en edjefe: 'yes'={n_yes_jefe:,}  'no'={n_no_jefe:,}")
print(f"               Strings en edjefa: 'yes'={n_yes_jefa:,}  'no'={n_no_jefa:,}")
print("    → Presencia de strings confirma necesidad de estandarización.")

n_ambos_cero = (
    (pd.to_numeric(df["edjefe"], errors="coerce").fillna(0) == 0) &
    (pd.to_numeric(df["edjefa"], errors="coerce").fillna(0) == 0)
).sum()
print(f"\n  [PRUEBA 2.5d] Registros con edjefe=0 Y edjefa=0 (antes): {n_ambos_cero:,}")
print("    → Estos registros generan sin_jefe_educado=1 tras la limpieza.")


def limpiar_educacion_jefe(columna: pd.Series) -> pd.Series:
    """
    Convierte la columna de educación del jefe a numérico.
    'yes' → 1, 'no' → 0; cualquier otro no-numérico → 0.
    """
    return (
        pd.to_numeric(columna.replace({"yes": 1, "no": 0}), errors="coerce")
        .fillna(0)
    )


df["edjefe"] = limpiar_educacion_jefe(df["edjefe"])
df["edjefa"] = limpiar_educacion_jefe(df["edjefa"])

df["ed_jefe_final"] = df[["edjefe", "edjefa"]].max(axis=1)
df["sin_jefe_educado"] = ((df["edjefe"] == 0) & (df["edjefa"] == 0)).astype(int)

print(f"  Registros con jefe sin educación: {df['sin_jefe_educado'].sum():,}")
print(f"  Shape: {df.shape}")

# -----------------------------------------------------------------------------
# 2.6  REPARACIÓN DE HOGARES SIN JEFE REGISTRADO
# -----------------------------------------------------------------------------
# Un pequeño número de hogares no tiene ningún miembro con parentesco1 = 1.
# Se designa jefe de facto a quien tiene mayor escolaridad (idxmax), como
# criterio de capacidad de gestión del hogar. En caso de empate, idxmax
# toma el primer registro (usualmente la persona de mayor edad).
# Tras la asignación, se recalcula ed_jefe_final.
# -----------------------------------------------------------------------------

print("\n" + "=" * 65)
print("2.6  Reparación de hogares sin jefe registrado")
print("=" * 65)

hogares_con_jefe = set(df[df["parentesco1"] == 1]["idhogar"].unique())
todos_hogares    = set(df["idhogar"].unique())
hogares_sin_jefe = todos_hogares - hogares_con_jefe

for idhogar in hogares_sin_jefe:
    miembros    = df[df["idhogar"] == idhogar]
    idx_jefe    = miembros["escolari"].idxmax()
    df.at[idx_jefe, "parentesco1"] = 1

# Recalcular ed_jefe_final: si el jefe asignado tenía ed_jefe_final = 0
# (no era jefe original y no estaba en edjefe/edjefa), usamos su escolari
df["ed_jefe_final"] = df[["edjefe", "edjefa"]].max(axis=1)
df.loc[
    (df["parentesco1"] == 1) & (df["ed_jefe_final"] == 0),
    "ed_jefe_final"
] = df["escolari"]

print(f"  Hogares sin jefe detectados y corregidos: {len(hogares_sin_jefe)}")

# -----------------------------------------------------------------------------
# 2.7  RECONSTRUCCIÓN DE tasa_dependencia
# -----------------------------------------------------------------------------
# La variable original 'dependency' contiene datos mixtos:
#   • 4,238 valores numéricos (tasa calculada correctamente)
#   • 1,619 "yes"  (hay dependientes; el entrevistador no calculó la tasa)
#   • 1,330 "no"   (sin dependientes; equivale a tasa = 0)
#
# Decisión: reconstruir desde las variables demográficas del hogar, usando
# la fórmula original del INEC Costa Rica verificada contra los valores
# numéricos disponibles (correlación de Pearson = 1.0000):
#
#   tasa_dependencia = (hogar_nin + hogar_mayor) / (hogar_adul - hogar_mayor)
#
#   Numerador  : dependientes = niños 0–19 años + adultos 65+
#   Denominador: adultos en edad de trabajar (18–64)
#               = hogar_adul (18+) − hogar_mayor (65+)
#
# Caso borde: denominador = 0 (hogar compuesto solo de adultos mayores)
# → tasa = 8 (valor techo del dataset original, convención INEC).
# -----------------------------------------------------------------------------

print("\n" + "=" * 65)
print("2.7  Reconstrucción de tasa_dependencia")
print("=" * 65)

# ── Diagnóstico de datos mixtos en dependency ─────────────────────────────
dep_raw = df["dependency"]
n_num = pd.to_numeric(dep_raw, errors="coerce").notna().sum()
n_yes = (dep_raw == "yes").sum()
n_no  = (dep_raw == "no").sum()

print("  [PRUEBA 2.7a] Composición de tipos de valor en dependency:")
print(f"    Numérico : {n_num:,} ({n_num/len(dep_raw)*100:.1f}%)")
print(f"    'yes'    : {n_yes:,} ({n_yes/len(dep_raw)*100:.1f}%)")
print(f"    'no'     : {n_no:,}  ({n_no/len(dep_raw)*100:.1f}%)")
print("    → Datos mixtos: no se puede usar la variable directamente.")
print("    → 'no' confirmado como tasa=0 por reconstrucción (100% den=0 → tasa=0).")
print("    → 'yes' es información incompleta del entrevistador (tasa no calculada).")

# ── Validación de la fórmula de reconstrucción ────────────────────────────
# Se comparan dos candidatas contra los valores numéricos verificables.
print("\n  [PRUEBA 2.7b] Validación de fórmulas candidatas:")
dep_num = pd.to_numeric(dep_raw, errors="coerce")
mask    = dep_num.notna()
d_val   = df[mask].copy()
d_val["dep_orig"] = dep_num[mask]

# Fórmula A (INEC Costa Rica): (nin + mayor) / (adul - mayor)
d_val["fA"] = ((d_val["hogar_nin"] + d_val["hogar_mayor"]) /
               (d_val["hogar_adul"] - d_val["hogar_mayor"]).replace(0, np.nan))

# Fórmula B (alternativa simple): (nin + mayor) / adul
d_val["fB"] = ((d_val["hogar_nin"] + d_val["hogar_mayor"]) /
               d_val["hogar_adul"].replace(0, np.nan))

corr_A  = d_val["dep_orig"].corr(d_val["fA"])
corr_B  = d_val["dep_orig"].corr(d_val["fB"])
exact_A = (d_val["dep_orig"].round(4) == d_val["fA"].round(4)).sum()
exact_B = (d_val["dep_orig"].round(4) == d_val["fB"].round(4)).sum()
n_val   = mask.sum()

print(f"    Fórmula A = (nin+mayor)/(adul-mayor) :"
      f"  corr={corr_A:.4f}  |  coincidencias exactas={exact_A:,}/{n_val:,}")
print(f"    Fórmula B = (nin+mayor)/adul          :"
      f"  corr={corr_B:.4f}  |  coincidencias exactas={exact_B:,}/{n_val:,}")
print(f"    → Fórmula A seleccionada: correlación perfecta con valores originales.")
print(f"    → Los {n_val - exact_A} no coincidentes son hogares con denominador=0,")
print(f"      codificados como 8 en el dataset original (valor techo convencional).")
del d_val

num_dep = df["hogar_nin"] + df["hogar_mayor"]
den_dep = df["hogar_adul"] - df["hogar_mayor"]

df["tasa_dependencia"] = np.where(den_dep == 0, 8.0, num_dep / den_dep)
df.drop(columns=["dependency"], inplace=True)

print("  Distribución de tasa_dependencia:")
print(df["tasa_dependencia"].describe().round(3).to_string())
print(f"\n  Tasa = 0 (sin dependientes):              {(df['tasa_dependencia'] == 0).sum():>5,}")
print(f"  Tasa = 8 (solo adultos mayores, techo):   {(df['tasa_dependencia'] == 8).sum():>5,}")
print(f"  Tasa > 1 (más dependientes que activos):  {(df['tasa_dependencia'] > 1).sum():>5,}")

print("\n  Tasa media por nivel de pobreza (Target original):")
print(
    df.groupby("Target")["tasa_dependencia"]
    .mean().round(3)
    .rename({1: "Pobreza extrema (1)", 2: "Pobreza moderada (2)",
             3: "Vulnerable (3)",      4: "No pobre (4)"})
    .to_string()
)

# Visualización diagnóstica
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

# Histograma (excluye el valor techo 8 para ver la distribución real)
axes[0].hist(
    df[df["tasa_dependencia"] < 8]["tasa_dependencia"],
    bins=25, color="steelblue", edgecolor="white"
)
axes[0].axvline(1, color="crimson", linestyle="--", label="Equilibrio (tasa = 1)")
axes[0].set_title("Distribución tasa_dependencia\n(excluye valor techo = 8)")
axes[0].set_xlabel("Tasa de dependencia")
axes[0].set_ylabel("Frecuencia")
axes[0].legend(fontsize=8)

# Media por nivel de Target
media_target = df.groupby("Target")["tasa_dependencia"].mean()
colores_pob  = ["#d73027", "#fc8d59", "#fee090", "#91bfdb"]
bars = axes[1].bar(
    [f"Target {i}" for i in media_target.index],
    media_target.values,
    color=colores_pob, edgecolor="white"
)
for bar, v in zip(bars, media_target.values):
    axes[1].text(
        bar.get_x() + bar.get_width() / 2, v + 0.02,
        f"{v:.2f}", ha="center", va="bottom", fontsize=9
    )
axes[1].set_title("Media por nivel de pobreza")
axes[1].set_ylabel("Tasa de dependencia media")

# Boxplot
df.boxplot(
    column="tasa_dependencia", by="Target", ax=axes[2],
    patch_artist=True,
    boxprops=dict(facecolor="steelblue", alpha=0.6),
    medianprops=dict(color="crimson", linewidth=2)
)
axes[2].set_title("Dispersión por nivel de pobreza")
axes[2].set_xlabel("Target  (1 = extrema  →  4 = no pobre)")
axes[2].set_ylabel("Tasa de dependencia")
plt.suptitle("")
plt.tight_layout()
os.makedirs(DIR_VIZS, exist_ok=True)
plt.savefig(f"{DIR_VIZS}/fig_tasa_dependencia.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"\n  Shape: {df.shape}")

# -----------------------------------------------------------------------------
# 2.8  ELIMINACIÓN DE VARIABLES ADICIONALES
# -----------------------------------------------------------------------------
# Se eliminan variables que:
#   a) Son totales/subtotales demográficos ya resumidos en hogar_nin/adul/mayor
#   b) Corresponden a tipos de parentesco con muy baja prevalencia
#      (los más raros ya fueron eliminados en 2.3; aquí se depura el resto,
#       conservando solo parentesco1 = jefe de hogar)
#   c) Son niveles educativos intermedios capturados por meaneduc continuo
#      EXCEPCIÓN: instlevel1 (sin educación) SE PRESERVA — es indicador
#      NBI directo con correlación confirmada con pobreza.
#   d) Son estados civiles sin asociación específica a vulnerabilidad
#      PRESERVADOS: estadocivil6 (viudo/a) y estadocivil7 (soltero/a),
#      por su relación con hogares monoparentales y riesgo económico.
#   e) Son dummies espejo que generan multicolinealidad perfecta en modelos
#      lineales (categoría de referencia).
# -----------------------------------------------------------------------------

print("\n" + "=" * 65)
print("2.8  Eliminación de variables adicionales")
print("=" * 65)

# ── Pruebas de relevancia de variables preservadas ────────────────────────
# Se documenta empíricamente por qué instlevel1, estadocivil6 y estadocivil7
# se mantienen en lugar de eliminarse junto con las demás variables del grupo.
# Un Target menor indica mayor pobreza (1=extrema, 4=no pobre).

print("  [PRUEBA 2.8a] instlevel1 (sin educación formal) vs nivel de pobreza:")
print(f"    Target medio — instlevel1=0 (tiene algún nivel): "
      f"{df[df['instlevel1']==0]['Target'].mean():.3f}")
print(f"    Target medio — instlevel1=1 (sin educación)    : "
      f"{df[df['instlevel1']==1]['Target'].mean():.3f}")
print(f"    Prevalencia en muestra: {df['instlevel1'].mean()*100:.1f}% de registros")
print("    → Target menor en instlevel1=1 confirma asociación con mayor pobreza.")
print("    → Se preserva como predictor NBI directo (no capturado por meaneduc).")

print("\n  [PRUEBA 2.8b] estadocivil6 (viudo/a) vs nivel de pobreza:")
print(f"    Target medio — estadocivil6=0: {df[df['estadocivil6']==0]['Target'].mean():.3f}")
print(f"    Target medio — estadocivil6=1: {df[df['estadocivil6']==1]['Target'].mean():.3f}")
print(f"    Prevalencia: {df['estadocivil6'].mean()*100:.1f}%")

print("\n  [PRUEBA 2.8c] estadocivil7 (soltero/a) vs nivel de pobreza:")
print(f"    Target medio — estadocivil7=0: {df[df['estadocivil7']==0]['Target'].mean():.3f}")
print(f"    Target medio — estadocivil7=1: {df[df['estadocivil7']==1]['Target'].mean():.3f}")
print(f"    Prevalencia: {df['estadocivil7'].mean()*100:.1f}%")
print("    → Diferencia en Target justifica preservar ambas variables de estado civil.")

drop_adicionales = [
    # a) Conteos demográficos desagregados
    #    (r4h3, r4m3, r4t3 ya eliminados en 2.1)
    "r4h1", "r4h2",   # Hombres < 12 años / ≥ 12 años
    "r4m1", "r4m2",   # Mujeres < 12 años / ≥ 12 años
    "r4t1", "r4t2",   # Total < 12 años  / ≥ 12 años

    # b) Parentesco — solo se conserva parentesco1 (jefe de hogar)
    #    (parentesco4,5,7,8,9,10,11,12 ya eliminados en 2.3)
    "parentesco2",  # Cónyuge / pareja
    "parentesco3",  # Hijo/a
    "parentesco6",  # Nieto/a

    # c) Niveles educativos intermedios (capturados por meaneduc)
    #    instlevel1 (sin educación) PRESERVADA — predictor NBI directo
    "instlevel2",  # Primaria incompleta
    "instlevel3",  # Primaria completa
    "instlevel4",  # Secundaria académica incompleta
    "instlevel5",  # Secundaria académica completa
    "instlevel8",  # Universitaria (capturada por meaneduc alto)

    # d) Estado civil — solo se eliminan los no asociados a vulnerabilidad
    #    estadocivil6 (viudo/a)   PRESERVADA
    #    estadocivil7 (soltero/a) PRESERVADA
    "estadocivil1",  # < 10 años (capturado por hogar_nin)
    "estadocivil2",  # Unión libre (categoría de referencia)
    "estadocivil3",  # Casado/a   (no asociado a vulnerabilidad específica)
    "estadocivil4",  # Divorciado/a
    "estadocivil5",  # Separado/a

    # e) Dummies espejo (multicolinealidad perfecta)
    "area2",   # Complemento exacto de area1 (urbano)
    "epared3", # Pared buena → referencia; epared1/2 capturan la variación
    "etecho3", # Techo bueno → referencia
    "eviv3",   # Piso bueno  → referencia
]

n_antes = df.shape[1]
df.drop(columns=[c for c in drop_adicionales if c in df.columns], inplace=True)
n_eliminadas = n_antes - df.shape[1]

print(f"  Eliminadas : {n_eliminadas} variables")
print(f"  Shape      : {df.shape}")
print("\n  Variables recuperadas respecto al notebook original:")
print("    • instlevel1  (sin educación — predictor NBI directo)")
print("    • estadocivil6 (viudo/a — hogar sin segundo ingreso)")
print("    • estadocivil7 (soltero/a — potencial hogar monoparental)")

# =============================================================================
# 3. BINARIZACIÓN DEL TARGET
# =============================================================================
# Regla de binarización:
#   Target ≤ 2  →  1  (POBRE: pobreza extrema o moderada)
#   Target  > 2  →  0  (NO POBRE: vulnerable o no pobre)
#
# Se aplica la regla de mayoría a nivel de hogar para garantizar consistencia:
# todos los miembros de un mismo hogar deben recibir el mismo Target.
# Cuando hay empate (50 % - 50 %), se asigna 1 (pobre) para minimizar
# falsos negativos y proteger a hogares en situación límite.
# =============================================================================

print("\n" + "=" * 65)
print("3. BINARIZACIÓN DEL TARGET")
print("=" * 65)

print("  Distribución original (individuos):")
print(
    df["Target"]
    .value_counts(normalize=True).mul(100).round(2)
    .rename({1: "Pobreza extrema (1)", 2: "Moderada (2)",
             3: "Vulnerable (3)",      4: "No pobre (4)"})
    .to_string()
)

# Paso 1: binarización individual
df["Target_bin"] = (df["Target"] <= 2).astype(int)

# Paso 2: regla de mayoría por hogar, con desempate pro-pobre
target_por_hogar = (
    df.groupby("idhogar")["Target_bin"]
    .agg(lambda x: x.mode().max())  # .max() → en empate 0-1 elige 1
    .to_dict()
)
df["Target_final"] = df["idhogar"].map(target_por_hogar)

# Diagnóstico: hogares con Target inconsistente entre miembros
inconsistentes = (df.groupby("idhogar")["Target_bin"].nunique() > 1).sum()
cambiados = sum(
    1 for h in df["idhogar"].unique()
    if df.loc[df["idhogar"] == h, "Target_bin"].iloc[0] != target_por_hogar[h]
)

print(f"\n  Hogares con Target inconsistente entre miembros : {inconsistentes}")
print(f"  Hogares cuyo Target cambia con la regla de mayoría: {cambiados}")

df.drop(columns=["Target", "Target_bin"], inplace=True)
df.rename(columns={"Target_final": "Target"}, inplace=True)

print("\n  Distribución final (individuos, tras regla de mayoría):")
print(
    df["Target"]
    .value_counts(normalize=True).mul(100).round(2)
    .rename({0: "No pobre (0)", 1: "Pobre (1)"})
    .to_string()
)

# =============================================================================
# 4. AGREGACIÓN AL NIVEL DE HOGAR (idhogar)
# =============================================================================
# El dataset original tiene un registro por individuo. El programa social
# focaliza hogares, no personas, por lo que se colapsa a un registro por hogar.
#
# Reglas de agregación por tipo de variable:
#   • female           → mean   : proporción de mujeres en el hogar.
#                                  (max daría presencia/ausencia, no proporción)
#   • tasa_dependencia → first  : ya es una variable de nivel hogar; todos los
#                                  miembros comparten el mismo valor.
#   • v2a1, bedrooms, overcrowding, activos del hogar
#                      → max    : valor representativo del hogar (es el mismo
#                                  para todos los miembros en el dataset original)
#   • escolari, meaneduc, age, composición del hogar
#                      → mean   : perfil promedio de los miembros del hogar.
#   • Resto de dummies → max    : el hogar "tiene" la característica si al menos
#                                  un miembro la presenta.
#   • Target           → first  : ya homogeneizado en paso 3.
# =============================================================================

print("\n" + "=" * 65)
print("4. AGREGACIÓN AL NIVEL DE HOGAR")
print("=" * 65)

agg_rules = {}

VARS_MAX  = {"v2a1", "rooms", "bedrooms", "overcrowding", "ed_jefe_final",
             "sin_jefe_educado", "paga_arriendo", "refrig", "computer",
             "television", "rez_esc"}
VARS_MEAN = {"escolari", "meaneduc", "age",
             "hogar_nin", "hogar_adul", "hogar_mayor", "hogar_total"}

for col in df.columns:
    if col in ("idhogar", "Target"):
        continue
    elif col == "female":
        agg_rules[col] = "mean"          # proporción de mujeres
    elif col == "tasa_dependencia":
        agg_rules[col] = "first"         # variable ya a nivel hogar
    elif col in VARS_MAX:
        agg_rules[col] = "max"
    elif col in VARS_MEAN:
        agg_rules[col] = "mean"
    else:
        agg_rules[col] = "max"           # dummies de infraestructura/ubicación

df_hogar = (
    df.groupby("idhogar")
    .agg({**agg_rules, "Target": "first"})
    .reset_index()
)

print(f"  Individuos → hogares : {df.shape[0]:,} → {df_hogar.shape[0]:,}")
print(f"  Tamaño medio de hogar: {df.shape[0] / df_hogar.shape[0]:.1f} personas")
print(f"  Shape dataset hogares: {df_hogar.shape}")
print(f"\n  Distribución del Target en hogares:")
print(
    df_hogar["Target"]
    .value_counts(normalize=True).mul(100).round(2)
    .rename({0: "No pobre (0)", 1: "Pobre (1)"})
    .to_string()
)

# =============================================================================
# 5. DIAGNÓSTICO DE MULTICOLINEALIDAD (VIF)
# =============================================================================
# El VIF (Variance Inflation Factor) mide cuánto se infla la varianza de un
# coeficiente por la correlación con otras variables.
#   VIF = ∞   → multicolinealidad perfecta (columna linealmente dependiente)
#   VIF > 10  → redundancia alta
#
# Este análisis es diagnóstico: guía las eliminaciones del paso 6.
# =============================================================================

print("\n" + "=" * 65)
print("5. DIAGNÓSTICO DE MULTICOLINEALIDAD (VIF)")
print("=" * 65)

X_vif = df_hogar.select_dtypes(include=[np.number]).drop(
    columns=["Target"], errors="ignore"
)

vif_data = pd.DataFrame({
    "variable": X_vif.columns,
    "VIF": [
        variance_inflation_factor(X_vif.values, i)
        for i in range(len(X_vif.columns))
    ]
})

n_inf  = np.isinf(vif_data["VIF"]).sum()
n_alto = (vif_data["VIF"] > 10).sum()

print(f"  Variables con VIF = ∞ (multicolinealidad perfecta) : {n_inf}")
print(f"  Variables con VIF > 10 (alta redundancia)          : {n_alto}")

if n_inf > 0:
    print("\n  Variables con VIF infinito:")
    vars_inf = vif_data[np.isinf(vif_data["VIF"])]["variable"].tolist()
    for v in vars_inf:
        print(f"    • {v}")

# =============================================================================
# 6a. RESOLUCIÓN DE MULTICOLINEALIDADES PERFECTAS (post-VIF)
# =============================================================================
# Se elimina una variable de cada grupo de dummies que suman exactamente 1
# (trampa de variables ficticias), y se eliminan redundancias confirmadas.
#
# Referencias usadas:
#   • Región        → lugar6 (Huetar Norte) eliminada como referencia
#   • Agua          → abastaguadentro eliminada como referencia
#   • Tenencia      → tipovivi1 (propia, pagada) eliminada como referencia;
#                     tipovivi3 (arrendada) también eliminada para romper
#                     el ciclo de VIF infinito en el grupo de tenencia
# =============================================================================

print("\n" + "=" * 65)
print("6a. Resolución de multicolinealidades perfectas")
print("=" * 65)

drop_post_vif = [
    # ── Composición del hogar ─────────────────────────────────────────────
    "hogar_total",   # Suma exacta de hogar_nin + hogar_adul → VIF = ∞
    "parentesco1",   # Constante = 1 para todos los hogares tras la agregación

    # ── Educación del jefe ────────────────────────────────────────────────
    "edjefe", "edjefa",   # Consolidadas en ed_jefe_final

    # ── Trampa de dummies: región (referencia = Huetar Norte) ─────────────
    "lugar6",

    # ── Trampa de dummies: acceso a agua (referencia = agua dentro) ────────
    "abastaguadentro",

    # ── Trampa de dummies: tenencia de vivienda ───────────────────────────
    "tipovivi1",   # Referencia: propia y pagada
    "tipovivi3",   # Arrendada (eliminada para romper ciclo VIF = ∞)

    # ── Redundancias de activos ───────────────────────────────────────────
    "paga_arriendo",  # Redundancia perfecta con v2a1 (v2a1 > 0 ↔ paga_arriendo = 1)
    "v18q",           # Redundante con v18q1 (cantidad de tablets ≥ 1 ↔ v18q = 1)
    "public",         # Electricidad pública: redundante con el conjunto de categorías

    # ── Energía para cocinar ──────────────────────────────────────────────
    "energcocinar2",  # Gas de cilindro   (alta correlación con NSE)
    "energcocinar3",  # Gas de red/elec.  (alta correlación con NSE)

    # ── Identificador individual ──────────────────────────────────────────
    "Id",   # Sin valor predictivo a nivel de hogar
]

df_model = df_hogar.drop(
    columns=[c for c in drop_post_vif if c in df_hogar.columns]
)

print(f"  Eliminadas: {len([c for c in drop_post_vif if c in df_hogar.columns])} variables")
print(f"  Shape: {df_model.shape}")

# Verificación post-corrección
X_check = df_model.select_dtypes(include=[np.number]).drop(
    columns=["Target"], errors="ignore"
)
vif_check = pd.DataFrame({
    "variable": X_check.columns,
    "VIF": [
        variance_inflation_factor(X_check.values, i)
        for i in range(len(X_check.columns))
    ]
})

n_inf_post = np.isinf(vif_check["VIF"]).sum()
print(f"\n  VIF infinitos restantes: {n_inf_post}")
if n_inf_post == 0:
    print("  Top 5 VIF tras corrección:")
    print(
        vif_check.sort_values("VIF", ascending=False)
        .head(5)
        .to_string(index=False)
    )

# =============================================================================
# 6b. REDUCCIÓN DE DIMENSIONALIDAD POST-AGREGACIÓN
# =============================================================================
# Tras colapsar al nivel de hogar, algunas variables pierden poder explicativo
# o quedan redundadas por versiones más parsimoniosas.
#
# DECISIÓN DELIBERADA sobre pareddes y pisonotiene:
#   Se PRESERVAN. Aunque la versión anterior del código las eliminaba bajo el
#   supuesto de que estaban implícitas en epared1 (Pared_Mala) y eviv1
#   (Piso_Malo), hay una diferencia importante:
#     • pareddes / pisonotiene : medición OBJETIVA del material específico
#     • epared1  / eviv1       : evaluación SUBJETIVA del entrevistador
#   En la metodología NBI de Costa Rica, los materiales son el criterio
#   oficial, no la evaluación subjetiva. Se mantienen ambas dimensiones.
# =============================================================================

print("\n" + "=" * 65)
print("6b. Reducción de dimensionalidad post-agregación")
print("=" * 65)

drop_post_agr = [
    "v14a",    # Tiene baño: redundante con los tipos específicos de sanitario
               # (genera VIF alto al coexistir con sanitario1, sanitario2, etc.)
    "mobilephone",   # Varianza casi cero; Cant_Celulares (qmobilephone) es más
                     # informativo al capturar la intensidad, no solo la presencia
    "rooms",   # Total de habitaciones: redundante con bedrooms para medir
               # capacidad física del hogar; bedrooms es más preciso para NBI
    "rez_esc", # Retraso escolar continuo: sustituido por tiene_retraso (binaria)
               # que es más robusta tras la imputación de 0 en edades no escolares
    "hacdor",  # Hacinamiento por dormitorios: capturado por overcrowding
               # (personas por cuarto, variable continua más informativa)
    "hacapo",  # Hacinamiento por cuartos: ídem anterior
    "ed_jefe_final",  # Años de educación del jefe: colineal con meaneduc
                      # (promedio educación adultos), que es más robusto al
                      # agregar información de todos los adultos del hogar
    # pareddes  (Pared_Desecho) → PRESERVADA (ver nota arriba)
    # pisonotiene (Piso_Tierra) → PRESERVADA (ver nota arriba)
]

n_antes_6b = df_model.shape[1]
df_model.drop(
    columns=[c for c in drop_post_agr if c in df_model.columns],
    inplace=True
)
n_elim_6b = n_antes_6b - df_model.shape[1]

print(f"  Eliminadas: {n_elim_6b} variables")
print(f"  Shape: {df_model.shape}")
print("\n  Preservadas intencionalmente:")
print("    • pareddes   (Pared_Desecho) — material objetivo, indicador NBI")
print("    • pisonotiene (Piso_Tierra)  — material objetivo, indicador NBI")

# =============================================================================
# 7. RENOMBRADO DE VARIABLES A NOMBRES DESCRIPTIVOS
# =============================================================================

print("\n" + "=" * 65)
print("7. RENOMBRADO DE VARIABLES")
print("=" * 65)

diccionario_nombres = {
    # ── Vivienda y activos ────────────────────────────────────────────────
    "v2a1":         "Monto_Alquiler",
    "v18q1":        "Cantidad_Tablets",
    "refrig":       "Tiene_Nevera",
    "computer":     "Tiene_Computador",
    "television":   "Tiene_TV",
    "qmobilephone": "Cant_Celulares",
    "bedrooms":     "Total_Dormitorios",
    "overcrowding": "Personas_por_Cuarto",

    # ── Educación ─────────────────────────────────────────────────────────
    "escolari":         "Promedio_Anos_Escolaridad",
    "meaneduc":         "Promedio_Educ_Adultos",
    "instlevel1":       "Sin_Educacion",          # predictor NBI preservado
    "instlevel6":       "Educ_Tecnica_Incompleta",
    "instlevel7":       "Educ_Tecnica_Completa",
    "instlevel9":       "Educ_Postgrado",
    "sin_jefe_educado": "Hogar_sin_Jefe_Educado",
    "tiene_retraso":    "Hogar_con_Rezago_Escolar",

    # ── Estado civil ──────────────────────────────────────────────────────
    "estadocivil6": "Jefe_Viudo",     # hogar sin segundo ingreso
    "estadocivil7": "Jefe_Soltero",   # potencial hogar monoparental

    # ── Materiales de construcción: paredes ───────────────────────────────
    "paredblolad": "Pared_Bloque_Ladrillo",
    "paredzocalo": "Pared_Zocalo",
    "paredpreb":   "Pared_Prefabricado",
    "pareddes":    "Pared_Desecho",      # NBI objetivo — preservado
    "paredmad":    "Pared_Madera",

    # ── Materiales de construcción: piso ──────────────────────────────────
    "pisomoscer":  "Piso_Mosaico_Ceramica",
    "pisocemento": "Piso_Cemento",
    "pisonotiene": "Piso_Tierra",        # NBI objetivo — preservado
    "pisomadera":  "Piso_Madera",

    # ── Materiales de construcción: techo ─────────────────────────────────
    "techozinc":  "Techo_Zinc",
    "cielorazo":  "Tiene_Cielorazo",

    # ── Estado general de la infraestructura (evaluación subjetiva) ───────
    "epared1": "Pared_Mala",    "epared2": "Pared_Regular",
    "etecho1": "Techo_Malo",    "etecho2": "Techo_Regular",
    "eviv1":   "Piso_Malo",     "eviv2":   "Piso_Regular",

    # ── Servicios básicos ─────────────────────────────────────────────────
    "abastaguafuera": "Agua_Fuera_Vivienda",
    "abastaguano":    "Sin_Abasto_Agua",
    "noelec":         "Sin_Electricidad",
    "coopele":        "Electricidad_Cooperativa",
    "sanitario1":     "Sin_Inodoro",
    "sanitario2":     "Inodoro_Alcantarillado",
    "sanitario3":     "Inodoro_Septico",
    "sanitario5":     "Inodoro_Letrina",
    "energcocinar4":  "Cocina_Carbon_Lena",
    "elimbasu1":      "Basura_Camion",
    "elimbasu2":      "Basura_Enterrada",
    "elimbasu3":      "Basura_Quemada",
    "elimbasu5":      "Basura_Rio_Creek",

    # ── Demografía ────────────────────────────────────────────────────────
    "dis":          "Tiene_Discapacitado",
    "female":       "Proporcion_Mujeres",    # mean en agregación
    "hogar_nin":    "Cant_Ninos",
    "hogar_adul":   "Cant_Adultos",
    "hogar_mayor":  "Cant_Adultos_Mayores",
    "age":          "Edad_Promedio",
    "tasa_dependencia": "Tasa_Dependencia",

    # ── Tenencia y ubicación ──────────────────────────────────────────────
    "tipovivi2": "Casa_Propia_Pagando",
    "tipovivi4": "Casa_Precario",
    "tipovivi5": "Casa_Prestada_Asignada",
    "lugar1":    "Region_Central",
    "lugar2":    "Region_Chorotega",
    "lugar3":    "Region_Pacifico_Central",
    "lugar4":    "Region_Brunca",
    "lugar5":    "Region_Huetar_Atlantica",
    "area1":     "Zona_Urbana",
}

df_model.rename(columns=diccionario_nombres, inplace=True)

print(f"  Variables en el dataset final: {len(df_model.columns)}")
print(f"  Shape: {df_model.shape}")
print("\n  Columnas finales:")
for i, col in enumerate(df_model.columns, 1):
    marker = " ←" if col in ("Pared_Desecho", "Piso_Tierra",
                              "Sin_Educacion", "Jefe_Viudo", "Jefe_Soltero",
                              "Tasa_Dependencia") else ""
    print(f"    {i:3d}.  {col}{marker}")
print("\n  ← Variables recuperadas respecto al notebook original")

# =============================================================================
# 8. EXPORTACIÓN
# =============================================================================
# El guardado se realiza al final del pipeline, después de todas las
# transformaciones (secciones 2–7), para que el archivo refleje el estado
# definitivo del dataset: limpio, agregado por hogar y renombrado.
# =============================================================================

print("\n" + "=" * 65)
print("8. EXPORTACIÓN")
print("=" * 65)

os.makedirs(DIR_DATOS, exist_ok=True)

output_path = f"{DIR_DATOS}/train_cleaned_hogar.csv"
df_model.to_csv(output_path, index=False)

print(f"  Archivo guardado : {output_path}")
print(f"  Filas            : {df_model.shape[0]:,}")
print(f"  Columnas         : {df_model.shape[1]}")
print("\n  Distribución final del Target:")
print(
    df_model["Target"]
    .value_counts(normalize=True).mul(100).round(2)
    .rename({0: "No pobre (0)", 1: "Pobre (1)"})
    .to_string()
)
print("\n" + "=" * 65)
print("  Pipeline completado exitosamente.")
print("=" * 65)
