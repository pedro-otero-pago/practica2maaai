# ============================================================
#  Práctica 2 — MAAAI · limpieza.jl
#  Reglas fijas de limpieza, aplicadas ANTES del holdout.
#  Se derivan del análisis de la fase 2 (exploracion.jl).
# ============================================================

# Mediciones defectuosas (se eliminan): la medición entera no es válida.
#  - 669:678   batch 2, etanol 500–600 ppmv: episodio sin respuesta del array
#  - 2722:2726 batch 3, etanol 10 ppmv: episodio sin respuesta
#              (en el resto del dataset el etanol a 10 ppmv responde siempre)
#  - 3271:3275 batch 3, acetona 100 ppmv: últimas 5 mediciones del batch, sin respuesta
#  - 287       batch 1: EMAd con signo invertido en los 16 sensores
#  - 4595      batch 6: respuesta invertida en los 16 sensores
const MEDICIONES_DEFECTUOSAS = Set(vcat(287, 669:678, 2722:2726, 3271:3275, 4595))

# Ventana de fallo de la pareja de sensores 3–4 (batch 2, etileno 250 ppmv).
# El resto del array funciona: sus variables se marcan como ausentes y se imputan
# (lo hace la parte de valores ausentes).
const VENTANA_FALLO_S3S4 = 825:833
const SENSORES_VENTANA   = [3, 4]

# Elimina las mediciones defectuosas (identificadas por su orden original)
eliminarDefectuosas(datos::DataFrame) = datos[.!in.(datos.orden, Ref(MEDICIONES_DEFECTUOSAS)), :]