# ============================================================
#  Práctica 2 — MAAAI · limpieza.jl
#  Reglas fijas de limpieza, aplicadas ANTES del holdout.
#  Se derivan del análisis de la fase 2 (exploracion.jl).
#  Criterio (indicado por el profesor): una medición con más del 10 %
#  de variables inválidas o ausentes se elimina entera.
# ============================================================

# Mediciones defectuosas (se eliminan): la medición entera no es válida.
#  - 663:678   batch 2, etanol 500–600 ppmv: toda la campaña es defectuosa
#              (respuesta hundida o absurda frente a la tendencia de 10–275 ppmv)
#  - 825:833   batch 2, etileno 250–300 ppmv: ventana de fallo de los sensores 3 y 4
#              (sensor 3 sin respuesta, sensor 4 errático: 16 de 128 variables
#               inválidas = 12,5 % > 10 %)
#  - 2722:2726 batch 3, etanol 10 ppmv: episodio sin respuesta
#              (en el resto del dataset el etanol a 10 ppmv responde siempre)
#  - 3271:3275 batch 3, acetona 100 ppmv: últimas 5 mediciones del batch, sin respuesta
#  - 287       batch 1: EMAd con signo invertido en los 16 sensores
#  - 4595      batch 6: respuesta invertida en los 16 sensores
#  - 9894      batch 9, tolueno 10 ppmv: ≥ 3 sensores con respuesta > 3 veces distinta
#              de su referencia (18,75 % > 10 %)
const MEDICIONES_DEFECTUOSAS = Set(vcat(287, 663:678, 825:833, 2722:2726, 3271:3275, 4595, 9894))

# Elimina las mediciones defectuosas (identificadas por su orden original)
eliminarDefectuosas(datos::DataFrame) = datos[.!in.(datos.orden, Ref(MEDICIONES_DEFECTUOSAS)), :]