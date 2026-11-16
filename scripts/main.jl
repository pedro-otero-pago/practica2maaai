# ============================================================
#  Práctica 2 — MAAAI · main.jl
#  Ejecución: julia --project=environment scripts/main.jl | tee results/log.txt
# ============================================================

# --- 0. Entorno -------------------------------------------------
using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "environment"))  
Pkg.instantiate()   # instala lo del Project.toml si falta
  

using Random
Random.seed!(104)

# --- Rutas --------------------------------------------------------
const RAIZ    = dirname(@__DIR__)
const DATOS   = joinpath(RAIZ, "data")
const RESULTS = joinpath(RAIZ, "results")
mkpath(RESULTS)

# --- Código del proyecto ------------------------------------------
# Se irá ampliando en cada bloque:
include(joinpath(RAIZ, "src", "datos.jl"))
include(joinpath(RAIZ, "src", "exploracion.jl"))
include(joinpath(RAIZ, "src", "preprocesado", "limpieza.jl"))
include(joinpath(RAIZ, "src", "preprocesado", "holdout.jl"))
# include(joinpath(RAIZ, "src", "estadistica.jl"))

# --- Utilidad para el log -----------------------------------------
function fase(n, titulo)
    println("\n", "="^60)
    println("FASE $n — $titulo")
    println("="^60)
end

# --- Fases (orden del enunciado) ----------------------------------
fase(1, "Carga y preparación de los datos")
datos = cargarDatos(DATOS) # Se aplica la carga de datos sin tratamiento
informeDatos(datos)

# idxTrain, idxTest = holdoutTemporal(datos)
# println("Holdout: train = ", length(idxTrain), " | test = ", length(idxTest))
# println("Batches en test: ", sort(unique(datos.batch[idxTest])))
# println("Test empieza en la fila ", minimum(idxTest), " (orden global)")

fase(2, "Análisis exploratorio")
println("Violaciones del signo esperado (celdas) por batch:")
show(stdout, violacionesSigno(datos); allrows = true, allcols = true)
println("\n\nResumen de violaciones (filas afectadas y sensores por fila):")
show(stdout, resumenViolaciones(datos); allrows = true)
println("\n\nCorrelación entre sensores gemelos (log absDR) por batch:")
show(stdout, correlacionGemelos(datos); allrows = true, allcols = true)
println("\n\nMediciones con ≥ 12 sensores sin respuesta:")
show(stdout, filasNoResponden(datos, 12); allrows = true)
println("\n\nVariables con ceros exactos: ", cerosPorVariable(datos))

limpio = eliminarDefectuosas(datos)
println("\nMediciones defectuosas eliminadas: ", nrow(datos) - nrow(limpio),
        " | Quedan: ", nrow(limpio))

fase(3, "Tratamiento de valores perdidos")
println("Pendiente (B1, Pedro): ventana de fallo de los sensores 3–4 → ausentes")

# Holdout definitivo: sobre los datos ya limpios (reglas fijas), antes de imputar
idxTrain, idxTest = holdoutTemporal(limpio)
println("\nHoldout: train = ", length(idxTrain), " | test = ", length(idxTest))
descripcionDataset(limpio, idxTrain, idxTest)

fase(4, "Construcción de pipelines")
println("Pendiente (B2)")

fase(5, "Entrenamiento y evaluación de modelos")
println("Pendiente (B2/B3)")

fase(6, "Validación experimental")
println("Pendiente (B2/B3)")

fase(7, "Visualización de resultados")
println("Pendiente (B4)")

fase(8, "Análisis comparativo")
println("Pendiente (B3/B5)")

fase(9, "Conclusiones parciales")
println("Pendiente (B5)")