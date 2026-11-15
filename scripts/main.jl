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
# include(joinpath(RAIZ, "src", "estadistica.jl"))

# --- Utilidad para el log -----------------------------------------
function fase(n, titulo)
    println("\n", "="^60)
    println("FASE $n — $titulo")
    println("="^60)
end

# --- Fases (orden del enunciado) ----------------------------------
fase(1, "Carga y preparación de los datos")
datos = cargarDatos(DATOS) # Se aplica la carga de datos sin trartamiento
informeDatos(datos)
println("\nPendiente (B1): holdout y CV por batch")

fase(2, "Análisis exploratorio")
println("Pendiente (B1)")

fase(3, "Tratamiento de valores perdidos")
println("Pendiente (B1)")

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