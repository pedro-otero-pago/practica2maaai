# ============================================================
#  Práctica 2 — MAAAI · setup_entorno.jl
#  Crea environment/Project.toml y environment/Manifest.toml.
#  Se ejecuta UNA vez, desde la raíz del repositorio:
#      julia scripts/setup_entorno.jl
#  El resto del equipo NO lo ejecuta: tras hacer pull basta con
#  main.jl, que llama a Pkg.instantiate().
# ============================================================

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "environment"))

# Paquetes agrupados por bloque del plan.
const PAQUETES = [
    # --- B0/B1 · datos, estadística y utilidades ---
    "DataFrames", "CategoricalArrays", "StatsBase", "Statistics",
    "Random", "HypothesisTests", "Printf",
    # --- MLJ (núcleo + modelos base) ---
    "MLJ", "MLJBase", "MLJModelInterface",
    "MLJLinearModels",                 # regresión logística (RFE)
    "NearestNeighborModels",           # KNN
    "LIBSVM", "MLJLIBSVMInterface",    # SVM
    "MLJFlux", "Flux",                 # MLP
    "MLJMultivariateStatsInterface",   # PCA, LDA, ICA
    "MultivariateStats",
    # --- B3 · ensembles ---
    "MLJDecisionTreeInterface",        # Random Forest, AdaBoostStumpClassifier
    "EvoTrees",                        # EvoTreeClassifier
    "XGBoost", "MLJXGBoostInterface",
    "LightGBM",
    "CatBoost",
    # --- B4 · visualización ---
    "Plots",
]

fallidos = String[]
for p in PAQUETES
    println(">>> Añadiendo $p")
    try
        Pkg.add(p)
    catch e
        println("!!! No se pudo añadir $p: ", sprint(showerror, e))
        push!(fallidos, p)
    end
end

Pkg.precompile()

println("\n", "="^60)
if isempty(fallidos)
    println("Entorno creado sin errores.")
else
    println("Paquetes que fallaron (revisar a mano): ", join(fallidos, ", "))
end
println("Versión de Julia: ", VERSION)
Pkg.status()
