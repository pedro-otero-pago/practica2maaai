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
include(joinpath(RAIZ, "src", "preprocesado", "ausentes.jl"))
include(joinpath(RAIZ, "src", "preprocesado", "imputacion.jl"))
include(joinpath(RAIZ, "src", "preprocesado", "holdout.jl"))
include(joinpath(RAIZ, "src", "validacion.jl"))
include(joinpath(RAIZ, "src", "preprocesado", "minmax.jl"))
include(joinpath(@__DIR__, "..", "src", "modelos", "red.jl"))
include(joinpath(@__DIR__, "..", "src", "filtros", "anova.jl"))
include(joinpath(@__DIR__, "..", "src", "filtros", "mi.jl"))
include(joinpath(@__DIR__, "..", "src", "experimentos.jl"))
include(joinpath(@__DIR__, "..", "src", "modelos", "knn.jl"))
include(joinpath(@__DIR__, "..", "src", "filtros", "rfe.jl"))
include(joinpath(@__DIR__, "..", "src", "modelos", "svm.jl"))
include(joinpath(@__DIR__, "..", "src", "modelos", "reduccion.jl"))
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

println("\n\nabsDR como razón de resistencias:")
comprobacionAbsDR(datos)

println("\nEtanol, batch 2: respuesta (mediana absDR) por concentración")
show(stdout, respuestaPorConcentracion(datos, 2, 1, (1, 7, 15)); allrows = true)

println("\n\nEtanol a 10 ppmv: filas con algún sensor sin respuesta, por batch")
show(stdout, noRespuestaPorBatch(datos, 1, 10.0); allrows = true)

refVentana = datos[(datos.batch .== 2) .& (datos.clase .== 2) .& (datos.concentracion .== 250.0) .&
                   .!in.(datos.orden, Ref(825:833)), :]
println("\n\nVentana 825–833 frente a etileno 250 ppmv del batch 2 (absDR de los 16 sensores):")
show(stdout, compararConReferencia(datos, 825:833, refVentana, columnasDe("absDR")); allrows = true)
println("\n\nVentana 825–833: las 8 características de los sensores 3 y 4")
show(stdout, compararConReferencia(datos, 825:833, refVentana,
     vcat(["S03_$c" for c in CARACTERISTICAS], ["S04_$c" for c in CARACTERISTICAS])); allrows = true)

ref9894 = datos[(datos.batch .== 9) .& (datos.clase .== 6) .& (datos.concentracion .== 10.0) .&
                (datos.orden .!= 9894), :]
println("\n\nFila 9894 frente a tolueno 10 ppmv del batch 9:")
show(stdout, compararConReferencia(datos, [9894], ref9894, columnasDe("absDR")); allrows = true)

println("\n\nGemelos divergentes (absDR > 3 veces distinto), por batch y pareja:")
show(stdout, sort(combine(groupby(divergenciaGemelos(datos), [:batch, :pareja]), nrow => :filas,
     :orden => minimum => :desde, :orden => maximum => :hasta), [:batch, :pareja]); allrows = true)

println("\n\nFilas con formato alternativo (final del batch 10, en test) frente a filas normales")
println("del batch 10 con el mismo gas y concentración (absDR). Cocientes entre 1/3 y 3 → válidas:")
for cl in (4, 5)
    alt    = datos[datos.formatoAlternativo .& (datos.clase .== cl), :]
    refAlt = datos[(datos.batch .== 10) .& .!datos.formatoAlternativo .& (datos.clase .== cl) .&
                   in.(datos.concentracion, Ref(Set(alt.concentracion))), :]
    println("\nClase $cl ($(NOMBRES_CLASES[cl])): ", nrow(alt), " filas alternativas frente a ",
            nrow(refAlt), " de referencia")
    show(stdout, compararConReferencia(datos, alt.orden, refAlt, columnasDe("absDR")); allrows = true)
    println()
end 

limpio = eliminarDefectuosas(datos)
println("\nMediciones defectuosas eliminadas: ", nrow(datos) - nrow(limpio),
        " | Quedan: ", nrow(limpio))

fase(3, "Tratamiento de valores perdidos")

# Análisis sobre los datos ORIGINALES (los 55 ceros), no sobre `limpio`
datosNA = marcarAusentes(datos)
pctVar, pctGlobal = porcentajeAusentes(datosNA)
println("Ausentes por variable:")
show(stdout, pctVar; allrows = true)
println("\n\nPorcentaje global de ausentes: ", round(pctGlobal, digits = 4), " %")
println("\nAusentes por batch:")
show(stdout, ausentesPorBatch(datosNA); allrows = true)
println("\n\nFilas con ausentes (eliminada = ya la quita la limpieza de la fase 2):")
show(stdout, filasConAusentes(datosNA); allrows = true)
println("\n\nPatrones de ausencia (variables que faltan a la vez):")
show(stdout, patronesAusentes(datosNA); allrows = true, truncate = 0)

# Regla fija, ANTES del holdout: los ceros que quedan tras la limpieza pasan a missing
limpio = marcarAusentes(limpio)
println("\n\nAusentes que quedan tras la limpieza: ", sum(ausentesPorFila(limpio)))

# Holdout definitivo: sobre los datos ya limpios (reglas fijas), antes de imputar
idxTrain, idxTest = holdoutTemporal(limpio)
println("\nHoldout: train = ", length(idxTrain), " | test = ", length(idxTest))
descripcionDataset(limpio, idxTrain, idxTest)
# 5 fold CV por batches (solo sobre entrenamiento)
folds = foldsPorBatch(limpio.batch[idxTrain])
println("\n5 fold CV por batches:")
show(stdout, comprobarFolds(folds, limpio.batch[idxTrain], limpio.clase[idxTrain]); allrows = true)
println()

# Comprobación del imputador KNN (ajustado SOLO con entrenamiento)
Xtrain = limpio[idxTrain, NOMBRES_VARIABLES]
Xtest  = limpio[idxTest,  NOMBRES_VARIABLES]
maqImp = machine(ImputadorKNN(k = 5), Xtrain)
fit!(maqImp, verbosity = 0)
XtestImp = DataFrame(MLJ.transform(maqImp, Xtest))

println("\nImputador KNN (k = 5): ausentes en test antes = ", sum(ausentesPorFila(Xtest)),
        " | después = ", sum(ausentesPorFila(XtestImp)))

# Fila 13481 (tolueno 1 ppmv, en test): valor imputado frente a filas parecidas de entrenamiento
i = findfirst(==(13481), limpio.orden[idxTest])
tr = limpio[idxTrain, :]
sel = (tr.clase .== 6) .& (tr.concentracion .== 1.0)
println("S05_DR imputado en la fila 13481: ", round(XtestImp[i, :S05_DR], digits = 2),
        " | mediana de S05_DR en tolueno 1 ppmv (train): ", round(median(skipmissing(tr.S05_DR[sel])), digits = 2))

# Comprobación del Min-Max
XtrainImp = DataFrame(MLJ.transform(maqImp, Xtrain))
maqMM = machine(NormalizadorMinMax(), XtrainImp)
fit!(maqMM, verbosity = 0)
Ztrain = Matrix(DataFrame(MLJ.transform(maqMM, XtrainImp)))
Ztest = Matrix(DataFrame(MLJ.transform(maqMM, XtestImp)))
println("\nMin-Max, entrenamiento: mínimo = ", minimum(Ztrain), " | máximo = ", maximum(Ztrain))
println("Min-Max, test (fuera de [0, 1] por la deriva): mínimo = ",
        round(minimum(Ztest), digits = 3), " | máximo = ", round(maximum(Ztest), digits = 3))


fase(4, "Construcción de pipelines")
yTodo = categorical(limpio.clase)              # niveles comunes a entrenamiento y test
X = limpio[idxTrain, NOMBRES_VARIABLES]        # con los missing: se imputan dentro de la red
y = yTodo[idxTrain]
println("Red: imputador KNN (k = 5) → Min–Max → [filtro] → [reducción] → clasificador")
println("Validación: 5 folds por batches | Métricas: balanced_accuracy (selección) y accuracy")

fase(5, "Entrenamiento y evaluación de modelos")
println("KNN (K = 1, 10, 20) sin filtro y con filtros ANOVA y MI (k = 16, 32, 64):")
resKNN = experimentosKNN(X, y, folds)
show(resKNN[:, [:filtro, :configuracion, :balanced_acc, :accuracy]]; allrows = true, truncate = 0)
println()

println("\nSVM (C = 0.1, 0.5, 1.0) sin filtro y con RFE (k = 16, 32, 64):")
resSVM = experimentosSVM(X, y, folds)
show(resSVM[:, [:filtro, :configuracion, :balanced_acc, :accuracy]]; allrows = true, truncate= 0)
println()

println("\nKNN(K = 10) con PCA, LDA e ICA:")
Random.seed!(104) # FastICA parte de una matriz aleatoria: fijamos una semilla fija para reproducir el log
resRed = experimentosReduccion(X, y, folds)
show(resRed[:, [:reduccion, :configuracion, :balanced_acc, :accuracy]]; allrows = true, truncate = 0)
println()

fase(6, "Validación experimental")
println("Pendiente (B2/B3)")

fase(7, "Visualización de resultados")
println("Pendiente (B4)")

fase(8, "Análisis comparativo")
println("Pendiente (B3/B5)")

fase(9, "Conclusiones parciales")
println("Pendiente (B5)")