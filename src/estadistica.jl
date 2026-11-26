# =============================================================
#  estadistica.jl — Comparación estadística de modelos sobre los
#  mismos folds (balanced accuracy por fold): t pareado y Wilcoxon
# =============================================================
using HypothesisTests, DataFrames, Statistics

"""
Compara por parejas las configuraciones de `tabla` (salida de evaluarRejilla
con columnas :nombre y :balanced_fold). Mismos folds → tests pareados.
Con 5 folds, el p mínimo de Wilcoxon es 0,0625: nunca es significativo al 5 %.
"""
function compararModelos(tabla::DataFrame; alfa = 0.05)
    res = DataFrame(modeloA = String[], modeloB = String[], difMedia = Float64[],
                    p_t = Float64[], p_wilcoxon = Float64[], significativo = Bool[])
    n = nrow(tabla)
    for i in 1:n-1, j in i+1:n
        a, b = tabla.balanced_fold[i], tabla.balanced_fold[j]
        d    = a .- b
        if all(iszero, d)                       # mismos valores en todos los folds
            pt, pw = 1.0, 1.0
        else
            pt = pvalue(OneSampleTTest(a, b))   # t pareado (sobre las diferencias)
            pw = pvalue(SignedRankTest(a, b))   # Wilcoxon pareado
        end
        push!(res, (tabla.nombre[i], tabla.nombre[j], mean(d), pt, pw, pt < alfa))
    end
    return res
end