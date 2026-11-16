# ============================================================
#  Práctica 2 — MAAAI · exploracion.jl
#  Caracterización de los sensores (fase 2 de main.jl)
# ============================================================

using Statistics

# Función que devuelve los 16 nombres de una caracteristica
columnasDe(caracteristica) = ["S$(lpad(s, 2, '0'))_$caracteristica" for s in 1:NUM_SENSORES]

# Signo esperado de cada caracteristica
const SIGNO_ESPERADO = Dict("DR" => 1, "absDR" => 1,
    "EMAi0001" => 1, "EMAi001" => 1, "EMAi01" => 1,
    "EMAd0001" => -1, "EMAd001" => -1, "EMAd01" => -1)

# Recuento de cuantas caracteristicas tienen el signo incorrecto o valen 0
function violacionesSigno(datos::DataFrame)
    tabla = DataFrame(batch = 1:10)
    for c in CARACTERISTICAS
        s = SIGNO_ESPERADO[c]
        tabla[!, c] = [count(x -> s * x <= 0, Matrix(datos[datos.batch .== b, columnasDe(c)])) for b in 1:10]
    end
    return tabla
end

# Ceros exactos por variable
function cerosPorVariable(datos::DataFrame)
    ceros = [(v, count(==(0.0), datos[!, v])) for v in NOMBRES_VARIABLES]
    return filter(t -> t[2] > 0, ceros)
end

function resumenViolaciones(datos::DataFrame)
    res = DataFrame(batch = Int[], caracteristica = String[], celdas = Int[],
                    filas = Int[], maxSensoresEnUnaFila = Int[])
    for b in 1:10, c in CARACTERISTICAS
        s = SIGNO_ESPERADO[c]
        M = Matrix(datos[datos.batch .== b, columnasDe(c)])
        porFila = vec(sum(s .* M .<= 0, dims = 2))     # sensores con violación en cada fila
        if sum(porFila) > 0
            push!(res, (b, c, sum(porFila), count(>(0), porFila), maximum(porFila)))
        end
    end
    return res
end

function correlacionGemelos(datos::DataFrame)
    tabla = DataFrame(batch = 1:10)
    for p in 1:2:NUM_SENSORES                      # p = 1, 3, 5, …, 15
        a = "S$(lpad(p, 2, '0'))_absDR"
        b = "S$(lpad(p + 1, 2, '0'))_absDR"
        tabla[!, "S$(p)-S$(p+1)"] = [round(cor(log.(datos[datos.batch .== k, a]),
                                                 log.(datos[datos.batch .== k, b])), digits = 3) for k in 1:10]
    end
    return tabla
end

function filasNoResponden(datos::DataFrame, umbral::Int)
    nDR = vec(sum(Matrix(datos[!, columnasDe("DR")]) .<= 0, dims = 2))
    sel = nDR .>= umbral
    res = datos[sel, [:orden, :batch, :clase, :concentracion]]
    res.sensoresSinRespuesta = nDR[sel]
    return res
end

const NOMBRES_CLASES = ["Etanol", "Etileno", "Amoníaco", "Acetaldehído", "Acetona", "Tolueno"]

function descripcionDataset(datos::DataFrame, idxTrain, idxTest)
    println("Instancias: ", nrow(datos), " | Variables: ", NUM_VARIABLES,
            " (", NUM_SENSORES, " sensores × ", NUM_CARACTERISTICAS, " características)",
            " | Batches: ", length(unique(datos.batch)), " | Clases: ", length(unique(datos.clase)))
    println("Clases: ", join(["$i = $(NOMBRES_CLASES[i])" for i in 1:6], ", "))
    println("Características por sensor: DR (variación de resistencia), absDR (variación normalizada),",
            " EMAi0001/EMAi001/EMAi01 (subida, α = 0.001/0.01/0.1),",
            " EMAd0001/EMAd001/EMAd01 (bajada, α = 0.001/0.01/0.1)")
    println("Batches en entrenamiento: ", sort(unique(datos.batch[idxTrain])))
    println("Batches en test:          ", sort(unique(datos.batch[idxTest])))
    println("Batch 10: ", count(==(10), datos.batch[idxTrain]), " filas en entrenamiento y ",
            count(==(10), datos.batch[idxTest]), " en test")
end