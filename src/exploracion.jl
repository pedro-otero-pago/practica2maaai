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

# Por batch y característica: celdas con signo incorrecto, filas afectadas y máximo de sensores en una fila
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

# Correlación (log absDR) entre cada pareja de sensores gemelos, por batch
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

# Mediciones con al menos `umbral` sensores sin respuesta (DR ≤ 0)
function filasNoResponden(datos::DataFrame, umbral::Int)
    nDR = vec(sum(Matrix(datos[!, columnasDe("DR")]) .<= 0, dims = 2))
    sel = nDR .>= umbral
    res = datos[sel, [:orden, :batch, :clase, :concentracion]]
    res.sensoresSinRespuesta = nDR[sel]
    return res
end

const NOMBRES_CLASES = ["Etanol", "Etileno", "Amoníaco", "Acetaldehído", "Acetona", "Tolueno"]

# Descripción básica del dataset y de la partición holdout
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

# Filas en las que un sensor responde más de `umbral` veces distinto que su gemelo (cociente de absDR)
function divergenciaGemelos(datos::DataFrame, umbral::Real = 3.0)
    res = DataFrame(orden = Int[], batch = Int[], clase = Int[], concentracion = Float64[],
                    pareja = String[], ratio = Float64[])
    for p in 1:2:NUM_SENSORES
        a = datos[!, "S$(lpad(p, 2, '0'))_absDR"]
        b = datos[!, "S$(lpad(p + 1, 2, '0'))_absDR"]
        r = a ./ b
        for i in findall((r .> umbral) .| (r .< 1 / umbral))
            push!(res, (datos.orden[i], datos.batch[i], datos.clase[i], datos.concentracion[i],
                        "S$p-S$(p+1)", round(r[i], digits = 2)))
        end
    end
    return res
end

# Mediana de absDR por concentración (debe crecer con la concentración)
function respuestaPorConcentracion(datos::DataFrame, batch::Int, clase::Int, sensores)
    d = datos[(datos.batch .== batch) .& (datos.clase .== clase), :]
    cols = ["S$(lpad(s, 2, '0'))_absDR" for s in sensores]
    return sort(combine(groupby(d, :concentracion), nrow => :filas,
                        [c => median => c for c in cols]...), :concentracion)
end

# Para un gas y una concentración, en cada batch: cuántas filas tienen algún sensor sin respuesta
function noRespuestaPorBatch(datos::DataFrame, clase::Int, conc::Real)
    sel = (datos.clase .== clase) .& (datos.concentracion .== conc)
    n = vec(sum(Matrix(datos[sel, columnasDe("DR")]) .<= 0, dims = 2))
    t = DataFrame(batch = datos.batch[sel], n = n)
    return combine(groupby(t, :batch), nrow => :filas,
                   :n => (x -> count(>(0), x)) => :filasConAlguno, :n => maximum => :max)
end

# Compara unas mediciones (por su orden) con una referencia, variable a variable
function compararConReferencia(datos::DataFrame, ordenes, ref::DataFrame, columnas)
    sel = datos[in.(datos.orden, Ref(Set(ordenes))), :]
    s = [median(sel[!, c]) for c in columnas]
    r = [median(ref[!, c]) for c in columnas]
    return DataFrame(variable = columnas, seleccion = round.(s, digits = 2),
                     referencia = round.(r, digits = 2), cociente = round.(s ./ r, digits = 2))
end

# absDR ≈ 1 va con DR ≈ 0 (absDR es una razón: 1 = sin cambio)
function comprobacionAbsDR(datos::DataFrame)
    x = vcat([datos[!, "S$(lpad(s, 2, '0'))_absDR"] for s in 1:NUM_SENSORES]...)
    y = vcat([datos[!, "S$(lpad(s, 2, '0'))_DR"] for s in 1:NUM_SENSORES]...)
    cerca = abs.(x .- 1) .< 0.01
    println("Celdas con absDR ≈ 1: ", count(cerca),
            " | mediana |DR| en ellas: ", round(median(abs.(y[cerca])), digits = 1),
            " | mediana |DR| global: ", round(median(abs.(y)), digits = 1))
    println("Filas donde 'absDR < 1' no coincide con 'DR < 0' (sensor 1): ",
            count((datos.S01_DR .< 0) .!= (datos.S01_absDR .< 1)))
end