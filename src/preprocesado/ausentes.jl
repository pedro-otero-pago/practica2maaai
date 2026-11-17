# ============================================================
#  Práctica 2 — MAAAI · ausentes.jl
#  Análisis de valores ausentes (fase 3 de main.jl)
#  Criterio: todo cero exacto es un valor ausente.
# ============================================================

# Copia de los datos en la que los ceros exactos de las 128 variables pasan a missing
function marcarAusentes(datos::DataFrame)
    res = copy(datos)
    for v in NOMBRES_VARIABLES
        res[!, v] = replace(res[!, v], 0.0 => missing)
    end
    return res
end

# Número de ausentes en cada fila
ausentesPorFila(datosNA::DataFrame) = [count(ismissing, r) for r in eachrow(datosNA[!, NOMBRES_VARIABLES])]

# Porcentaje de ausentes por variable (solo las que tienen alguno) y porcentaje global
function porcentajeAusentes(datosNA::DataFrame)
    n = [count(ismissing, datosNA[!, v]) for v in NOMBRES_VARIABLES]
    porVariable = DataFrame(variable = NOMBRES_VARIABLES, ausentes = n,
                            pct = round.(100 .* n ./ nrow(datosNA), digits = 3))
    porVariable = sort(porVariable[porVariable.ausentes .> 0, :], :ausentes, rev = true)
    pctGlobal = 100 * sum(n) / (nrow(datosNA) * NUM_VARIABLES)
    return porVariable, pctGlobal
end

# Por batch: filas, filas con algún ausente y celdas ausentes
function ausentesPorBatch(datosNA::DataFrame)
    t = DataFrame(batch = datosNA.batch, n = ausentesPorFila(datosNA))
    return sort(combine(groupby(t, :batch), nrow => :filas,
                        :n => (x -> count(>(0), x)) => :filasAfectadas,
                        :n => sum => :celdasAusentes), :batch)
end

# Filas con ausentes: contexto (batch, gas, concentración) y si la fase 2 ya las elimina
function filasConAusentes(datosNA::DataFrame)
    n = ausentesPorFila(datosNA)
    res = datosNA[n .> 0, [:orden, :batch, :clase, :concentracion]]
    res.ausentes = n[n .> 0]
    res.eliminada = in.(res.orden, Ref(MEDICIONES_DEFECTUOSAS))
    return res
end

# Patrones: qué variables faltan juntas, y en cuántas filas se repite cada patrón
function patronesAusentes(datosNA::DataFrame)
    n = ausentesPorFila(datosNA)
    afectadas = datosNA[n .> 0, NOMBRES_VARIABLES]
    patrones = DataFrame(patron = [join(NOMBRES_VARIABLES[findall(ismissing, collect(r))], ", ")
                                   for r in eachrow(afectadas)])
    return sort(combine(groupby(patrones, :patron), nrow => :filas), :filas, rev = true)
end