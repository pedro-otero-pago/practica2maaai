# ============================================================
#  Práctica 2 — MAAAI · datos.jl
#  Ejecución: Se incluye desde scripts/main.jl
# ============================================================

# Lectura de ficheros de datos

# No se realiza ninguna técnica de tratamiento de datos, solo se leen 
# y se devuelven en el formato adecuado.

# Los tratamientos se realizarán en el módulo de análisis de datos
# habiendo sido discutidos y valorados por el grupo previamente, para
# ser justificados y documentados en el informe final.

using DataFrames



# El array tiene 16 sensores y de cada uno se dan 8 características (128 variables)
const NUM_SENSORES = 16
const NUM_CARACTERISTICAS = 8
const NUM_VARIABLES = NUM_SENSORES * NUM_CARACTERISTICAS # 128

# Características de cada sensor, en el orden en que aparecen en el fichero
const CARACTERISTICAS = ["DR", "absDR", "EMAi0001", "EMAi001", "EMAi01", "EMAd0001", "EMAd001", "EMAd01"]

# Nombres de las 128 columnas: S01_DR, S01_absDR, ..., S16_EMAd01
const NOMBRES_VARIABLES = ["S$(lpad(s, 2, '0'))_$c" for s in 1:NUM_SENSORES for c in CARACTERISTICAS]
@assert length(NOMBRES_VARIABLES) == NUM_VARIABLES



# === Ficheros === (batchN.dat --> N)
function numeroBatch(nombre::String)
    return parse(Int, filter(isdigit, nombre))
end

# Rutas de los ficheros batch ordenadas por número
# Reordena los ficheros batch por número de batch (batchN.dat --> N)
function ficherosBatch(carpeta::String)
    nombres = filter(f -> startswith(f, "batch") && endswith(f, ".dat"), readdir(carpeta))
    nombres = sort(nombres, by = numeroBatch)
    return [joinpath(carpeta, n) for n in nombres]
end



# === Lectura de ficheros batch ===

# Formato de línea: clase;concentracion 1:valor 2:valor ... 128:valor
# Devolvemos: (clase, concentracion, vector de valores, formatoAlternativo)
function leerLinea(linea::AbstractString, ruta::String, numLinea::Int)
    partes = split(linea, ';')
    if length(partes) != 2
        error("Linea sin el formato esperado 'clase;concentracion ...' en $ruta, línea $numLinea")
    end

    clase = parse(Int, partes[1])

    campos = split(partes[2]) # Separamos por espacios
    concentracion = parse(Float64, campos[1])
    formatoAlternativo = !('.' in campos[1]) # Concentración sin punto decimal == formato alternativo (ej. "600" en vez de "600.000000")

    pares = split.(campos[2:end], ':') # Separamos por ':'
    # Para cada valor (i:valor), sacamos ["i", "valor"]
    indices = [parse(Int, p[1]) for p in pares]
    valores = [parse(Float64, p[2]) for p in pares]

    # Comprobación de la estructura (variables del 1..128 en orden)
    if indices != collect(1:NUM_VARIABLES)
        error("Los índices de las variables no son 1..$NUM_VARIABLES en $ruta, línea $numLinea")
    end

    return(clase, concentracion, valores, formatoAlternativo)
end

# Lectura de un fichero batch completo, transformando cada línea en un vector de valores
# Devolvemos un DataFrame con las columnas: batch, clase, concentracion, formatoAlternativo y las 128 variables (S01_DR ... S16_EMAd01)
# Una fila por cada línea, en el orden del fichero
function cargarBatch(ruta::String)
    lineas = readlines(ruta)
    filas = [leerLinea(l, ruta, i) for (i, l) in enumerate(lineas) if !isempty(strip(l))]
    n = length(filas)
    # Matriz n x 128 con las variables de cada medición
    valores = [f[3] for f in filas] # filas es un vector de n tuplas (clase, concentracion, valores, formatoAlternativo).
    variables = [valores[i][j] for i in 1:n, j in 1:NUM_VARIABLES] # i recorre las mediciones (1:n), j recorre las variables (1:128)
    # El elemento (i, j) de la matriz es la variable j de la medición i
    meta = DataFrame(
        batch = fill(numeroBatch(basename(ruta)), n), 
        clase = [f[1] for f in filas], 
        concentracion = [f[2] for f in filas],
        formatoAlternativo = [f[4] for f in filas]
    )
    return hcat(meta, DataFrame(variables, NOMBRES_VARIABLES))
    # DataFrame(variables, NOMBRES_VARIABLES) convierte la matriz en una tabla con una columna por variable
    # hcat(meta, ...) concatena las columnas de meta y de variables en un único DataFrame
end




# Carga de datos (batches de la carpeta) en orden numérico, unidos en un único DataFrame
function cargarDatos(carpeta::String)
    rutas = ficherosBatch(carpeta)
    if isempty(rutas)
        error("No se han encontrado ficheros batchN.dat en la carpeta $carpeta")
    end
    datos = reduce(vcat, [cargarBatch(r) for r in rutas])
    insertcols!(datos, 1, :orden => collect(1:nrow(datos))) # Añadimos columna de orden
    return datos
end



# Informe de datos cargados (meramente informativo)
# No modifica los datos, solo imprime información por pantalla
function informeDatos(datos::DataFrame)
    println("Filas: ", nrow(datos), " | Variables: ", count(in(NOMBRES_VARIABLES), names(datos)))
    # names(datos) da los nombres de las columnas (contamos cuántas están en NOMBRES_VARIABLES)
    println("Filas en formato alternativo: ", count(datos.formatoAlternativo)," (batches: ", sort(unique(datos.batch[datos.formatoAlternativo])), ")")
    println("\nFilas por batch:")
    show(stdout, sort(combine(groupby(datos, :batch), nrow => :filas), :batch); allrows = true)
    println()
    # groupby(datos, :batch) agrupa las filas por batch; combine(..., nrow => :filas) cuenta cuántas hay en cada grupo (columna "filas")
    # sort(..., :batch) las ordena por batch; show(...; allrows = true) imprime la tabla entera sin recortar filas
    println("\nFilas por clase:")
    show(stdout, sort(combine(groupby(datos, :clase), nrow => :filas), :clase); allrows = true)
    # Igual que la tabla anterior, pero agrupando por clase: cuántas filas hay de cada clase (1 a 6)
    println()
end
