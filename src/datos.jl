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



# Número de sensores (variables de entrada) que debe tener cada fila
const NUM_SENSORES = 128



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

# Formato de línea:     clase;concentración 1:valor 2:valor ... 128:valor
# Devolvemos:           (clase, concentración, vector de valores) igual que en el fichero
function leerLinea(linea::AbstractString, ruta::String, numLinea::Int)
    partes = split(linea, ';')
    if length(partes) != 2
        error("Linea sin el formato esperado 'clase;concentración ...' eno $ruta, línea $numLinea")
    end

    clase = parse(Int, partes[1])

    campos = split(partes[2]) # Separamos por espacios
    concentración = parse(Float64, campos[1])

    pares = split.(campos[2:end], ':') # Separamos por ':'
    # Para cada valor (i:valor), sacamos ["i", "valor"]
    indices = [parse(Int, p[1]) for p in pares]
    valores = [parse(Float64, p[2]) for p in pares]

    # Comprobación de la estructura (sensores del 1..128 en orden)
    if indices != collect(1:NUM_SENSORES)
        error("Los índices de los sensores no son 1..$NUM_SENSORES en $ruta, línea $numLinea")
    end

    return(clase, concentración, valores)
end

# Lectura de un fichero batch completo, transformando cada línea en un vector de valores
# Devolvemos un DataFrame con las columnas: batch, clase, concentración, valores (vector de 128 elementos)
# Una fila por cada línea, en el orden del fichero
function cargarBatch(ruta::String)
    lineas = readlines(ruta)
    filas = [leerLinea(l, ruta, i) for (i, l) in enumerate(lineas) if !isempty(strip(l))]
    n = length(filas)
    # Matriz n x 128 con los valores de los sensores
    valores = [f[3] for f in filas] # filas es un vector de n tuplas (clase, concentración, valores).
    sensores = [valores[i][j] for i in 1:n, j in 1:NUM_SENSORES] # i recorre las mediciones (1:n), j recorre los sensores (1:128)
    # El elemento (i, j) de la matriz es el sensor j de la medición i, así que sale
    meta = DataFrame(
        batch = fill(numeroBatch(basename(ruta)), n), 
        clase = [f[1] for f in filas], 
        concentración = [f[2] for f in filas]
    )
    return hcat(meta, DataFrame(sensores, ["x$i" for i in 1:NUM_SENSORES]))
    # ["x$i" for i in 1:NUM_SENSORES] son los nombres de columnas: x1, x2, ..., x128
    # Dataframe(sensores, nombres) conviertte la matriz en una tabla con las columnas
    # hcat(meta, ...) concatena las columnas de meta y de sensores en un único DataFrame
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
    println("Filas: ", nrow(datos), " | Variables de sensores: ", count(startswith("x"), names(datos)))
    # names(datos) da los nombres de las columnas (contamos cuántas del vector empiezan por "x" == sensores)
    println("\nFilas por batch:")
    show(stdout, sort(combine(groupby(datos, :batch), nrow => :filas), :batch); allrows = true)
    # groupby(datos, :batch) agrupa las filas por batch; combine(..., nrow => :filas) cuenta cuántas hay en cada grupo (columna "filas")
    # sort(..., :batch) las ordena por batch; show(...; allrows = true) imprime la tabla entera sin recortar filas
    println("\nFilas por clase:")
    show(stdout, sort(combine(groupby(datos, :clase), nrow => :filas), :clase); allrows = true)
    # Igual que la tabla anterior, pero agrupando por clase: cuántas filas hay de cada clase (1 a 6)
    println()
end
