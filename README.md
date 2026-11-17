# Práctica 2 — MAAAI

## Requisitos
- Julia 1.12.4 (comprobar con `julia --version`).
- Conexión a internet en la primera ejecución.

## Primera ejecución
La primera vez, `main.jl` llama a `Pkg.instantiate()`, que descarga, instala y precompila
todos los paquetes del entorno, incluido Python para CatBoost (en `environment/.CondaPkg`,
ignorado por git). Este paso tarda unos 35–40 minutos. Las siguientes ejecuciones no lo repiten.

## Ejecución
Desde la raíz del repositorio:

    julia --project=environment scripts/main.jl

Para guardar el log (Windows / Linux):

    julia --project=environment scripts/main.jl | Tee-Object -FilePath results/log.txt
    julia --project=environment scripts/main.jl | tee results/log.txt

## Estructura
- `data/`: batches originales (batch1.dat … batch10.dat)
- `environment/`: entorno de Julia (Project.toml, Manifest.toml)
- `scripts/main.jl`: script principal, ejecuta todas las fases
- `scripts/setup_entorno.jl`: se usó una sola vez para crear el entorno. **No forma parte de la ejecución** y no hace falta lanzarlo.
- `src/`: código del proyecto, un fichero por componente
- `results/`: figuras, tablas y log de ejecución
- `memoria/`: memoria en LaTeX
- `memoria/secciones/`: una sección de la memoria por archivo