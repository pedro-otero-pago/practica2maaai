# Práctica 2 — MAAAI

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
- `src/`: código del proyecto, un fichero por componente
- `results/`: figuras, tablas y log de ejecución
- `memoria/`: memoria en LaTeX