#!/usr/bin/env bash
# =============================================================================
# Invariantes del proyecto. Rápidos, sin GPU salvo donde se indica.
#
#   bash tests/run.sh
#
# Qué NO es esto: un smoke test. `scripts/smoke_test.sh` corre las 7 etapas con
# datos sintéticos y comprueba que el pipeline no revienta. Esto comprueba que
# no está devolviendo respuestas equivocadas sin reventar, que es peor.
#
# Cada test de aquí guarda un fallo que YA ocurrió y que NO daba síntomas:
# ni excepción, ni warning, ni número raro. Solo un resultado plausible y falso.
#
# Los que necesitan `pyg-lib` (el sampler nativo) se SALTAN en macOS con aviso,
# no fallan: allí no se entrena.
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."

fallos=0
saltados=0

correr() {
    local nombre="$1" archivo="$2" nota="${3:-}"
    echo "── $nombre ${nota:+($nota)}"
    salida=$(python3 "$archivo" 2>&1)
    codigo=$?
    echo "$salida" | sed 's/^/  /'
    if [ $codigo -ne 0 ]; then
        # El sampler nativo no existe en macOS: no es un fallo del código.
        if echo "$salida" | grep -q "sampler nativo"; then
            echo "  -> SALTADO (necesita pyg-lib; córrelo en el pod)"
            saltados=$((saltados + 1))
        else
            fallos=$((fallos + 1))
        fi
    fi
    echo
}

correr "E0 · el embedding de vecinos contiene vecinos" \
       tests/test_embedding_vecinos.py
correr "E1 · la primera transacción de una entidad no recibe de ella" \
       tests/test_grado_minimo_entidad.py "necesita el grafo construido"
correr "E2 · el grafo tiene las aristas que dicen los datos" \
       tests/test_poda_grado_maximo.py "necesita el grafo construido"
correr "A2 · el muestreo solo mira hacia atrás" \
       tests/test_causalidad_muestreo.py "necesita el grafo construido"

echo "────────────────────────────────────────────────────────────"
if [ $fallos -gt 0 ]; then
    echo "  $fallos invariante(s) ROTO(S)"
    exit 1
fi
echo "  invariantes OK${saltados:+ ($saltados saltado(s))}"
