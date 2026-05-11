# CUDA Learning Project

Proyecto personal de aprendizaje práctico de CUDA C++. El objetivo es dominar de verdad la programación de GPUs: escribir kernels correctos, medirlos, profilearlos y optimizarlos.

---

## Modo de trabajo (IMPORTANTE)

**El usuario codea todo. Claude es solo asesor.** No escribas ni edites archivos de código fuente (`.cu`, `.cuh`, `.cpp`, `.h`) en este proyecto.

Lo que sí puedes hacer:
- Explicar conceptos (SIMT, warps, memoria, coalescing, etc.).
- Dar pseudocódigo y snippets cortos en el chat para ilustrar un punto.
- Revisar código que el usuario ya escribió (`Read` está permitido).
- Diagnosticar errores de compilación o runtime.
- Sugerir el siguiente paso o experimento.
- Interpretar salidas de `nvcc`, `nsys`, `ncu`.

Lo que NO debes hacer sin pedir confirmación explícita:
- Crear o modificar `.cu` / `.cuh` / `.cpp` / `.h`.
- Crear CMakeLists.txt, Makefiles, scripts de build (preguntar primero).
- Refactorizar código del usuario "por iniciativa".

Si el usuario dice "escríbelo tú" o "hazlo por mí", confirma una vez y procede.

---

## Hardware

- Laptop con **NVIDIA GeForce RTX 2050** (Ampere, compute capability **sm_86**).
- Verificar con `nvidia-smi` y `deviceQuery` antes de asumir specs.
- Toolkit: CUDA + nvcc. Confirmar versión con `nvcc --version`.

---

## Plan de aprendizaje (alto nivel)

| Fase | Tema | Entregable |
|------|------|------------|
| 0 | Setup + modelo mental (SIMT, jerarquía de memoria) | GPU specs documentadas |
| 1 | Vector addition (4 versiones: CPU, naive, grid-stride, pinned) | Tabla de tiempos para N = 1M / 10M / 100M |
| 2 | Reducción (6 versiones progresivas, estilo Mark Harris) | Gráfica de bandwidth GB/s por versión |
| 3 | Matrix multiplication (naive → tiled → cuBLAS) | TFLOPS y % del pico teórico |
| 4 | Patrón real: stencil / scan / histograma / SpMV / N-body | Kernel profileado con Nsight Compute |
| 5 | Profiling avanzado, streams, CUDA graphs | Duplicar perf del kernel de Fase 4 |
| 6 | Capstone: N-body con OpenGL interop o suite de kernels DL | Demo funcional |

Fase actual: **1** (vector-addition).

---

## Reglas de cada ejercicio

Todo kernel nuevo debe tener:

1. **Baseline CPU** — el mismo cómputo en C++ secuencial. Es el ground truth.
2. **Verificación numérica** — comparar GPU vs CPU con `EXPECT_NEAR` o equivalente. FP32 tolerance ~1e-5 relativa.
3. **Medición con `cudaEvent_t`** — nunca `std::chrono` para tiempos de GPU. Separar tiempo de transfer del tiempo de kernel.
4. **Out-of-bounds guard** — `if (idx >= N) return;` siempre.
5. **`CUDA_CHECK` macro** en cada llamada de la API, y `cudaGetLastError()` después de cada launch.

---

## Estándares de código

Aplicar las reglas globales en `~/max/confs/claude-config/claude/rules/cuda/coding-style.md`:

- Naming: `verbNounKernel` para kernels, `d_` para device pointers, `h_` para host, `s_` para shared.
- Punteros de solo lectura: `const T* __restrict__`.
- Prefer `float` sobre `double` salvo justificación.
- Block size múltiplo de 32 (warp size); empezar con 256.
- Headers públicos en C++ puro (no exponer `cudaStream_t` ni `dim3`).

Skill disponible para optimización profunda: `cuda-kernel-optimization` (úsala vía Skill cuando toque profilear).

---

## Estructura del repo

```
learn/cuda/
├── vector-addition/        # Fase 1
│   └── vecAdd.cu
├── reduction/              # Fase 2 (futuro)
├── matmul/                 # Fase 3 (futuro)
└── ...
```

Cada ejercicio en su propio directorio con su propio CMakeLists.txt o Makefile.

---

## Comandos útiles (referencia rápida)

```bash
# Build (cuando exista CMake)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# Ver registros / shared mem por kernel
nvcc -Xptxas -v ...

# Timeline profiling
nsys profile --stats=true ./mi_programa

# Kernel-level profiling (métricas detalladas)
ncu --set full ./mi_programa

# Inspeccionar PTX/SASS generado
cuobjdump --dump-sass mi_programa
```

---

## Cuándo preguntar al usuario

- Antes de instalar dependencias o tocar CMake/Makefiles.
- Antes de proponer un atajo que se salta una fase del plan.
- Cuando el código actual tiene un bug Y Claude tiene una sospecha — describir el bug y dejar que el usuario lo arregle.
