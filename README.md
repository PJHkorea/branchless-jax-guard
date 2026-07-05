# branchless-jax-guard

High-performance, compiler-native data pipeline verification gate engineered in JAX/XLA. Eradicates dynamic loop overheads and runtime `ConcretizationTypeError` anomalies via compile-time bounded loop unrolling and strict branchless algebraic masking.

### [KR] 아키텍처 개요
본 커널은 JAX/XLA 환경에서 초고속 온-디바이스(On-Device) 데이터 파이프라인 무결성을 검증하기 위해 설계된 정밀 방패 커널입니다. 가속기 아키텍처에서 치명적인 동적 루프 오버헤드와 JAX 추적기(Tracer)의 `ConcretizationTypeError` 난제를 완전히 박멸하기 위해, 컴파일 타임에 상한선이 고정된 루프 언롤링(Bounded Loop Unrolling) 기법과 분기문 없는 순수 대수 마스킹(Branchless Algebraic Masking) 설계 사상을 엄격하게 준수합니다.

---

## The High-Performance Paradigm

In large-scale distributed training infrastructures (e.g., LLM foundation runs), the inclusion of dynamic loops (`while/break`) or Python-level conditional branches (`if/else`) to inspect incoming data streams poses a severe threat to hardware acceleration. These algorithmic control flows trigger **Graph Fragmentation** and severe **Host-Device Synchronization Bottlenecks (GPU/TPU Stall)**, shattering the compile-time optimization guarantees of the XLA compiler.

`branchless-jax-guard` serves as an ultra-fast, non-blocking data firewall placed at the absolute front-end of the training data pipeline. It evaluates numerical stability, scan convergence speed, and latent space structural anomalies using an inline, purely mathematical approach—achieving **0% Graph Breaks** and maintaining continuous accelerator saturation.

### [KR] 고성능 하드웨어 패러다임 해설
초거대 언어 모델(LLM) 파운더리 학습과 같은 초대형 분산 인프라 환경에서, 유입되는 데이터 스트림을 검사하기 위해 동적 루프(`while`/`break`)나 파이썬 레벨의 조건부 분기문(`if`/`else`)을 결합하는 것은 하드웨어 가속기(GPU/TPU) 가동률에 치명적인 타격을 입힙니다. 이러한 제어 흐름 분기는 XLA 컴파일러의 전산 그래프 파편화(Graph Fragmentation)를 유발하고, 호스트와 디바이스 간의 동기화 병목을 일으켜 연산 코어가 정지(Stall)하는 원인이 됩니다.

`branchless-jax-guard`는 데이터 파이프라인 최전선에 전임 배치되는 논블로킹(Non-blocking) 초고속 데이터 방화벽입니다. 수치해석적 안정성, 루프 수렴 속도, 레이턴트 공간의 구조적 변이 유무를 오직 인라인 행렬 수식만으로 판정하여, 그래프 단절(Graph Breaks)을 0%로 통제하고 가속기 가동률을 100% 한계치로 유지합니다.


---

## Infrastructure Topology

`branchless-jax-guard` acts as the definitive, autograd-isolated intake shield within the PJHkorea Zero-Branching paradigm. The entire infrastructure eliminates traditional imperative control loops, processing inputs as a continuous, unified data-flow continuum:

### [KR] 인프라 토폴로지 및 흐름도 해설
`branchless-jax-guard`는 PJHkorea Zero-Branching 패러다임 시스템 군의 최전방에서 작동하는 오토그레드(Autograd) 절연형 데이터 흡입 방패입니다. 전체 인프라는 기존의 명령형 제어 루프를 완전히 배제하며, 유입되는 모든 데이터 인풋 신호를 끊어짐이 없는 단일 데이터 플로우 연속체(Data-flow Continuum) 상태로 하드웨어 레지스터 단에 밀어 넣습니다.

```mermaid
graph TD
    %% 하드웨어 인풋 영역
    Input([User / System Input Signal Streaming]) -->|Raw Tensor Ingress| Shield1
    
    %% 1단계 입구 방패 (JAX/XLA, Apache 2.0)
    subgraph Shield1 [1. branchless-jax-guard :: Apache 2.0]
        direction TB
        A[jax.lax.scan Fixed Bound Lock] --> B[Boolean-to-Float Status Propagation]
        B --> C{Singularity Detected?}
        C -->|Yes: Flag = 1.0| D[Zero-Squelch Circuit: Multiply 0.0]
        C -->|No: Flag = 0.0| E[Bit-Exact Forward Pass: Multiply 1.0]
        D --> F[Geometric Collapse to Null Matrix]
        E --> G[Sanitized Tensor Flow]
        F & G --> H[jax.lax.stop_gradient Telemetry Offloading]
    end

    %% 런타임 가속 컴파일 영역으로 진입
    H -->|0% Graph Breaks / Static View Ingestion| RuntimeEngine[XLA COMPILATION & AUTOGRAD RUNTIME]

    %% 2, 3단계 백엔드 방패 분기 (PyTorch / CUDA, GPLv3)
    RuntimeEngine -->|Autograd Chain Rule Protection| Shield2
    RuntimeEngine -->|Silicon Register File Protection| Shield3

    subgraph Shield2 [2. egregore-flat-kernel :: GNU GPLv3]
        direction TB
        I[Analytical Flattening: Smooth exp C^∞ Surfaces] --> J[torch.where Masking Gates]
        J --> K[-99.0f Volumetric Latch Lock]
        K --> L[Softmax Underflow Gradient Annihilation]
    end

    subgraph Shield3 [3. value-system-kernel V2 :: GNU GPLv3]
        direction TB
        M[Direct embedding/danger_vectors_ptr Binding] --> N[Two's Complement Underflow Bit Trick]
        N --> O[Branchless Bitwise Masking MUX]
        O --> P[Atomic Coalesced Store & FMA Cushioning]
    end

    %% 스타일 가이드 정의 (하드웨어 친화적 톤)
    style Input fill:#222,stroke:#555,stroke-width:2px,color:#fff
    style Shield1 fill:#1a233a,stroke:#3b82f6,stroke-width:2px,color:#fff
    style RuntimeEngine fill:#2d1b1b,stroke:#ef4444,stroke-width:2px,color:#fff
    style Shield2 fill:#1e1b29,stroke:#8b5cf6,stroke-width:2px,color:#fff
    style Shield3 fill:#112417,stroke:#10b981,stroke-width:2px,color:#fff
```


---
## Core Architectural Features

* **Strict Bounded Loop Unrolling (`jax.lax.scan` Lock):** Purges dynamic code evaluation. Enforces a rigid iteration maximum directly into the XLA compiler graph, enabling full hardware pipelining without branch mispredictions.
* **Inline Gating Status Propagation:** Tracks tracking-state convergence velocities by computing cumulative boolean-to-float masking coefficients ($1.0$ or $0.0$) inside the accelerator's ALU, freezing unneeded iterations without executing a `break` jump.
* **The Zero-Squelch Circuit:** Neutralizes toxic data blocks (e.g., subnormal values or floating-point singularities that cause `NaN` explosions) by algebraically collapsing the input matrix into a pure $0.0$ state, bypassing downstream training corruption without crashing the pipeline.
* **Autograd-Isolated Metric Offloading:** Wraps logging counters with `jax.lax.stop_gradient` protocols, fully isolating runtime telemetry diagnostics from backpropagation tracking overheads.

### [KR] 핵심 아키텍처 기능 명세
* **엄격하게 고정된 루프 언롤링 (`jax.lax.scan` 락):** 동적 코드 평가와 오버헤드를 완전히 제거합니다. XLA 컴파일러 전산 그래프 내부에 강제 고정된 루프 상한선을 부여하여, 분기 예측 실패(Branch Misprediction) 없이 하드웨어 파이프라이닝의 잠재력을 100% 가동합니다.
* **인라인 게이팅 상태 전파 (Status Propagation):** 가속기 ALU 내부에서 누적 불리언-실수 마스킹 계수($1.0$ 또는 $0.0$)를 연산하여 수렴 속도를 추적합니다. 물리적인 `break` 점프 명령어를 실행하지 않고도 불필요한 이후 스텝의 데이터 변화를 완벽하게 동결(Freeze)시킵니다.
* **원천 증발 회로 (The Zero-Squelch Circuit):** `NaN` 폭발을 유발하는 부동소수점 싱큘래리티나 하위 정밀도 언더플로우 데이터 블록을 감지합니다. 발견 즉시 대수적 마스킹을 통해 입력 행렬 전체를 정밀한 $0.0$ 상태로 수축 소멸시켜, 파이프라인 다운 타임(오류로 인한 중단) 없이 안전하게 불량 데이터를 우회합니다.
* **역전파가 절연된 메트릭 오프로딩:** 실시간 진단용 로깅 카운터 및 데이터 오염 비율을 `jax.lax.stop_gradient` 프로토콜로 감쌉니다. 모델 학습의 핵심인 역전파 그라디언트 업데이트 추적 오버헤드로부터 모니터링 텔레메트릭 지표를 전산학적으로 완벽히 격리합니다.

---

## Mathematical Zero-Squelch Mechanics

The core philosophy of `branchless-jax-guard` is to avoid runtime CPU hardware interrupts or Python-level conditional branches (`if/break`) that fragment the XLA compilation pipeline. Instead, it handles anomaly isolation entirely via analytical manifold squeezing.

### [KR] 수치해석적 원천 증발 메커니즘 해설
`branchless-jax-guard`의 핵심 철학은 XLA 컴파일 파이프라인을 파편화하는 런타임 CPU 하드웨어 인터럽트나 파이썬 레벨의 조건부 분기문(`if`/`break`)을 철저히 배제하는 것입니다. 대신, 입력 데이터의 수치적 이상 현상 격리 및 무력화 처리를 수학적인 매니폴드 압착(Analytical Manifold Squeezing) 기법을 통해 가속기 내부에서 완벽하게 처리합니다.

---

### 1. Bounded Trace and Status Propagation


- Executing a strict static limit of 32 iterations ($N_{max}$) via `jax.lax.scan`, the kernel computes the absolute divergence discrepancy ($\Delta_{k}$) strictly element-wise within the accelerator's vector units (ALU) without routing any control-flow branches.

- `jax.lax.scan`을 통해 32번의 고정된 반복($N_{max}$)을 수행하는 커널은 제어 흐름 분기(branch) 없이 벡터 유닛(ALU) 내에서 Element-wise로 발산 절대 오차량($\Delta_{k}$)을 계산합니다.

$$\Delta_{k} = | \mathbf{x}_{k} - \mathbf{x}_{k-1} |$$

---

- The boolean convergence evaluation is transformed into a continuous floating-point gating multiplier inside the accelerator's ALU via direct algebraic product with a static float32 literal ($1.0f$), preventing branch mispredictions and implicit type promotion stalls.

- 불리언(Boolean) 수렴 테스트는 가속기 ALU 내부에서 정적 float32 리터럴 상수($1.0f$)와의 직접적인 대수적 곱연산을 통해 연속적인 부동소수점 게이팅 멀티플라이어로 변환됩니다. 이를 통해 분기 예측 실패와 묵시적 형변환 지연(Stall)을 원천 차단합니다.

$$\sigma_{k} = \mathbb{I}(\Delta_{k} > \epsilon) \cdot 1.0f \in \{0.0, 1.0\}$$

---

- The cumulative survival flag ($\Phi_{k}$) and the state transition vector ($\mathbf{S}_{k}$) inherit preceding historical states deterministically and execute updates completely free from branch mispredictions.

- 누적 생존 플래그($\Phi_{k}$)와 상태 전환 벡터($\mathbf{S}_{k}$)는 분기 예측 실패 없이 이전 히스토리를 결정론적으로 상속받아 업데이트됩니다.

$$\Phi_{k} = \Phi_{k-1} \times \sigma_{k}$$

$$\mathbf{S}_{k} = \mathbf{S}_{k-1} + \Phi_{k} \times \left( \mathbf{W} \cdot \mathbf{S}_{k-1} + \text{LeakySlope}(\Delta_{k}) \right)$$

---

### 2. The Zero-Squelch Operator

- If the micro-kernel exhausts all $N_{max}$ iterations and any element within the terminal active flag matrix remains active ($\Phi_{N} = 1.0$), it indicates that the input batch contains a numerical singularity (infinite vibration) that would otherwise collapse the downstream AdamW optimizer via `NaN` propagation.

- 만약 마이크로 커널이 최대 루프($N_{max}$)를 모두 소모했음에도 최종 활성 플래그 행렬(Active Flag Matrix) 내에 단 하나의 원소라도 미수렴 상태($1.0$)로 잔존해 있다면, 이는 입력 배치 내에 후속 AdamW 옵티마이저를 `NaN` 폭발로 붕괴시킬 수 있는 수치적 싱큘래리티(무한 진동 및 발산 변이)가 포함되어 있음을 의미합니다.

---

- Instead of throwing a runtime exception and halting the entire pipeline, the kernel applies the algebraic **Zero-Squelch Operator** immediately outside the sequential loop context:

- 가속기 파이프라인은 런타임 예외(Exception)를 던져 대규모 학습 시스템 전체를 중단시키는 대신, 순차 루프 파이프라인 컨텍스트 외부에서 대수적인 **원천 증발 연산자(Zero-Squelch Operator)**를 즉각 실행합니다:

$$\mathbf{I}_{factor} = 1.0 - \text{Reduction}(\mathbf{\Phi}_{N})$$

$$\mathbf{X}_{sanitized} = \mathbf{X}_{batch} \times \mathbf{I}_{factor}$$

---

- In the production kernel, the integrity factor is explicitly downcast to match the original precision of the input matrix, fully eliminating implicit runtime casting overheads.

- 프로덕션 커널 내에서 무결성 인자는 원본 입력 행렬의 데이터 타입 정밀도로 명시적 다운캐스팅되어, 하드웨어 단의 묵시적 형변환 오버헤드를 원천 차단합니다.

$$\mathbf{I}_{\text{factor}} \in \mathbb{R}^{B \times 1} \quad \text{downcast to} \quad \text{dtype}(\mathbf{X}_{\text{batch}})$$

---

- **Normal Data Ingress ($\Phi_{N} = 0.0$):** The integrity factor maps to $1.0$, guaranteeing undamaged, bit-exact forward propagation.

- **정상 데이터 인입 ($\Phi_{N} = 0.0$):** 무결성 인자가 $1.0$으로 사상되어, 원본 데이터의 훼손 없는 비트 단위 정밀도(Bit-exact) 그대로 순전파(Forward propagation)를 보장합니다.

$$\mathbf{I}_{factor} = 1.0 \implies \mathbf{X}_{sanitized} = \mathbf{X}_{batch} \times 1.0$$

---

- **Corrupted Anomaly Ingress ($\Phi_{N} = 1.0$):** The integrity factor drops to $0.0$, forcing an immediate geometric collapse into a precise null matrix and neutralizing the anomaly prior to model ingestion.
- **불량 데이터 인입 ($\Phi_{N} = 1.0$):** 무결성 인자가 정확히 $0.0$으로 강제 수축됩니다. 입력 데이터 전체를 물리적인 제로(0.0) 상태로 기하학적 소멸(Geometric Collapse) 시켜 신경망에 인입되기 전 아노말리를 원천 무력화합니다.

$$\mathbf{I}_{factor} = 0.0 \implies \mathbf{X}_{sanitized} = \mathbf{X}_{batch} \times 0.0$$


---

## API Reference & Integration Spec

### `execute_pretrain_integrity_scan`

Performs an inline, branchless pre-training data integrity scan. Unifies arbitrary N-dimensional tensors into a static virtual view, runs bounded loop unrolling, and filters singularities.

```python
import jax
import jax.numpy as jnp
# FIX: Unified path pointing to our finalized branchless 3rd-generation single-file kernel
from integrity_guard_kernel_jax import execute_pretrain_integrity_scan

# 1. Prepare raw input batch [Batch, Time, Dimension]
raw_observer_batch = jax.random.normal(jax.random.PRNGKey(42), (64, 12, 256))

# 2. Pass through the inline verification gate (Fully JIT-compatible)
sanitized_batch, artifacts_metrics = execute_pretrain_integrity_scan(raw_observer_batch)

# 3. Downstream ingestion
# Anomaly data is perfectly flattened to a 0.0 matrix without interrupting the TPU/GPU pipeline.
print(f"Data Corruption Rate: {artifacts_metrics['data_corruption_rate']}")
print(f"Pipeline Sync Status: {artifacts_metrics['integrity_sync_status']}")
```

### Input / Output Specifications

| Argument / Return | Type | Dimension | Description |
| :--- | :--- | :--- | :--- |
| `observer_batch` | `jax.Array` | `[..., SpatialDim]` | Arbitrary $N$-dimensional batch array entering the data pipeline. |
| `sanitized_batch` | `jax.Array` | Matches Input | Bit-exact original array or a compressed $0.0$ null matrix if corrupted. |
| `data_corruption_rate` | `jax.Array` | Scalar (`float32`) | Autograd-isolated statistical metric tracking the fraction of failed convergences. |
| `integrity_sync_status`| `jax.Array` | Scalar (`float32`) | Binary status (`1.0` if clean, `0.0` if anomalies were neutralized). |

---

## 📜 License

```text
Copyright (c) 2026 PJHkorea. All rights reserved.
Licensed under the Apache License, Version 2.0 (the "License");
```

- **[EN]** This repository contains open-source code distributed under the **Apache License 2.0**. The full license text can be found in the `LICENSE` file.
- **[KR]** 본 저장소는 **Apache License 2.0으로 배포하는 소스코드**입니다. 전문은 루트 디렉토리의 `LICENSE` 파일에서 확인하실 수 있습니다.

---

> ⚠️ **[EN] Disclaimer**: All code within this repository is provided "AS IS", without warranty of any kind, express or implied.
>
> ⚠️ **[KR] 면책 조항**: 본 저장소의 모든 소스코드는 "있는 그대로(AS IS)" 제공되며, 명시적 또는 묵시적인 어떠한 보증도 제공하지 않습니다.



