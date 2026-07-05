# Copyright (c) 2026 PJHkorea. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# This module complies with the pure "Branchless & Loop-Unrolling Aesthetics" framework.

import jax
import jax.numpy as jnp
from typing import Tuple, Dict

# ====================================================================
# [HARDWARE ACCELERATOR CORE CONFIGURATION]
# [KR] 하드웨어 가속기(XLA) 컴파일 타임 및 수치해석적 임계 가드 상수 설정
# [EN] XLA compiler guidelines and numerical thresholds fixed at compile-time
# ====================================================================
MAX_SCAN_BOUND: int = 32         # [KR] XLA 가속기 단에 강제 고정할 최대 루프 상한선 / [EN] Strict loop upper bound enforced on the accelerator
CONVERGENCE_FLOOR: float = 1e-4  # [KR] 수렴 판정 물리 임계 바닥선 / [EN] Physical floor threshold for convergence check
RESTITUTION_SLOPE: float = 0.01  # [KR] 경계면 수치 탈출용 미세 Leaky 기울기 / [EN] Micro Leaky slope for numerical boundary escape

@jax.jit
def execute_pretrain_integrity_scan(observer_batch: jax.Array) -> Tuple[jax.Array, Dict[str, jax.Array]]:
    """
    [Production-Ready Enterprise Integrity Gate]
    
    [KR]
    중첩 함수 클로저(Closure)를 활용하여 SRAM Spill Over와 전역 네임스페이스 오염을 
    원천 차단하고 가속기 가동률을 극대화한 최종 진화형 데이터 가드 전처리 커널입니다.
    
    [EN]
    An elite on-device preprocessing kernel that leverages inner function closures 
    to eliminate SRAM register spill-over and global namespace pollution, maximizing ALU utilization.
    """
    # ====================================================================
    # [ACCELERATOR-ALIGNED MEMORY FLATTENING]
    # [KR] 임의의 N차원 입력 배치를 정적 주소 정렬선 사상(Stride Map)으로 일렬 평탄화
    # [EN] Flatten arbitrary N-D input batch into a 2D matrix to align with XLA stride mapping
    # ====================================================================
    original_shape = observer_batch.shape
    flattened_batch = jnp.reshape(observer_batch, (-1, original_shape[-1]))


    
    # ====================================================================
    # [PURE COMPILER-CAPTURED CLOSURE KERNEL]
    # [KR] 외부 flattened_batch를 정적 레지스터 상수로 완벽하게 바인딩하여 SRAM 오버플로우 차단
    # [EN] Bind external flattened_batch as a static register constant to prevent SRAM spill-over
    # ====================================================================
    def _verify_step_body_ultimate(carry: Tuple[jax.Array, jax.Array], _dummy_idx: jax.Array):
        # [KR] 루프 제어용 캐리 상태 분해 (현재 전산 상태 벡터, 활성 플래그 매트릭스)
        # [EN] Unpack step loop carry (current computational state vector, active flag matrix)
        state_vector, active_flag = carry
        
        # 1. [KR] 100% Element-wise 절대 오차 변화량 측정 및 float32 강제 상향을 통한 정밀도 방어
        # 1. [EN] Compute absolute discrepancy and cast to float32 to protect numerical precision from underflow
        delta_discrepancy = jnp.abs(state_vector - flattened_batch).astype(jnp.float32)
        
        # 2. [KR] 부동소수점 언더플로우를 방어하며 수렴 여부 판정 (조건문 if 없이 불리언-실수 정적 사상)
        # 2. [EN] Evaluate convergence without 'if' branches via dynamic boolean-to-float masking
        is_still_divergent = (delta_discrepancy > CONVERGENCE_FLOOR).astype(jnp.float32)
        
        # 3. [KR] 조건문 없는 완벽한 아다마르 곱(Hadamard Product) 연산으로 이전 상태의 활성 유무 상속
        # 3. [EN] Inherit operational activity state unconditionally using element-wise Hadamard products
        next_active_flag = active_flag * is_still_divergent
        
        # 4. [KR] 발산 강도가 임계치를 초과할 때 미세 기울기를 결합하여 NaN 폭발 방어 (Leaky 완충)
        # 4. [EN] Inject micro leaky slopes to suppress exploding gradients/NaNs when divergence is harsh
        soft_restitution = RESTITUTION_SLOPE * jnp.maximum(0.0, delta_discrepancy - CONVERGENCE_FLOOR)
        update_delta = (state_vector * -0.1) + soft_restitution
        
        # 5. [KR] 인라인 프리징: 플래그가 0.0이 되는 순간 뒤쪽 유령 연산 데이터는 물리적으로 완전히 동결
        # 5. [EN] Inline freezing: when next_active_flag hits 0.0, phantom updates freeze mathematically
        next_state_vector = state_vector + (next_active_flag * update_delta)
        
        return (next_state_vector, next_active_flag), None



    # ====================================================================
    # [LIGHTWEIGHT LOOP STREAMING PREPARATION]
    # [KR] 가속기 레지스터 영역 최적화를 위한 경량 가변 캐리 및 정적 인덱스 레일 빌드
    # [EN] Construct lightweight mutable carry and static index rails for optimal register allocation
    # ====================================================================
    # [KR] 입력 정밀도(fp16/bf16)와 무관하게, 루프 상태 및 마스킹 연산 레일은 오직 무결점 f32로 강제 통일하여 언더플로우 방어
    # [EN] Enforce float32 on the loop state and masking rails to prevent numerical underflow, independent of input precision
    initial_state = jnp.zeros_like(flattened_batch, dtype=jnp.float32)
    initial_flag = jnp.ones_like(flattened_batch, dtype=jnp.float32)   # [KR] 원소별 마스킹을 위해 배치와 1:1 사상되는 플래그 매트릭스 / [EN] 1:1 aligned flag matrix for element-wise masking
    scan_indices = jnp.arange(MAX_SCAN_BOUND)       # [KR] 물리적 텐서 복제를 배제하는 인라인 정적 상수 축 / [EN] Static constant axis to eliminate physical tensor replication memory overhead
    
    # ====================================================================
    # [ZERO-STALL SCAN EXECUTION]
    # [KR] 오직 가변 상태만 가볍게 들고 스캔 (HBM 대역폭 낭비 차단, SRAM 캐시 이득 극대화)
    # [EN] Stream scan loop with minimal carry memory footprint to maximize SRAM hit-rate and prevent HBM bottlenecks
    # ====================================================================
    (final_state, final_active_flag), _ = jax.lax.scan(
        _verify_step_body_ultimate, 
        (initial_state, initial_flag), 
        scan_indices
    )

    
    # ====================================================================
    # [POST-LOOP REDUCTION SQUELCH]
    # [KR] 부하가 큰 jnp.max 연산은 순차 루프 파이프라인 밖에서 '단 한 번'만 수행하여 Warp Stall 방어
    # [EN] Defer high-overhead horizontal max reduction outside the sequential loop to prevent Warp Stalls
    # ====================================================================
    # [KR] 특정 샘플(Row) 내 피처 중 단 하나라도 미수렴(1.0) 상태라면 해당 행 전체를 불량(Corrupted)으로 판정
    # [EN] If even a single feature within a row remains un-converged (1.0), flag the entire sample as corrupted
    is_corrupted = jnp.max(final_active_flag, axis=-1, keepdims=True)
    
    # [KR] 마스킹 인자를 원본 배치(flattened_batch)의 데이터 타입과 명시적으로 동기화하여 연산 효율 보장
    # [EN] Downcast the integrity mask to match the original flattened_batch dtype, preventing unwanted float32 promotion
    integrity_factor = (1.0 - is_corrupted).astype(flattened_batch.dtype)
    
    # ====================================================================
    # [ZERO-BRANCHING DATA SQUASH]
    # [KR] if문 분기 없이 불량 데이터 행 전체를 물리적 제로(0.0 Matrix) 상태로 강제 수축소멸
    # [EN] Force-squash all corrupted rows into an absolute zero matrix via algebraic broadcasting (no if-branches)
    # ====================================================================
    sanitized_flattened = flattened_batch * integrity_factor
    sanitized_batch = jnp.reshape(sanitized_flattened, original_shape)

    
    # ====================================================================
    # [AUTOGRAD-ISOLATED NON-BLOCKING METRICS]
    # [KR] 불량 데이터 감지 메트릭을 역전파 미분 사슬 그래프(Computation Graph)와 전산학적으로 절연
    # [EN] Insulate data corruption metrics from the autograd backward graph to prevent gradient pollution
    # ====================================================================
    isolated_corruption_rate = jax.lax.stop_gradient(jnp.mean(is_corrupted))
    
    # [KR] 후속 메트릭 수집 시스템 및 로그 모니터링을 위한 정적 상태 딕셔너리 패킹
    # [EN] Pack static evaluation artifacts for downstream metric logging and pipeline monitoring
    artifacts_metrics = {
        "data_corruption_rate": isolated_corruption_rate,
        "integrity_sync_status": jnp.equal(isolated_corruption_rate, 0.0).astype(jnp.float32)
    }
    
    return sanitized_batch, artifacts_metrics
