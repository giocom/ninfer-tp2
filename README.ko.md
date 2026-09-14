# NInfer

> 선택된 체크포인트. 최대 단일 GPU 추론 성능, 그리고 1,048,576 토큰 컨텍스트를 위한
> 2-GPU 경로.

NInfer는 NVIDIA GeForce RTX 5090에서 명시적으로 등록된 Qwen 체크포인트를 위해 처음부터 만든
C++/CUDA 추론 엔진입니다. 로컬 CLI 또는 OpenAI/Anthropic 호환 HTTP API를 통해 텍스트,
이미지, 비디오 프롬프트를 처리합니다. 27B 실행 패키지는 두 개의 RTX 5090에서 텐서 병렬로
실행되며, YaRN 위치 스케일링을 사용하면 최대 1,048,576 토큰의 컨텍스트를 서빙할 수
있습니다 — [듀얼 GPU (TP2)와 YaRN 1M 컨텍스트](#dual-gpu-tp2-and-yarn-1m-context) 참조.

> **이번은 포크입니다.** 상류(upstream)는 [Neroued/ninfer](https://github.com/Neroued/ninfer)이며,
> 이 트리는 커밋 `feaf4dd`에서 분기하여 27B 실행 패키지에 두 가지를 추가합니다. **듀얼 GPU
> 텐서 병렬성** (`--tp 2 --devices A,B`)은 카드당 가중치와 KV 상주량을 절반으로 줄이며,
> 긴 컨텍스트에서 ~40% 더 빠릅니다 — 하나의 상주 모델, 하나의 프로세스, 두 개의 디바이스,
> NVLink 없음, 분산 서빙 없음. **YaRN ×4 위치 스케일링** (`--rope yarn`)은 등록된 262,144
> 토큰에서 주소 가능한 상한을 1,048,576으로 높이며, vLLM 배포와 일치하도록 계산되고 설치된
> vLLM에 대한 드리프트 테스트로 보호됩니다. 나머지는 모두 상류의 것입니다: `--tp 1` 출력은
> [`tests/data/tp1-golden/`](tests/data/tp1-golden/MANIFEST.md)의 탐욕적 케이스에서
> `feaf4dd`와 바이트 동일하며, 단일 GPU 동작, 지원되는 ID, 아티팩트 형식 및 프로토콜 서피스는
> 변경되지 않았습니다. 두 기능의 설계 결정, 수학적 계약 및 자격 증거는
> [듀얼 GPU (TP2) 실행과 YaRN 1M 컨텍스트](docs/maintainer/tp2-yarn-1m.md)에 있습니다.
> [NOTICE](NOTICE)에서 속성 정보를 참조하세요.

NInfer는 일반적인 모델 런타임 역할을 대신하여 의도적으로 폐쇄된 모델 아티팩트 세트를 지원합니다:

| 모델 | 가중치 | NInfer 아티팩트 | 크기 | SHA-256 |
|---|---|---|---:|---|
| [Qwen3.6-27B](https://huggingface.co/neroued/Qwen3.6-27B-NInfer) | `groupwise-int` | `qwen3_6_27b.ninfer` | 17,495,365,888 bytes (16.29 GiB) | `7b51600ffd10632b9660f56085efdd9b751d79733ad32036a652234b64bebe7b` |
| [Qwen3.6-27B NVFP4](https://huggingface.co/neroued/Qwen3.6-27B-nvfp4-NInfer) | `nvfp4` | `qwen3_6_27b_nvfp4.ninfer` | 18,324,064,000 bytes (17.07 GiB) | `bce5f00d066c0f20f1317bf1fdcb458264cf95837c3b1f3fbec163694627893a` |
| [Qwen3.8-27B](https://huggingface.co/neroued/Qwen3.8-27B-NInfer) | `groupwise-int` | `qwen3_8_27b.ninfer` | 18,210,531,328 bytes (16.96 GiB) | `eec39564993d6e9c7d5e383382a760f093465c9d163ec9a1bd6b80199514bf3e` |
| [Qwen3.8-27B NVFP4](https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer) | `nvfp4` | `qwen3_8_27b_nvfp4.ninfer` | 21,492,695,040 bytes (20.02 GiB) | `bb3360522a06e136e0367f5703414d26272b7285c8a6ab6194135c17dbd81b32` |
| [Qwen3.6-35B-A3B](https://huggingface.co/neroued/Qwen3.6-35B-A3B-NInfer) | `groupwise-int` | `qwen3_6_35b_a3b.ninfer` | 22,783,246,080 bytes (21.22 GiB) | `1fb9ea0b5b8561e49d9604115ec89e5d9f2b6f6434e32c37c57fffd480a325d2` |

Qwen3.6-27B와 Qwen3.8-27B는 각각 두 개의 등록된 가중치 프로필을 노출합니다. 버전 2 아티팩트
ID는 별도의 런타임 플래그 없이 프로필을 선택합니다. Qwen3.8은 27B 실행 패키지를 공유하면서
대상 키 `qwen3_8_27b`를 사용합니다. Qwen3.6 `nvfp4` 프로필은 프리필에서 W4A4 Tensor Core MMA를
사용하고 디코드에서 A16 NVFP4 커널을 사용합니다. Qwen3.8 `nvfp4` 프로필은 소스의 혼합 할당을
보존합니다: Text 레이어 0–55에서 NVFP4 MLP 가중치와 토큰 임베딩, 어텐션 입력/출력 프로젝션,
GDN Q/K/V/Z 및 출력 프로젝션, 출력 헤드, 나머지 MLP 가중치에 행 스케일 FP8. 네 가지 27B
아티팩트는 모두 동일한 Text, Vision, MTP, 접두사 재사용, CLI 및 서빙 경로를 유지합니다.

## 성능

공개된 측정값은 Qwen3.6 아티팩트 프로필 세 가지와 Qwen3.8-27B NVFP4 프로필을 다룹니다.
Qwen3.8-27B `groupwise-int` 프로필은 현재 NInfer 빌드에서 지원되지만 아직 공개된 벤치마크
캠페인에 포함되지 않았습니다.

### 동시 MTP3 디코드

포화 디코드는 INT8 그룹-64 KV 캐시, CUDA Graphs, MTP3, 활성 요청당 8,192 토큰 생성으로
RTX 5090에서 측정되었습니다. 아래 값은 실제 디코드 배치가 설정된 동시성과 동일하게 유지된
완전한 1초 구간의 집합적 커밋된 디코드 처리량입니다. MTP 수락은 전체 요청 웨이브에 걸쳐
집계됩니다. 각 동시성 셀은 `디코드 tok/s / MTP 수락`을 보고하며, 프로필은 독립적으로
읽어야 합니다.

| 모델 프로필 | C=1 tok/s / 수락 | C=2 tok/s / 수락 | C=4 tok/s / 수락 | C=8 tok/s / 수락 | C8 / C1 |
|---|---:|---:|---:|---:|---:|
| Qwen3.6-27B `groupwise-int` | 185.8 / 68.2% | 247.0 / 69.0% | 309.5 / 68.4% | 535.0 / 68.3% | 2.88× |
| Qwen3.6-27B `nvfp4` | 202.4 / 69.3% | 399.7 / 71.4% | 699.7 / 69.3% | 1,146.9 / 68.6% | 5.67× |
| Qwen3.6-35B-A3B `groupwise-int` | 593.0 / 67.2% | 877.7 / 68.2% | 1,166.0 / 69.8% | 1,313.8 / 67.3% | 2.22× |
| Qwen3.8-27B `nvfp4` | 143.8 / 48.9% | 267.6 / 48.1% | 461.1 / 45.8% | 766.6 / 46.0% | 5.33× |

C=8에서 Qwen3.6-35B-A3B는 **1,313.8 집합적 디코드 tok/s**에 도달합니다. Qwen3.6-27B NVFP4는
**1,146.9 tok/s**와 C=1 대비 **5.67×**를 달성합니다. Qwen3.8-27B NVFP4는 **45.8–48.9%** MTP
수락률을 보이며, 다른 측정 프로필의 **67.2–71.4%**와 대조됩니다. 따라서 집합적 커밋 처리량은
실행 성능과 관망적 수락률을 모두 반영합니다.

### 단일 요청 서빙

단일 요청 코퍼스는 동일한 GPU에서 INT8 그룹-64 KV 캐시, CUDA Graphs, 1,024 토큰 프리필
청크로 측정되었습니다. 각 보고된 픽스처는 서버 워밍업 후 다섯 개의 고정 시드를 사용합니다.
대상과 가중치 프로필은 교차 대상 비교가 아닌 독립적으로 보고됩니다. 요청은 지속적인 서버에
순차적으로 제출되었습니다. Qwen3.8-27B NVFP4 MTP0 결과는 Qwen3.6 프로필과 동일한 전용 순차
코퍼스 러너를 사용합니다. MTP3 결과는 [성능](docs/performance.md)에 문서화된 고정 동시 코퍼스
캠페인의 C=1 지점에서 나옵니다.

**Qwen3.6-35B-A3B**

- 7,680 토큰 프롬프트에서 MTP0: **15,544.3 프리필 tok/s** 및 **271.1 디코드 tok/s**.
- 260,096 토큰 프롬프트에서 MTP0: **5,157.1 프리필 tok/s** 및 **188.2 디코드 tok/s**.
- MTP3 장거리 추론: **620.3–726.2 디코드 tok/s**, **72.7–82.8% 수락률**.
- MTP3 구조화된 출력: **770.9 디코드 tok/s**, **89.1% 수락률**, **3.67 토큰/라운드**.

**Qwen3.6-27B (`groupwise-int`)**

- 7,680 토큰 프롬프트에서 MTP0: **3,218.1 프리필 tok/s** 및 **77.6 디코드 tok/s**.
- 260,096 토큰 프롬프트에서 MTP0: **1,614.8 프리필 tok/s** 및 **54.8 디코드 tok/s**.
- MTP3 장거리 추론: **161.9–175.4 디코드 tok/s**, **73.4–78.8% 수락률**.
- MTP3 구조화된 출력: **193.0 디코드 tok/s**, **88.7% 수락률**, **3.66 토큰/라운드**.

**Qwen3.6-27B (`nvfp4`)**

- 7,680 토큰 프롬프트에서 MTP0: **11,191.5 프리필 tok/s** 및 **86.4 디코드 tok/s**.
- 260,096 토큰 프롬프트에서 MTP0: **2,510.6 프리필 tok/s** 및 **59.9 디코드 tok/s**.
- MTP3 장거리 추론: **213.1–231.0 디코드 tok/s**, **76.3–81.1% 수락률**.
- MTP3 구조화된 출력: **252.2 디코드 tok/s**, **89.8% 수락률**, **3.69 토큰/라운드**.

- 동일한 코퍼스와 런타임 옵션에서 groupwise-int 대비: **7,680 토큰 프리필 처리량의 3.48×**,
  **260,096 토큰 프리필 처리량의 1.55×**, **MTP3 디코드 처리량 30–32% 향상**.

**Qwen3.8-27B (`nvfp4`)**

- 7,680 토큰 프롬프트에서 MTP0: **8,340.4 프리필 tok/s** 및 **71.2 디코드 tok/s**.
- 260,096 토큰 프롬프트에서 MTP0: **2,203.1 프리필 tok/s** 및 **52.9 디코드 tok/s**.
- MTP3 장거리 추론: **151.4–195.2 디코드 tok/s**, **56.2–76.0% 수락률**.
- MTP3 구조화된 출력: **219.8 디코드 tok/s**, **90.8% 수락률**, **3.72 토큰/라운드**.

전체 방법론, 변동성, 재현 명령어 및 픽스처별 결과는 [성능](docs/performance.md)을 참조하세요.

## 평가

기능 점수는 추론이 활성화된 NInfer의 OpenAI 호환 서빙 경로에서 측정되었으며, MTP=3 및
EvalScope 1.9.0 (0-shot, 규칙 기반 채점, 문제당 1개 샘플)을 사용했습니다:

| 모델 프로필 | AIME 2025 | AIME 2026 | GPQA-Diamond | ERQA | RealWorldQA |
|---|---:|---:|---:|---:|---:|
| [Qwen3.6-27B groupwise-int](model-cards/Qwen3.6-27B-NInfer/README.md) | 86.67% | 93.33% | 86.87% | — | — |
| [Qwen3.6-27B NVFP4](model-cards/Qwen3.6-27B-nvfp4-NInfer/README.md) | 93.33% | 93.33% | 84.34% | — | — |
| [Qwen3.6-35B-A3B groupwise-int](model-cards/Qwen3.6-35B-A3B-NInfer/README.md) | 90.00% | 90.00% | 85.35% | — | — |
| [Qwen3.8-27B groupwise-int](model-cards/Qwen3.8-27B-NInfer/README.md) | 96.67% | 96.67% | 87.37% | 66.25% | 82.22% |
| [Qwen3.8-27B NVFP4](model-cards/Qwen3.8-27B-nvfp4-NInfer/README.md) | 96.67% | 96.67% | 90.40% | 66.25% | 83.53% |

Qwen3.6 행은 온도 0.6 및 존재 페널티 1.0을 사용했고, Qwen3.8-27B 행은 온도 1.0 및 존재
페널티 0.0을 사용했습니다. 멀티모달 열(ERQA와 RealWorldQA)은 `--vision`과 81,920 토큰
컨텍스트 제한으로 실행되었습니다. 텍스트 열은 Qwen3.8-27B NVFP4를 제외하고 262,144 토큰
제한을 사용했으며, Qwen3.8-27B NVFP4는 가중치 후 RTX 5090에 맞추기 위해 252,928이 필요합니다.

이는 해당 NInfer 평가 프로필 아래의 단일 샘플 결과이며 pass@k가 아닙니다. 올바른/전체
카운트와 평가 참고 사항은 모델 카드와 [전체 성능 문서](docs/performance.md)를 참조하세요.

## 요구 사양

NInfer는 현재 다음을 요구합니다:

- 64비트 Linux;
- NVIDIA GeForce RTX 5090 (`sm_120a`) 또는 RTX 3060 (`sm_86`) 한 개, 또는 `--tp 2`용 두 개;
- CUDA 13.1을 지원하는 NVIDIA 드라이버와 CUDA 툴킷 13.1 이상;
- CMake 3.28 이상 및 C++20을 지원하는 호스트 컴파일러;
- `pkg-config`;
- FFmpeg 개발 라이브러리: `libavformat >= 60`, `libavcodec >= 60`,
  `libavutil >= 58`, `libswscale >= 7`;
- `libcurl >= 7.85`;
- 아래 명령어 사용 시 Ninja.

빌드는 CUDA 아키텍처 `86`과 `120a`를 허용합니다. 설치 타겟이나 패키지화된 바이너리 배포는
없으며, NInfer는 소스 빌드 트리에서 실행됩니다.

## 빌드

이 포크를 클론하세요, 상류가 아닙니다 — 상류에는 `--tp 2`도 `--rope yarn`도 없습니다.

```bash
git clone https://github.com/giocom/ninfer-tp2.git
cd ninfer-tp2-1m

cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

RTX 3060 (`sm_86`) 빌드의 경우 기본 아키텍처는 이미 `86`입니다. 명시적으로 설정하려면:

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build --parallel
```

참고: sm_86에서는 `groupwise-int` 가중치 프로필만 작동합니다. `nvfp4` 프로필은 sm_100+를
요구하며 sm_86 빌드에서는 제외됩니다.

기본 구성은 다음을 빌드합니다:

```text
build/apps/ninfer
build/apps/ninfer-serve
```

테스트, 벤치마크 및 유지 관리 도구는 기본 빌드에서 제외됩니다.

## Docker

64비트 Linux 호스트에서 RTX 5090 또는 RTX 3060, CUDA 13.1 호환 NVIDIA 드라이버, Docker 및
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)과
함께 런타임 이미지를 빌드하세요.

```bash
docker build --tag ninfer:local .
```

아래 설명처럼 `models/`에 모델을 다운로드한 후 HTTP 서버를 실행합니다:

```bash
docker run --rm \
  --gpus '"device=0"' \
  --publish 8080:8080 \
  --volume "$PWD/models:/models:ro" \
  ninfer:local \
  ninfer-serve /models/qwen3_6_27b.ninfer \
  --host 0.0.0.0
```

동일한 이미지에서 CLI를 실행합니다:

```bash
docker run --rm \
  --gpus '"device=0"' \
  --volume "$PWD/models:/models:ro" \
  ninfer:local \
  ninfer /models/qwen3_6_27b.ninfer \
  --prompt "Explain prefill and decode in three sentences." \
  --max-new 256
```

## 모델 다운로드

Hugging Face CLI를 사용하여 등록된 아티팩트 중 하나를 다운로드합니다:

```bash
hf download neroued/Qwen3.6-27B-NInfer \
  qwen3_6_27b.ninfer \
  --local-dir models

# 또는 27B NVFP4 가중치 변형:
hf download neroued/Qwen3.6-27B-nvfp4-NInfer \
  qwen3_6_27b_nvfp4.ninfer \
  --local-dir models

# 또는 Qwen3.8-27B:
hf download neroued/Qwen3.8-27B-NInfer \
  qwen3_8_27b.ninfer \
  --local-dir models

# 또는 Qwen3.8-27B NVFP4:
hf download neroued/Qwen3.8-27B-nvfp4-NInfer \
  qwen3_8_27b_nvfp4.ninfer \
  --local-dir models

# 또는:
hf download neroued/Qwen3.6-35B-A3B-NInfer \
  qwen3_6_35b_a3b.ninfer \
  --local-dir models
```

현재 NInfer 빌드는 버전 2 아티팩트 컨테이너만 허용하며, 위의 다섯 가지 다운로드는 모두 버전
2입니다. 마이그레이션은 버전 2 공개 이전에 다운로드된 Qwen3.6 아티팩트에만 적용됩니다.
Qwen3.8-27B 프로필 두 가지는 모두 버전 2로 직접 공개되었습니다. 이전 정확한 로컬 파일을
제자리에서 마이그레이션합니다:

```bash
python3 -m tools.artifact.migrate_v1_to_v2 models/qwen3_6_27b.ninfer
```

`qwen3_6_27b_nvfp4.ninfer` 또는 `qwen3_6_35b_a3b.ninfer`에 대해 동일한 명령어를 사용합니다.
마이그레이션은 컨테이너 메타데이터만 업데이트하며, 가중치 페이로드를 다시 작성하지 않습니다.
대안으로 Hugging Face 저장소에서 현재 버전 2 파일을 다시 다운로드하세요.

각 `.ninfer` 파일에는 NInfer가 필요로 하는 가중치와 프론트엔드 리소스가 포함됩니다. 이는
Transformers 체크포인트, Safetensors 배포 또는 GGUF 파일이 아닙니다.

각 아티팩트는 완전하며, GPU 상주량은 프로세스 시작 시 고정됩니다. 관망적 디코딩은 기본적으로
비활성화되어 있어 MTP/DFlash 상태와 최적화된 제안 헤드가 업로드되지 않습니다. Vision도
기본적으로 비활성화되어 있어 가중치, Vision 스cratch 단계 및 동결된 요청-임시 할당이 제외됩니다.
이미지 또는 비디오 입력을 수락해야 하는 CLI 또는 서버 프로세스에 `--vision`을 추가하세요.
비활성화된 기능은 나중의 요청으로 활성화할 수 없습니다. DFlash는 35B-A3B 대상에만 사용 가능하며
텍스트 전용입니다.

## CLI 실행

```bash
./build/apps/ninfer models/qwen3_6_27b.ninfer \
  --prompt "Explain prefill and decode in three sentences." \
  --max-context 16384 \
  --max-new 256 \
  --spec mtp --draft-tokens 3 \
  --lm-head-draft
```

채팅 기록, 이미지 또는 비디오의 경우 `--prompt` 대신 `--messages FILE`을 사용합니다:

```bash
./build/apps/ninfer models/qwen3_6_27b.ninfer \
  --messages examples/cli/messages/image_chart.json \
  --max-context 8192 \
  --max-new 128 \
  --vision
```

답변 내용은 stdout에 기록됩니다. 로딩 진행률, 추론, 타이밍, 처리량, 메모리 및 관망적
디코딩 통계는 stderr에 기록됩니다. 구조화된 입력과 런타임 옵션은 [CLI 가이드](docs/cli.md)와
[커밋된 예제](examples/cli/)를 참조하세요.

## HTTP 서버 실행

```bash
./build/apps/ninfer-serve models/qwen3_6_27b.ninfer \
  --max-context 16384 \
  --kv-capacity auto \
  --max-concurrency 2 \
  --spec mtp --draft-tokens 3 \
  --lm-head-draft
```

공개 모델 ID는 아티팩트의 `identity.model_id`를 기본값으로 사용합니다. 배포별 별칭을
공개하려면 `--model-id`만 사용하세요.

그런 다음 OpenAI 스타일 요청을 보냅니다:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.6-27b",
    "messages": [{"role": "user", "content": "Reply with one short sentence."}],
    "max_tokens": 64
  }'
```

서버는 또한 타입화된 Items, 의미적 SSE, 로컬 연속 상태 및 함수 호출을 포함하는 OpenAI
Responses Core와 Anthropic Messages, 토큰 카운팅 및 멀티모달 입력을 구현합니다.
[HTTP 서빙](docs/serving.md)을 참조하세요.

## 듀얼 GPU (TP2)와 YaRN 1M 컨텍스트

`--tp 2`는 하나의 상주 모델을 두 개의 RTX 5090으로 분할하고, `--rope yarn`은 주소 가능한
컨텍스트 상한을 등록된 네이티브 262,144 토큰에서 1,048,576으로 높입니다. 두 기능은 독립적
입니다 — TP2는 모든 컨텍스트에서 카드당 가중치와 KV 상주량을 절반으로 줄이고, YaRN은 어떤
`--tp` 폭에서든 위치를 확장합니다 — 그러나 1,048,576 토큰은 INT8 KV와 함께 사용할 때만
맞습니다.

TP2는 스케일아웃 기능이 아닌 용량 기능입니다: 하나의 프로세스, 하나의 상주 모델, 두 개의 CUDA
디바이스, NVLink 없음, 분산 서빙 없음. 27B 실행 패키지(`qwen3.6-27b` 및 `qwen3.8-27b`,
모든 가중치 프로필)에 구현되었습니다. `qwen3.6-35b-a3b`에는 텐서 병렬 경로가 없어 시작 시
`--tp 2`를 거부합니다. 아래 모든 측정값은 Qwen3.8-27B NVFP4 아티팩트에서 수행되었습니다.

### 사용법

```bash
# 1,048,576 토큰 컨텍스트, 양쪽 GPU, INT8 KV, MTP3 관망적 디코딩
./build/apps/ninfer-serve models/qwen3_8_27b_nvfp4.ninfer \
  --tp 2 --devices 0,1 \
  --rope yarn --yarn-factor 4.0 --yarn-origin 262144 \
  --max-context 1048576 --kv-dtype int8 --kv-capacity auto \
  --max-concurrency 1 \
  --spec mtp --draft-tokens 3 --lm-head-draft
```

동일한 플래그가 CLI를 구동합니다. 짧은 `--max-new` 예산이 답변 채널에 도달해야 하는 경우
`--no-thinking`을 추가하세요. 이 체크포인트의 기본 추론 모드에서는 16 토큰 예산이 추론 스트림
내에서 완전히 소비되기 때문입니다:

```bash
./build/apps/ninfer models/qwen3_8_27b_nvfp4.ninfer \
  --tp 2 --devices 0,1 \
  --rope yarn --yarn-factor 4.0 --yarn-origin 262144 \
  --max-context 1048576 --kv-dtype int8 --kv-capacity auto \
  --messages long_prompt.json --max-new 256 --no-thinking
```

- `--tp 2`는 동일한 연산 능력을 가진 두 개의 별도 디바이스를 명명하는 명시적 `--devices A,B`를
  요구합니다. `--tp 1`은 여전히 기본값이며 변경되지 않았습니다: `scripts/tp1-regression.sh`는
  이 빌드의 탐욕적 출력을 상류 `ninfer`의 기본 커밋 `feaf4dd`에서 기록된 토큰 스트림과 비교하며,
  짧은 채팅, 2,191 토큰 명령어 및 28,677 토큰 문서에서 토큰 ID, 생성된 텍스트 및 결정론적
  요약 행이 바이트 동일해야 합니다. 범위는 정확히 다음과 같습니다: 탐욕적 텍스트 디코드,
  `--tp 1`, `--rope native`, NVFP4 가중치, `qwen3.8-27b`, MTP 꺼짐, 동시성 1.
  `tests/data/tp1-golden/MANIFEST.md`를 참조하세요.
- `--rope yarn`은 `--yarn-origin`이 아티팩트의 등록된 네이티브 용량(`262144`)과 같기를 요구합니다.
  `--yarn-factor`는 `[1.0, 64.0]`의 유한 값이며 `origin x factor`는 `1048576` 이하의 정수
  토큰 수여야 합니다.
- 1M에서는 `--kv-dtype int8`이 필수입니다: BF16 KV는 풀의 네 배를 필요로 하며 맞지 않습니다.
- 1M에서는 `--max-concurrency 1`이 정책이 아닌 산술입니다: MTP 없이 하나의 시퀀스는 디바이스당
  16.66 GiB를 소비하고 MTP가 있으면 17.69 GiB를 소비하므로 32 GiB 카드에 두 번째 슬롯이
  맞지 않습니다.
- MTP 관망적 디코딩(`--spec mtp --draft-tokens 1..5`, 선택적으로 `--lm-head-draft`)은 1M에서도
  `--tp 2`에서 작동합니다. `--spec dflash`는 `--tp 2`에서 거부됩니다. sm_120a 빌드에서는
  `--tp 2`에서 `--vision`도 거부됩니다. sm_86 빌드에서는 Vision이 기본 디바이스에서 복제된
  타워에 대해 `--tp 2`로 실행됩니다.

전체 옵션 계약은 [CLI 가이드](docs/cli.md)와 [HTTP 서빙](docs/serving.md)을 참조하세요.

### 메모리, GPU당

Qwen3.8-27B NVFP4, `--tp 2 --devices 0,1 --kv-dtype int8 --kv-capacity auto`, 활성 요청 1개.
`상주량`은 `nvidia-smi` 프로세스별 메모리이며, 모든 실행의 모든 샘플에서 평탄했습니다 —
1M 토큰 프리필이 로드 시 선택된 값보다 상주량을 밀어 올리지 않습니다.

| 컨텍스트 | MTP | 가중치 | 시퀀스 (KV + GDN 상태) | 워크스페이스 | 예약됨 (로드 요약) | 상주량 (`nvidia-smi`) |
|---:|---|---:|---:|---:|---:|---:|
| 262,144 | 꺼짐 | 10.08 GiB | 4.28 GiB | 0.18 GiB | — | **15.04 GiB** |
| 262,144 | MTP3 | 10.46 GiB | 4.54 GiB | 0.19 GiB | — | **15.69 GiB** |
| 1,048,576 | 꺼짐 | 10.08 GiB | 16.66 GiB | 182.81 MiB | 26.93 GiB | **27.41 GiB** |
| 1,048,576 | MTP3 | 10.46 GiB | 17.69 GiB | 192.93 MiB | 28.42 GiB | **28.84 GiB** |

KV 풀은 INT8 그룹-64에서 디바이스당 토큰당 16.5 KiB이며, 양자화 스케일 평면이 포함됩니다.
이는 1,048,576 토큰에서 16.50 GiB 풀입니다.
MTP3를 켜면 **측정된 디바이스당 예약 메모리 1.49 GiB (1M) 및 0.65 GiB (262k)**의 비용이
발생합니다 — 상수 값이 아닙니다. 두 가지 주요 항목은 **0.38 GiB의 헤드 가중치** (모든 윈도우에서
고정)와 **1M 토큰 윈도우당 1.03 GiB의 MTP KV**입니다. 각 측정 델타의 나머지는 워크스페이스와
시퀀스-아레나 반올림입니다. KV 항목은 윈도우에 걸쳐 하나의 전체 추가 어텐션 레이어의 KV —
텍스트 풀의 1/16 — 이며, 이것이 1레이어 MTP 헤드의 비용입니다. 262,144 토큰 행은
`ninfer-serve`의 시작 레코드와 `nvidia-smi`를 통해 측정되었으므로 CLI 로드 요약 `예약` 행이
없습니다.

MTP3와 함께 1M에서 30 GiB 디바이스당 예산까지의 여유는 **1.16 GiB**로, 가장 빡빡한 출시
구성입니다. 비교를 위해, 동일한 아티팩트를 `--tp 1`로 실행하면 MTP 꺼짐 시 252,928 토큰
윈도우에 대해 한 장에서 27.9 GiB가 필요하고 MTP3에서는 29.16 GiB가 필요하며, 262,144에
도달할 수 없습니다.

### 측정된 성모

**모든 수치는 GPU당 전력 제한을 수반합니다.** 캠페인은 **400 W** 캡(이 카드의 최소 설정
제한)에서 측정되었으며, 공개 가능한 세트는 양쪽 카드를 **575 W**로 재측정했습니다(공급업체
기본값은 600 W 및 575 W; 최대 600 W). 아래 어떤 수치도 해당 전력 조건 없이 인용하지
마세요.

단일 요청, INT8 KV, CUDA Graphs 켜짐, 탐욕적 디코딩, 두 폭에서 바이트 동일한 프롬프트:

| 워크로드 | TP1 @400 W | TP2 @400 W | TP1 @575 W | TP2 @575 W |
|---|---:|---:|---:|---:|
| 249,955 토큰 프롬프트, MTP 꺼짐 — 프리필 | 2,269.8 | **2,680.1** | 2,484.2 | **2,787.0 tok/s** |
| 249,955 토큰 프롬프트, MTP 꺼짐 — 디코드 | 52.35 | **75.18** | 53.95 | **75.32 tok/s** (1.40x) |
| 249,955 토큰 프롬프트, MTP3 — 디코드 | 101.7 | **152.1** | 113.60 | **159.39 tok/s** |
| 249,955 토큰 프롬프트, MTP3 — 드래프트 수락률 | 50.83% | **57.96%** | 50.83% | **57.96%** |
| 249,955 토큰 프롬프트 — 첫 토큰까지 시간 | 111.0 s | **93.7 s** | 101.0 s | **90.0 s** |
| 536 토큰 추론 프롬프트, MTP3 — 디코드 | 152.0 | **189.5** | — | — |
| 652,955 토큰 프롬프트, MTP 꺼짐 — 프리필 / 디코드 | 맞지 않음 | **1,348.4 / 58.22** | 맞지 않음 | 재측정 안 함 |
| 1,045,954 토큰 프롬프트, MTP 꺼짐 — 프리필 / 디코드 | 맞지 않음 | **928.9 / 46.39** | 맞지 않음 | **975.1 / 48.08** |
| 1,045,954 토큰 프롬프트, MTP3 — 디코드 (512 토큰) | 맞지 않음 | — | 맞지 않음 | **100.54 tok/s** at 56.41% |

캡을 올리면 TP1보다 TP2에 더 도움이 되며, 그 이유는 샘플링된 전력에서 볼 수 있습니다: 전체
모델을 실행하는 한 장은 한계(피크 575.5 W)를 포화시키는 반면, 공유하는 두 장은 391 W와
406 W에서 피크를 찍습니다. 따라서 TP2 대비 TP1 디코드 이점은 **400 W에서 1.44x에서
575 W에서 1.40x로 좁아집니다** — TP2는 여전히 더 빠르며, 훨씬 더 작은 전력 범위 내에서
도달합니다.

575 W에서 1,045,954 토큰 프리필은 요청당 **17.9분**이 걸리며, 400 W 캡의 18.8분에
대조됩니다. 디코드는 급락보다는 컨텍스트에 따라 매끄럽게 저하됩니다: 400 W 캡에서 8k에서
97.9 tok/s, 653k에서 58.2, 1,046k에서 46.4; 1,046k 지점은 575 W에서 48.1로 상승합니다.
250k 및 1,046k 티어만 575 W에서 재측정되었습니다. 긴 컨텍스트에서 TP2는 TP1보다 *더 빠르며*,
단순히 더 큰 것이 아닙니다. 카드당 가중치와 KV 트래픽이 절반으로 줄어들고 CUDA Graphs 하에서
크로스 디바이스 콜렉티브가 토큰당 약 0.2ms의 비용이 들기 때문입니다.

262,144 토큰 윈도우에서 포화 동시 디코드, `--tp 2`, 집합적 커밋 tok/s:

| 동시성 | MTP 꺼짐 @400 W | MTP3 @400 W | MTP 꺼짐 @575 W | MTP3 @575 W |
|---:|---:|---:|---:|---:|
| 1 | 93.5 | 172.4 | 94.1 | 177.8 |
| 2 | 183.0 | 287.0 | — | — |
| 4 | **314.3** (3.36x) | **466.0** (2.70x) | **320.3** (3.40x) | **475.2** (2.67x) |

C=4에서 GPU당 상주량은 15.64 GiB 이하를 유지했습니다.

### vLLM 대비

바이트 동일한 250k / 653k / 700k 토큰 프롬프트에서 **양쪽 카드를 500 W로 캡한 상태에서**
— 위의 400 W 및 575 W 행과 비교할 수 없는 세 번째 전력 조건 — vLLM 0.25.1 (TP2, FP8 KV,
`--hf-overrides`를 통한 YaRN, MTP3)은 프리필이 **1.17-1.32배 빠르며**, NInfer는 디코드에서
**250k에서 1.41배 빠르고 653k 및 700k에서 2.5-2.8배 빠릅니다**. 이는 vLLM의 MTP 수락률이
네이티브 262,144 토큰 윈도우를 넘어서 **정확히 0%**인 반면 NInfer는 51-60%를 유지하기
때문입니다. NInfer는 또한 **더 적은 메모리로 38% 더 큰 윈도우**를 보유합니다: GPU당
27.41 GiB에서 1,048,576 토큰 대비 vLLM의 28.90 GiB에서 759,297 토큰. 512 토큰 답변에서
요청은 거의 프리필이므로 vLLM이 모든 티어에서 먼저 완료합니다. NInfer는 대략 4,800 /
7,600 / 8,800 출력 토큰 이상에서 승리합니다. 두 엔진은 다른 파인튜닝의 다른 NVFP4 양자화와
다른 KV 데이터 유형을 사용했으며 품질 주장은 없습니다 — 전체 표, 방법 및 주의 사항은
[성능](docs/performance.md#cross-engine-comparison-against-vllm-nvfp4-500-w-per-gpu)에
있습니다.

### 검색

바늘-in-건초더미, 다섯 깊이 (10/30/50/70/90%) x 두 독립 건초더미, 규칙 기반 정확한
일치, 추론 비활성화, 탐욕적:

| 엔진 및 구성 | 건초더미 | 프롬프트 토큰 | 검색됨 |
|---|---|---:|:-:|
| NInfer TP2, 네이티브 rope | 262k, **타일링된** 코퍼스 | 259,954 | **10 / 10** |
| NInfer TP2 + YaRN x4 | 653k, 독립 텍스트 | 652,954-652,955 | **10 / 10** |
| NInfer TP2 + YaRN x4 | **1,046k, 독립 텍스트** | 1,045,954-1,045,955 | **10 / 10** |
| vLLM 대조 (모델 상한) | 653k, 독립 텍스트 | 652,954-652,955 | **10 / 10** |

해당 표에는 세 가지 자격이 따릅니다:

- **262k 행은 타일링된 건초더미를 사용합니다.** 기본 코퍼스는 2.7 MB(644 KB의 영어 에세이
  중국 소설)이며, 어떤 영어 티어든 대략 150k 토큰을 지나면 코퍼스가 반복됩니다 — 262k에서
  각 윈도우는 약 두 번 발생합니다. 바늘이 고유하기 때문에 검색은 유효하지만, 이는 653k 및
  1,046k 행보다 쉬운 작업입니다. 후자의 건초더미는 200개의 샘플링된 윈도우 중 200개가 정확히
  한 번 발생하는 의도적으로 구축된 독립 텍스트입니다.
- **vLLM 대조는 다른 체크포인트입니다** (`orcarouter/Qwen3.8-27B-Uncensored-FP8`, FP8
  가중치, FP8 KV, 추론 켜짐) NInfer의 다른 파인튜닝의 NVFP4 변환과 대조됩니다. 두 행은
  동일한 모델 패밀리와 YaRN 구성을 공유하며, 가중치는 공유하지 않습니다. 이 행은 이 YaRN
  구성 하에서 모델 측 검색 상한을 설정합니다. 이는 엔진 간 비교가 아닙니다.
- **10/10은 높은 깊이별 성공률의 증거가 아닙니다.** 1M 그리드는 깊이당 두 샘플을 실행했으며,
  10회 시도에서 10회 성공을 위한 정확한 단측 95% Clopper-Pearson 하한은 74%입니다.
- **262k를 조금 넘는 바늘만으로 YaRN이 작동한다는 증거가 아닙니다.** YaRN 설명자가
  무효화된 상태에서도 270k 바늘은 여전히 해결되었습니다 — 모델은 보조 없이 훈련된 상한을
  대략 10% 초과하여 허용합니다. 따라서 실제 가중치 YaRN 테스트의 >262k 구간은 *위치
  주소 지정 능력*을 확립하며, YaRN 품질을 확립하는 것이 아닙니다. 653k 및 1,046k 검색
  행이 진정한 네이티브 윈도우의 배수에서 YaRN의 가치를 입증하는 것입니다.

검색, 회상이 아닙니다: 훈련 세트에 있을 수 없는 바늘 — 훈련된 장소와 훈련된 활동 —

## 기능

등록된 모델 ID 세 가지 모두 다음을 지원합니다:

- 추론 및 비추론 프롬프트 모드의 텍스트 생성;
- 이미지, 다중 이미지, 비디오 및 혼합 멀티모달 메시지;
- 청크 프리필 및 CUDA Graphs 디코드;
- 실제 배치 디코딩을 포함한 시작 시 고정된 소규모 동시 서빙;
- 1~5까지의 드래프트 윈도우를 가진 MTP 관망적 디코딩;
- BF16 및 INT8 그룹-64 KV 캐시;
- 모델 및 추론 모드 인식 공식 샘플링 기본값, 명시적 탐욕적, 온도, top-k, top-p, min-p 및
  존재/빈도 페널티 오버라이드;
- 접두사 재사용 호환;
- 스트리밍 및 사용량 회계를 포함하는 OpenAI Responses Core, OpenAI Chat Completions 및
  Anthropic Messages;
- 프롬프트 렌더링 함수 도구 및 파싱된 도구 호출.

35B-A3B 대상은 1~15까지의 드래프트 윈도우를 가진 텍스트 전용 DFlash 관망적 디코딩을
추가로 지원합니다.

## 현재 제한 사항

- 위에 나열된 다섯 가지 `(model_id, weights_id)` 아티팩트 ID만 허용됩니다.
- 실행은 NVIDIA GeForce GPU를 위해 특화되어 있습니다. CUDA 디바이스 하나가 기본값이며, 27B
  실행 패키지는 `--tp 2 --devices A,B`로 정확히 두 개에서 실행되기도 합니다. 이는 스케일아웃이
  아닌 용량 기능입니다. sm_120a (RTX 5090)와 sm_86 (RTX 3060)가 지원됩니다.
- 하나의 엔진은 하나의 상주 모델을 소유하며 시작 시 고정된 1–8개 활성 요청 용량을 지원합니다.
  디코드 준비된 요청은 라운드 경계에서 압축되고 하나의 배치 모델 순회로 실행됩니다.
- NInfer는 대규모 또는 선점형 연속 배치, 우선순위/QoS 스케줄링, CPU/GPU 오프로드 또는 분산
  서빙을 제공하지 않습니다. 멀티 GPU 실행은 위에서 설명한 정확히 두 디바이스 텐서 병렬 폭
  입니다: 하나의 프로세스, 하나의 상주 모델, NVLink 없음, 두 디바이스 이하.
- `--max-context`는 각 시퀀스의 논리적 상한이며 등록된 모델의 네이티브 262,144 토큰 제한
  또는 27B 대상에서 `--rope yarn` 하에서 최대 1,048,576 토큰까지 구성 가능합니다.
  `--kv-capacity N`은 모든 활성 및 유지된 시퀀스를 위한 공유 메인 텍스트 KV 풀을 명시적으로
  크기 조정하며, `--kv-capacity auto`는 가중치 로드 후 남은 메모리에서 사용 가능한 최대
  용량을 선택하면서 1 GiB의 크기 조정 여유를 보존합니다. 생략하면 기본적으로 `--max-context`
  만큼의 페이지입니다. 해석된 풀은 시작 시 고정되며 요청 려인 간에 정적으로 분할되지 않습니다.
- 도구 호출은 파싱되어 클라이언트에 반환됩니다. NInfer는 도구를 실행하지 않습니다.
- C++ 헤더는 트리 내 애플리케이션에서 사용되며 설치된 SDK로 배포되지 않습니다.

## 문서

- [기여](CONTRIBUTING.md)
- [문서 목록](docs/README.md)
- [CLI](docs/cli.md)
- [HTTP 서빙](docs/serving.md)
- [성능](docs/performance.md)
- [CLI 예제](examples/cli/)

## 라이선스

NInfer는 [Apache 라이선스 2.0](LICENSE) 하에서 라이선스됩니다. 이 포크의 수정 사항은 동일한
라이선스 하에 있으며, Apache-2.0 §4(b)가 요구하는 속성 정보는 [NOTICE](NOTICE)를 참조하세요.

공개된 아티팩트는 다음에서 파생되었습니다:
[Qwen/Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B),
[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), 및
[Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B). Qwen3.6-27B NVFP4 아티팩트는
또한 다음의 고정 패킹 가중치를 사용합니다:
[rdtand/Qwen3.6-27B-PrismaSCOUT-Blackwell-NVFP4-BF16-vllm](https://huggingface.co/rdtand/Qwen3.6-27B-PrismaSCOUT-Blackwell-NVFP4-BF16-vllm).
Qwen3.8-27B NVFP4 아티팩트는 또한 다음의 고정 혼합 FP8/NVFP4 가중치를 사용합니다:
[unsloth/Qwen3.8-27B-NVFP4](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4). 이 소스
저장소는 Apache-2.0 하에서 배포됩니다. 벤더된 의존성은 `third_party/` 하에서 자체 라이선스
파일을 보유합니다.
