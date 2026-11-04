# MuJoCo Model Studio

MuJoCo 물리 시뮬레이션용 **시각적 모델 편집·조립 프로그램**입니다. 기존 MJCF를 불러와 계층/속성/관절을
편집하고, 부품(손·발·머리 등)을 독립 파일로 만들어 다른 모델에 조립하며, MuJoCo로 즉시 검증·시뮬레이션합니다.
인간형(AGI 연구용 신체)이 주 목적이지만 편집기 자체는 로봇·동물·기계 등 범용입니다.

![editor](docs/screenshots/editor_overview.png)

## 현재 상태 (이번 세션 기준)

| Phase | 내용 | 상태 |
|---|---|---|
| 1 | MJCF 불러오기/저장, 계층 트리, 3D 뷰포트, Body/Joint 속성 편집, MuJoCo 컴파일 검증 | **완료** (+ Undo/Redo, 이동/회전 기즈모, 관절 슬라이더, 시뮬레이션 패널) |
| 2 | Body 추가·복사·삭제·미러, 관절 생성/제어, 외부 모델 부분 가져오기, 손가락·발·눈·턱 예제 | **완료** (부품 추출/조립/교체, 관절 연동·그룹, 프로젝트 형식) |
| 3 | 프리미티브 생성, 폴리곤 편집, OBJ/STL 가져오기, 시각/충돌 분리 | 미착수 — 프리미티브 geom 추가·시각/충돌 표시 분리만 구현 |
| 4 | 얼굴 표정 시스템 (Shape Key / Bone) | 미착수 — 렌더링 경로 기술 검증만 완료 (`docs/DESIGN.md` §8) |
| 5 | 전신 인간형 통합 검증 | 부분 — 공식 humanoid + 손/발/머리 부품 조립 예제, 물리 검증 테스트 |
| 6 | 품질 개선 | 부분 — Undo/Redo, 오류 위치 표시, 검증, 회귀 테스트 116개 |

검증 환경: Linux(Ubuntu 24.04, Xvfb + Mesa llvmpipe 소프트웨어 OpenGL)에서 GUI 포함 전체 테스트 통과.
**Windows 11에서의 GUI 실행은 이 환경에서 검증하지 못했습니다** (코드상 Windows 전용 제약은 없으며, 아래 설치 절차를 따르면 됩니다).

## 설치

MuJoCo **3.0.1** 휠은 Python 3.8–3.12용만 있고 numpy 1.x로 빌드되어 있습니다.

```bash
python -m venv .venv
```

Windows: `.venv\Scripts\activate` / Linux: `source .venv/bin/activate`

```bash
pip install -r requirements.txt
```

`requirements.txt`는 검증한 조합(MuJoCo 3.0.1, numpy 1.26, PySide6, lxml, PyOpenGL)입니다.
최신 MuJoCo(3.17.0 + numpy 2.5에서도 전체 테스트 통과)를 쓰려면 `requirements-latest.txt`를 사용합니다.

## 실행

```bash
python -m mjstudio examples/models/humanoid.xml
```

`src`를 경로에 두려면 `pip install -e .` 후 `mjstudio` 명령을 쓰거나, `PYTHONPATH=src`를 지정합니다.
헤드리스 검증 도구:

```bash
mjstudio-validate examples/models/humanoid.xml
```

### 뷰포트 조작
- 왼쪽 드래그: 회전 · 오른쪽 드래그: 이동 · 휠/가운데 드래그: 확대
- 클릭: 선택(Shift/Ctrl 추가) · 더블클릭: 해당 지점으로 시점 중심 이동 · F: 선택 영역 맞춤
- Q/W/E: 선택/이동/회전 도구 · L: 로컬/월드 축 전환 · 드래그 중 Ctrl: 1 cm / 15° 스냅 · Esc: 취소
- Space: 시뮬레이션 재생/정지

## 주요 기능

**편집 (원본 보존 원칙)**
- MJCF를 lxml DOM 그대로 유지: 미지원 요소·속성·주석까지 저장 시 보존, 원본 파일 덮어쓰기 전 경고(기본값 "다른 이름으로 저장")
- 기본 클래스 상속 해석: 명시값(굵게)과 상속값(회색, 출처 클래스 툴팁)을 구분해 표시
- 방향 지정자(quat/euler/axisangle/xyaxes/zaxis/fromto)를 MuJoCo 컴파일러와 동일하게 해석 — 24가지 오일러 순서 포함 MuJoCo 결과와 대조 테스트
- 이름 변경 시 참조(액추에이터·센서·tendon·equality·contact·카메라 target 등) 자동 갱신
- 삭제 시 의존 요소 연쇄 처리(관절 → tendon → 그 tendon의 액추에이터), keyframe의 qpos/qvel/ctrl/mocap 자동 재매핑
- 관절이 굽은 자세에서도 정확한 월드 기준 이동/회전, 재부모화 시 월드 포즈 유지
- Undo/Redo: 모든 편집이 하나의 명령 스택(드래그 한 번 = Undo 한 번)

**부품 조립**
- 선택 Body를 독립 MJCF 부품으로 추출 (필요 에셋·기본 클래스·내부 액추에이터/센서 포함, 단독 컴파일 가능)
- 다른 모델에 조립: 이름 충돌 재명명, 동일 에셋 재사용, 에셋 경로 재계산, **각도 단위/euler 순서/autolimits 차이 자동 변환**,
  부품의 기본 클래스를 격리된 네임스페이스로 복사(대상 모델 기본값이 부품 물리를 바꾸지 않음 — 테스트로 검증)
- 부위 교체(예: 머리만 교체), 좌우 미러 복제(관절축을 축벡터로 변환해 같은 각도 = 대칭 동작)

**관절·물리**
- 관절 슬라이더(경첩 °, 슬라이드 m, 볼 관절), 선택 관절의 범위 직접 편집, 뷰포트에 관절 축과 범위 호 표시
- 관절 연동: 물리적 연동은 MJCF `<equality joint>`, 제어용 관절 그룹(주먹 쥐기, 양안 동시 회전)은 project.json
- 시뮬레이션: 재생/정지/스텝/리셋(keyframe 선택), 액추에이터 ctrl 슬라이더, 센서 값, 접촉 목록, 불안정(NaN, MuJoCo 경고, 속도 폭주) 감지

**해부학 부품 라이브러리** (`src/mjstudio/library/anatomy.py`, 자세한 내용은 [docs/ANATOMY.md](docs/ANATOMY.md))
- 손: 검지~소지 각각 MCP(굽힘+벌림)·PIP·DIP와 3개 지골, 엄지 CMC(회전+벌림)·MCP·IP, 마디별 접촉 센서 — 22관절
- 발: 발목(거퇴) → 후족부(거골하, 내/외번) → 중족부 → 전족부 → 발가락(사선 MTP 축), 뒤꿈치/전족부/발가락 별도 충돌·접촉 센서
- 머리: 좌우 눈 각각 수평/수직 관절 + 시선 방향 센서, 경첩 턱(선택적 전방 활주), 목

| | |
|---|---|
| ![hand](docs/screenshots/hand_left_fist.png) | ![head](docs/screenshots/head_eyes_jaw.png) |

## 예제

```bash
python examples/build_parts.py
```
손/발/머리 부품을 `examples/parts/*.xml`로 저장합니다 (각각 MuJoCo에서 바로 열리는 독립 모델).

```bash
python examples/assemble_humanoid.py
```
공식 MuJoCo humanoid(Apache-2.0)를 불러와 양손을 5손가락 손으로, 양발을 분절 발로, 머리를 눈·턱 머리로 교체하고,
왼손 검지 PIP 범위와 중위지골 길이를 수정하고, 왼발 전족부에 새 관절(엄지발가락 IP)을 추가한 뒤
관절 그룹과 함께 프로젝트(`examples/output/humanoid_assembled/`)로 저장 → 순수 MuJoCo로 다시 불러와 시뮬레이션합니다.
(관절 22 → 81개, keyframe 2개 자동 재매핑)

## 테스트

```bash
python -m pytest tests
```

GUI 테스트(`tests/test_gui.py`)는 디스플레이가 필요합니다. 헤드리스 Linux에서는:

```bash
xvfb-run -a -s "-screen 0 1920x1080x24" python -m pytest tests
```

요구사항 8.2–8.5의 검증 질문은 `tests/test_anatomy.py`에서 물리 시뮬레이션으로 확인합니다
(예: 40 kg 하중에서 발목 저측굴곡 시 뒤꿈치 접촉력 0, 전족부가 하중 지지, MTP 굽힘 / 손가락 22관절 각각 독립 제어 /
양안 독립 회전과 시선 방향 센서 값 / 턱 개폐와 관절 센서).

## 구조

```
src/mjstudio/
  core/        GUI 비의존 로직 (스크립트에서 직접 사용 가능)
    doc.py       MJCF 문서(lxml), 기본 클래스 해석, 포즈 해석/기록, 저장 시 에셋 경로 재작성
    compiler.py  MuJoCo 컴파일 + 요소↔ID 매핑 + 오류 위치 추적
    editor.py    편집 세션: Undo, 참조 갱신, 연쇄 삭제, keyframe 재매핑, 관절 상태
    ops.py       요소 추가/복제/미러/액추에이터/센서/연동
    parts.py     부품 추출/조립/교체
    project.py   프로젝트 폴더(model.xml + project.json), 패키징, 누락 리소스 검사, 관절 그룹
    preview.py   기즈모 드래그 실시간 미리보기
    sim.py       시뮬레이션 제어, 불안정 감지, 센서/접촉
    schema.py, mathx.py, undo.py
  gui/         PySide6 GUI (viewport, tree, props, jointpanel, simpanel, ...)
  library/     anatomy.py (손/발/머리 부품 생성기)
tests/  examples/  docs/
```

설계 상세와 MuJoCo 버전별 차이, 제약사항: [docs/DESIGN.md](docs/DESIGN.md) · 다음 단계: [docs/ROADMAP.md](docs/ROADMAP.md)

## 라이선스
- 코드: Apache-2.0
- `examples/models/humanoid.xml`: DeepMind MuJoCo 저장소의 모델(Apache-2.0, 원본 헤더 유지)
