# 다음 단계 (Phase 3–6)

현재 구조에서 각 단계가 들어갈 위치와 구현 방법입니다. 핵심 기술 위험(렌더링 경로)은 이미 검증했습니다.

## Phase 3 — 메시 생성과 Blender 연동
- `core/meshes.py`: 프리미티브 메시(삼각형/사각형 판, 박스, UV 구 등) 생성, OBJ/STL 읽기·쓰기(numpy, 외부 의존 없음).
  생성/편집 중인 메시는 `MjcfDoc.mem_assets`(이미 구현: 메모리 에셋으로 컴파일)에 두고 프로젝트 저장 시 `meshes/`로 패키징(이미 구현).
- 가져오기 옵션: 단위(mm/cm/m), 축 변환(Blender Z-up/Y-forward ↔ MuJoCo), 기존 Body에 부착/새 Body 생성,
  시각 geom(`contype=0 conaffinity=0 group=2`) + 충돌 geom(볼록 껍질 또는 프리미티브 근사, `group=3`) 분리.
- 간단한 폴리곤 편집: 뷰포트 오버레이에 정점 표시(이미 있는 오버레이 패스), 화면 투영 거리로 정점 피킹, 기즈모로 이동 —
  드래그 중 `mjr_uploadMesh`로 미리보기 (3.0.1에서 메시 정점 순서 보존과 `mesh_pos/mesh_quat` 관계 확인 완료).

## Phase 4 — 얼굴 표정
- `core/expr.py`: 표정 파라미터(0..1) → Shape Key 가중합 + 제어 bone LBS → 머리 국소 좌표 정점.
- 렌더링: MJCF `<skin>` + `Viewport.scene_hooks`에서 `scn.skinvert/skinnormal` 덮어쓰기 (검증 완료, `docs/DESIGN.md` §8).
- `core/gltf.py`: glTF/GLB 읽기(accessor, morph target, `extras.targetNames`, skin) — Blender 내보내기 그대로 사용.
- 외부 API 예: `rig.set_expression("smile", 0.8)`, `rig.set_eye_rotation("left", yaw, pitch)`(관절), `rig.set_jaw_open(0.5)`(관절).
- 저장: `expressions/*.npz` + `project.json["expressions"]` (프리셋: 기본, 미소, 찡그림, 놀람, 눈 깜빡임, 윙크).

## Phase 5 — 전신 통합
- `library/anatomy.py`에 몸통·팔다리 생성기 추가 후 손/발/머리와 조립하는 `build_humanoid`.
- 현재도 `examples/assemble_humanoid.py`로 공식 humanoid에 부품을 조립한 전신 모델(81관절)을 만들 수 있음.

## Phase 6 — 품질
- 비동기 컴파일(대형 메시 모델), 크기 조절 기즈모, `<include>` 참조 보존 저장, 경고 패널 강화, 성능 프로파일링.
