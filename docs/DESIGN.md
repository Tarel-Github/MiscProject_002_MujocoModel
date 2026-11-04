# 설계 문서 및 구현 제약사항

## 1. 기술 스택 결정과 근거

| 결정 | 근거 |
|---|---|
| Python 3.12 + PySide6 | MuJoCo 3.0.1 공식 휠이 cp38–cp312만 제공. PySide6는 Windows/Linux 공통 데스크톱 GUI. |
| 공식 `mujoco` 바인딩 (3.0.1 우선) | 컴파일·시뮬레이션·렌더링을 MuJoCo 자체에 맡김 (자체 물리 엔진/범용 렌더러 미구현). |
| 뷰포트 = MuJoCo 렌더러(`mjr_render`) | 범용 렌더러를 재구현하지 않고 MuJoCo와 동일한 화면(재질·그림자·반사·스킨). 편집용 오버레이(기즈모, 관절 범위 호)만 소규모 레거시 GL로 그림. |
| 문서 모델 = lxml DOM | `MjSpec`은 3.2+에만 존재하고 `to_xml()`은 주석·서식·일부 구성을 보존하지 않음. 원본 XML 트리를 진실 공급원으로 두면 미지원 요소도 그대로 저장됨. |
| Rust 미사용 | 편집 기능 완성도를 우선. MuJoCo Python 바인딩·PySide6 생태계가 가장 검증됨. |

## 2. 아키텍처

```
          ┌──────────────────────── core (GUI 비의존) ────────────────────────┐
 XML 파일 →│ MjcfDoc (lxml 트리 = 문서)  ←── Editor.push(Cmd) ── UndoStack        │
          │      │ compile_doc(복사본)                                         │
          │      ▼                                                             │
          │ CompileResult: MjModel + IdMap(요소↔ID) + 오류 요소                 │
          │ Editor.data (MjData): 관절 자세, 시뮬레이션 상태                      │
          └───────────────────────────────────────────────────────────────────┘
                 ▲ signals (doc_changed / compiled / selection_changed / state_changed)
          ┌──────┴──────── gui (PySide6) ─────────┐
          │ Viewport  TreePanel  PropsPanel  JointPanel  SimPanel  LogPanel │
          └───────────────────────────────────────┘
```

- GUI는 `Editor`의 메서드만 호출하고 신호를 구독합니다. 같은 API를 스크립트에서 그대로 쓸 수 있습니다
  (예: `examples/assemble_humanoid.py`, `library/anatomy.py`는 GUI 없이 동일 연산을 사용).
- 편집은 모두 명령(`SetAttrs`, `Insert`, `Remove`, `Move`, `Macro`)으로 lxml 트리에 직접 적용됩니다.
  명령이 요소 객체를 보유하므로 Undo/Redo 후에도 요소 동일성(선택, 트리 항목, ID 매핑)이 유지됩니다.
- 매 편집 후 문서를 재컴파일합니다(배치 안에서는 끝에서 한 번). 실패해도 마지막 유효 모델을 계속 표시하고
  오류 요소를 트리에서 빨간색으로 표시합니다.

## 3. 원본 보존과 MJCF 처리

- 파싱 시 주석·미지원 요소·속성 보존(`remove_comments=False`), 루트 밖 주석(라이선스 헤더)도 보존.
- 저장은 들여쓰기만 정규화합니다(공백 서식은 보존하지 않음). 속성 순서·값 문자열은 유지되며 Undo는 원래 문자열을 그대로 복원합니다.
- `<include>`는 불러올 때 **인라인으로 펼칩니다**(주석으로 범위 표시). 저장 결과는 단일 파일입니다. *(제약사항)*
- 다른 폴더로 저장하면 `meshdir/texturedir/assetdir` 또는 개별 `file` 경로를 새 위치 기준으로 재작성합니다.
- 원본 보호: 외부에서 연 파일에 "저장"하면 덮어쓰기 경고가 뜨고 기본 선택은 "다른 이름으로 저장"입니다.

### 컴파일과 ID 매핑
문서 복사본에서만 (1) 이름 없는 body/joint/geom/site/camera/light/액추에이터/센서에 예약 이름 `__mms_*`를 주입하고,
(2) 에셋 경로를 절대경로(또는 메모리 에셋)로 바꾼 뒤 `MjModel.from_xml_string`으로 컴파일합니다.
`mj_name2id`로 ID→요소 매핑을 만들고(에셋은 등장 순서), 오류 메시지의 `line = N`/`Object name = X`로 원본 요소를 찾습니다.
저장 파일에는 주입 이름이 들어가지 않습니다.

### 방향 지정자
MuJoCo 컴파일러와 같은 규칙으로 quat/axisangle/euler(소문자=내재, 대문자=외재, 혼합 허용)/xyaxes/zaxis/fromto를 해석합니다.
편집 시 요소가 쓰던 표기 방식(euler면 euler)을 유지합니다. 테스트가 MuJoCo 컴파일 결과와 직접 대조합니다.

### keyframe 재매핑
관절·액추에이터·mocap 구조가 바뀌는 편집(삭제, 이동, 관절 추가, 관절 type 변경, 부품 조립/교체)은
문서에서 직접 계산한 상태 벡터 레이아웃(MuJoCo 규칙: body 전위 DFS, body 내 관절은 XML 순서, free=7/ball=4/hinge·slide=1)으로
`<key>`의 qpos/qvel/ctrl/mpos/mquat를 요소 동일성 기준으로 재배치합니다. 새 관절은 qpos0(ref 또는 단위 쿼터니언/초기 포즈)로 채웁니다.
컴파일이 필요 없어 중간 상태가 일시적으로 무효(질량 없는 관절 body 등)여도 동작합니다.
`replicate/composite/flexcomp/attach`가 있는 모델은 확장 결과를 알 수 없어 재매핑하지 않고 경고합니다. *(제약사항)*

## 4. 뷰포트 렌더링 파이프라인

1. `mjv_updateScene` → 선택 강조(색 혼합 + emission), 시각/충돌 geom 표시 필터(`mjGEOM_NONE`으로 숨김)
2. `scene_hooks` (표정 시스템이 skin 정점을 덮어쓸 확장 지점)
3. MuJoCo 오프스크린 FBO에 `mjr_render` → 오버레이(기즈모, 관절 축·범위 호, 카메라 절두체, 축 표시) → Qt FBO로 `glBlitFramebuffer`

구현 중 확인한 사항:
- **Compatibility 프로파일만 요청해야 함.** 버전(3.3)을 함께 지정하면 Mesa/Qt 조합에서 Core 컨텍스트가 생성되어 MuJoCo 렌더러가 동작하지 않음.
- MuJoCo 3.x는 **reversed-Z**(`glClipControl(GL_LOWER_LEFT, GL_ZERO_TO_ONE)`, `glDepthFunc(GL_GREATER)`)로 렌더링합니다.
  고전적 `glOrtho/glFrustum`을 그대로 쓰면 클립 공간의 절반이 잘리므로 오버레이 동안 `NEGATIVE_ONE_TO_ONE`으로 바꿨다가 복원합니다.
- QOpenGLWidget의 기본 FBO는 0이 아니므로 MuJoCo의 윈도우 버퍼(FBO 0) 대신 오프스크린 버퍼 + 블릿을 사용합니다.
- 피킹은 `mjv_select`(geom) + 화면 투영 거리(관절 앵커, site). 투영 수학은 `mjr_render`의 절두체 계산과 같으며 GUI 테스트가 이를 검증합니다.

### 기즈모 드래그 실시간 미리보기
드래그 중에는 재컴파일하지 않고 컴파일된 모델 필드(`body_pos`, `geom_pos` 등)를 직접 수정합니다.
메시 geom은 컴파일 시 메시 관성 프레임으로 재배치되므로 시작 시점의 오프셋을 보존해 적용하고,
MuJoCo의 `geom_sameframe/site_sameframe` 최적화 플래그를 미리보기 동안 해제합니다(해제하지 않으면 `geom_pos` 변경이 무시됨).
놓는 순간 문서에 한 번의 Undo 단계로 커밋합니다. 변환은 `L' = P⁻¹ ∘ D ∘ P ∘ L`(P: 부모 현재 월드 포즈)이라 관절이 굽은 자세에서도 정확합니다.

## 5. 부품 시스템
`parts.extract_part` / `parts.import_part` / `parts.replace_body` — README 참고. 조립 시 대상 모델의 기본 클래스가
부품에 새어 들어가지 않도록, 부품의 main 기본값을 새 클래스(`<prefix>main`)로 복사하고 대상 main이 지정한 속성 중
부품이 지정하지 않은 것은 MuJoCo 내장 기본값으로 "중화"합니다(중화 값을 알 수 없는 속성은 경고).

## 6. 프로젝트 형식
```
project/
  model.xml       항상 순수 MuJoCo로 열리는 MJCF
  project.json    {"format": "mjstudio-project", "version": 1, "model": "model.xml",
                   "joint_groups": [...], "expressions": {...}, "meta": {...}}
  meshes/ textures/   패키징된 에셋 (저장 시 복사, compiler meshdir/texturedir 설정)
```
MJCF로 표현할 수 없는 데이터(관절 그룹, 향후 표정 리그)는 project.json에 저장하며, 모르는 키도 다시 저장할 때 보존합니다.

## 7. MuJoCo 버전별 차이와 마이그레이션

| 항목 | 3.0.1 | 이후 버전 | 이 프로젝트의 처리 |
|---|---|---|---|
| Python/numpy | cp38–cp312 휠, numpy 1.x 빌드 | numpy 2 지원 | 3.0.1은 `numpy<2` 고정 (`requirements.txt`) |
| `<position kv>` / `timeconst` | 없음 (스키마 오류) | 3.1+ | 감쇠는 관절 `damping`으로 부여. 속성 패널에 버전 주석 |
| 텍스처 데이터 필드 | `tex_rgb` | `tex_data` (3.2+) | 둘 다 확인 |
| `MjSpec` 편집 API | 없음 | 3.2+ (API 변경 잦음) | 사용하지 않음 (lxml 문서 모델) |
| `<frame>`, `<replicate>`, `<attach>` | 없음 | 3.1~3.2+ | 보존·표시는 되지만 keyframe 재매핑은 경고 후 생략 |
| `fromto` 프레임 방향 | z = from − to 방향 | 동일(3.17에서 테스트 통과) | 실제 컴파일러 동작에 맞춤 |

검증: 전체 테스트 116개가 MuJoCo 3.0.1/numpy 1.26과 MuJoCo 3.17.0/numpy 2.5.4 양쪽에서 통과(Linux, Python 3.12).

## 8. 얼굴 표정 시스템 설계 (Phase 4 — 미구현, 기술 검증 완료)

MuJoCo 렌더러는 Shape Key를 지원하지 않으므로 다음 경로를 택합니다(3.0.1에서 동작 확인):
- 얼굴 메시를 MJCF `<skin>`(시각 전용, 충돌 없음)으로 넣고 bone을 머리·턱 body에 바인딩 → 외부 MuJoCo 뷰어에서도 턱을 따라 움직임
- 에디터는 `mjv_updateScene` 이후 `scn.skinvert`/`skinnormal`을 **Shape Key 가중합 + 제어용 bone LBS** 결과로 덮어씀
  (`Viewport.scene_hooks`). 3.0.1에서 `skinvert` 수정이 `mjr_render`에 반영됨을 확인했습니다.
- 표정은 물리와 같은 `MjData`의 머리·턱 포즈와 `data.time`으로 평가되므로 시뮬레이션 시간 기준으로 동기화됩니다.
- Blender 데이터는 glTF/GLB의 morph target(Shape Key) + skin(JOINTS_0/WEIGHTS_0)로 가져오고, MJCF로 표현 불가한 Shape Key는
  `expressions/*.npz` + project.json에 보존합니다.
- 메시 대안 경로: 3.0.1은 메시 정점 순서를 보존하며 `v_orig = R(mesh_quat)·v + mesh_pos` 관계가 성립(확인)하므로
  `mjr_uploadMesh`로 메시 geom을 변형하는 방식도 가능합니다.

## 9. 제약사항 (사실대로)

- Windows 11 GUI 실행은 검증하지 못함. 개발·검증은 Linux + Xvfb + Mesa 소프트웨어 렌더러.
- `<include>`는 불러올 때 인라인화되어 단일 파일로 저장됨.
- 폴리곤 편집, OBJ/STL/glTF 가져오기, 메시 내보내기(Phase 3), 얼굴 표정(Phase 4)은 아직 구현되지 않음.
- 기즈모는 이동·회전만 제공(크기는 속성 패널의 `size`로 편집).
- `mirror_duplicate`는 프리미티브 geom 기준. 메시 geom의 형상 자체는 반사하지 않음.
- 부품 조립 시 대상 main 기본값 중 내장 기본값이 없는 속성(예: material)은 부품에도 적용될 수 있음(경고 출력).
- 해부학 부품의 단순화 사항은 `docs/ANATOMY.md` 참고.
