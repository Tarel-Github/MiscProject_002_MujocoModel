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
| 평면 메시 | 시각 전용이면 허용 | 메시 단계에서 부피 계산 → `<mesh inertia="shell">` 필요 (3.0.1에는 속성 없음) | 기능 감지(`compiler.supports`) 후 지원 시에만 부여 |
| `MjvScene.skinvert` 덮어쓰기, `mjr_uploadMesh` | 동작 | 동작 | 양쪽 테스트 통과 |

검증: 전체 테스트 161개가 MuJoCo 3.0.1/numpy 1.26과 MuJoCo 3.17.0/numpy 2.5.4 양쪽에서 통과(Linux, Python 3.12).

## 8. 메시 파이프라인 (Phase 3)

- **가져오기는 파일 참조**: OBJ/STL은 원본 파일을 `<mesh file>`로 참조하고 단위는 `<mesh scale>`, 축 프리셋(MuJoCo / Blender 장면 Z-up·정면 −Y / Y-up)은
  geom 방향(quat)으로 표현합니다. 원본의 UV·법선이 보존되고 원본 파일은 수정되지 않습니다.
- **glTF/GLB**는 축 규약이 정해져 있어(Y-up, +Z 정면) 노드 변환과 축 변환을 정점에 적용해 메모리 OBJ 에셋으로 만듭니다.
- **생성·편집한 메시**는 `doc.mem_assets`(파일 키 → OBJ 바이트)에 두고 Undo 명령(`SetMemAsset`)으로 바꿉니다.
  컴파일 시 MuJoCo VFS로 전달되고, 저장 시 `meshes/`(프로젝트는 패키징)로 기록됩니다. 디스크의 메시를 처음 편집하면 `*_edit.obj`로 분기합니다.
- **시각/충돌 분리**: 시각 geom(`contype=conaffinity=0`, group 2, 질량 0)과 충돌 geom(group 3: MuJoCo 볼록 껍질, 또는 `type="box" mesh="…"`처럼
  MuJoCo가 메시에 맞춰 피팅하는 프리미티브). 평면 메시는 볼록 껍질을 만들 수 없어(qhull 오류) 자동으로 시각 전용이 됩니다.
- **정점 편집 모드**: 컴파일된 모델의 `mesh_vert`(3.0.1에서 정점 순서 보존과 `v_file = (R(mesh_quat)·v + mesh_pos)/scale` 관계를 확인)를
  드래그 중 직접 수정하고 `mjr_uploadMesh`로 GPU에 올려 미리보기, 놓으면 파일 프레임 데이터로 환산해 Undo 단계로 커밋합니다.
  MuJoCo는 메시당 정점 4개 이상, 일관된 면 방향을 요구하므로(위반 시 컴파일 오류로 표시) 삼각형 프리미티브는 무게중심 분할(정점 4)로 만듭니다.

## 9. 얼굴 표정 시스템 (Phase 4)

```
FaceRig (머리 국소 좌표)                         MuJoCo
  V0 + Σ wₖ·ΔShapeKeyₖ          (방식 B)
  → 제어 본 LBS (눈꺼풀 회전)     (방식 A)
  → 머리/턱 LBS ← data.xpos/xmat(head, jaw)  ←  mj_step / mj_forward (같은 MjData, 같은 data.time)
  → scn.skinvert / skinnormal 덮어쓰기  →  mjr_render
```
- 얼굴은 MJCF `<skin file="face.skn">`: 기준 프레임을 머리 국소 좌표로 두어(head bindpos = 0, jaw bindpos = 머리 기준 턱 휴지 포즈)
  머리 위치가 바뀌어도 바인딩이 유지되고, **순수 MuJoCo 뷰어에서도 턱 관절을 따라 얼굴이 움직입니다**(표정은 에디터/API에서 적용).
  .skn 형식은 MuJoCo 3.0.1 소스(`mjCSkin::LoadSKN`)에서 확인했고, 중립 표정에서 우리 계산이 MuJoCo 자체 스킨 계산과 float32 정밀도로 일치함을 테스트합니다.
- 눈 회전과 턱은 MuJoCo 관절(시뮬레이션·액추에이터·센서)이고, 얼굴 변형은 시각 전용입니다(물리 계산 불필요). `FaceController`가 둘을 한 API로 묶습니다.
- 시간 트랙(`Track`)은 `data.time`으로 평가되어 물리와 같은 시간 기준으로 동기화됩니다.
- 저장: `project.json["expressions"]`(파라미터·프리셋·현재 값·트랙·본 피벗) + `expressions/<rig>.npz`(정점, 면, 턱 가중치, Shape Key, 본 가중치) + `meshes/<skin>.skn`.
- Blender: Shape Key → glTF morph target(희소 accessor 포함), `jaw` 본 가중치 → 턱 바인딩. 내보내기가 법선 때문에 나눈 정점은 합치되,
  본 가중치나 Shape Key가 다른 정점(입술 틈처럼 의도적으로 나눈 이음매)은 유지합니다.
- 절차적 얼굴(`library/face.py`): 비균일 대칭 격자(정점 약 12,000), 안구 주위 구면 눈꺼풀 + 안검열, 턱에 묶인 아랫입술 틈,
  유한 지지 커널로 정의한 Shape Key(영역이 겹치지 않으면 정확히 독립).

## 10. 감각 데이터 인터페이스

`core/sensing.py`의 `SensorBus`는 MuJoCo 센서를 종류별 그룹(touch, proprio_pos, proprio_vel, vestibular, pose, force, effort)으로 묶어
관측 벡터와 레이아웃 명세(`spec()`)를 제공하고, `register()`로 MuJoCo 밖의 채널(표정 상태, 시선, 향후 통각·피부 모델)을 같은 `MjData` 위에 추가합니다.

## 11. 한국어 UI와 기능 배치

- 화면 문구는 코드에 한국어로 직접 씁니다(번역 파일 없음). MJCF 태그·속성 이름(`fromto`, `damping` 등)은 파일 형식 용어라 그대로 두고,
  그룹 제목·툴팁·트리의 종류 열은 한국어(`gui/ko.py`, `props.GROUP_KO`)로 표시합니다. 파일에는 항상 원래 MJCF 이름이 저장됩니다.
- 표정 파라미터·프리셋은 한국어 이름으로 보이고, 외부 API 이름(`blink_left`, `smile` 등)은 툴팁으로 남겨 스크립트와 화면을 대응시킵니다.
- 뷰포트 HUD는 MuJoCo 내장 글꼴(ASCII 전용)로는 한글을 그릴 수 없어 `mjr_overlay` 대신 뷰포트 위 Qt 라벨(마우스 이벤트 통과)로 그립니다.
- 한글 글꼴은 `theme._hangul_font()`가 대체 글꼴 목록(맑은 고딕, Apple SD Gothic Neo, Noto Sans CJK KR, 나눔고딕)에서 설치된 것을 골라 지정합니다.
- 추가·편집 기능은 `gui/addpanel.py`의 상시 도크(`추가·편집`)에 모으고, 같은 기능을 메뉴와 뷰포트 오른쪽 클릭(끌지 않은 클릭) 메뉴에도 둡니다.
  폴리곤 편집은 기본 형상이면 `meshops.convert_geom_to_mesh()`로 같은 위치·크기의 메시로 변환한 뒤 시작합니다
  (`fromto`가 기본 클래스에서 상속된 경우 `doc.class_without()`으로 상속받지 않는 부모 클래스로 옮기고 나머지 상속값은 명시값으로 복사).
- 손가락 하나 추가는 `anatomy.finger_part()`로 별도 문서에서 손가락 하나를 만든 뒤 일반 부품 조립(`parts.import_part`)으로 붙입니다.

## 12. 제약사항 (사실대로)

- Windows 11 GUI 실행은 검증하지 못함. 개발·검증은 Linux + Xvfb + Mesa 소프트웨어 렌더러.
- MuJoCo 엔진이 내는 컴파일 오류 문장은 영어 원문 그대로 표시됨(오류 위치 요소는 트리에 빨간색으로 표시).
- `<include>`는 불러올 때 인라인화되어 단일 파일로 저장됨.
- 얼굴 표정은 에디터 뷰포트와 API에서 적용됨. 순수 MuJoCo 뷰어는 스킨을 휴지 형태 + 턱 추종으로만 보여 줌(Shape Key 미적용).
- 절차적 얼굴은 양식화된 형상이며(사실적 얼굴 아님), 턱을 크게 열면 LBS 경계에 주름이 보임. 사실적 얼굴은 Blender에서 만들어 GLB로 가져오는 경로를 권장.
- Blender GLB의 제어 본은 head/jaw만 사용(그 외 본은 무시). glTF 가져오기는 메시·Shape Key·skin 가중치만 사용하며 재질/텍스처는 가져오지 않음.
- 메시 편집으로 다시 쓰는 OBJ에는 UV가 포함되지 않음(가져오기만 한 메시는 원본 파일을 참조하므로 UV 유지).
- 기즈모는 이동·회전만 제공(크기는 속성 패널의 `size`, 메시는 `<mesh scale>`).
- `mirror_duplicate`는 프리미티브 geom 기준. 메시 geom의 형상 자체는 반사하지 않음.
- 전신 인간형은 위치 제어만 있고 균형 제어기가 없어, 서 있는 자세에서 약 2초 후 넘어짐(물리적으로 정상).
- 부품 조립 시 대상 main 기본값 중 내장 기본값이 없는 속성(예: material)은 부품에도 적용될 수 있음(경고 출력).
- 해부학 부품의 단순화 사항은 `docs/ANATOMY.md` 참고.
