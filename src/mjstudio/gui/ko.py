"""화면 표시용 한국어 이름 (MJCF 태그 → 한국어). MJCF 파일에는 원래 이름이 그대로 저장됩니다."""
from __future__ import annotations

TAG_KO = {"body": "바디", "geom": "형상", "joint": "관절", "freejoint": "자유 관절", "site": "사이트",
          "camera": "카메라", "light": "조명", "inertial": "관성", "frame": "프레임", "mesh": "메시",
          "texture": "텍스처", "material": "재질", "skin": "스킨", "default": "기본 클래스", "position": "위치 액추에이터",
          "motor": "모터", "velocity": "속도 액추에이터", "general": "일반 액추에이터", "weld": "용접 구속",
          "connect": "연결 구속", "joint_eq": "관절 구속", "key": "키프레임", "option": "옵션", "compiler": "컴파일러",
          "worldbody": "월드", "touch": "접촉(터치) 센서", "jointpos": "관절 위치 센서", "jointvel": "관절 속도 센서",
          "accelerometer": "가속도 센서", "gyro": "자이로 센서", "force": "힘 센서", "torque": "토크 센서",
          "tendon": "텐던", "fixed": "고정 텐던", "spatial": "공간 텐던", "pair": "접촉 쌍",
          "exclude": "접촉 제외", "hfield": "높이맵", "numeric": "숫자 데이터", "text": "텍스트 데이터",
          "framepos": "위치 센서", "framequat": "방향 센서", "framexaxis": "X축 방향 센서", "velocimeter": "속도 센서",
          "subtreecom": "질량 중심 센서", "actuatorfrc": "액추에이터 힘 센서", "size": "크기 설정",
          "visual": "표시 설정", "statistic": "통계", "custom": "사용자 정의", "extension": "확장", "flex": "플렉스",
          "composite": "복합체", "plugin": "플러그인", "replicate": "반복", "attach": "부착"}


def tag_ko(tag: str) -> str:
    """한국어 이름 (없으면 태그 그대로)."""
    return TAG_KO.get(tag, tag)
