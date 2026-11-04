"""MJCF attribute schema (MuJoCo 3.0.x XML reference subset).

The schema drives the generic property editor and the reference-aware rename.
It is *descriptive*, never restrictive: attributes not listed here are kept and
shown as raw text, and unknown elements are fully preserved.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# attribute kinds
STR, NAME, INT, REAL, VEC, BOOL, ENUM, RGBA, REF, ORIENT = (
    "str", "name", "int", "real", "vec", "bool", "enum", "rgba", "ref", "orient")

ORIENT_KEYS = ("quat", "axisangle", "xyaxes", "zaxis", "euler")


@dataclass(frozen=True)
class Attr:
    name: str
    kind: str = STR
    default: str | None = None       # MuJoCo built-in default (string form)
    n: int | None = None             # fixed vector length
    nmax: int | None = None          # max vector length (variable)
    choices: tuple[str, ...] = ()
    ref: str | None = None           # referenced object kind (body, joint, mesh, ...)
    group: str = "General"
    tip: str = ""
    angle: bool = False              # value expressed in compiler angle units


def A(name, kind=STR, default=None, n=None, nmax=None, choices=(), ref=None, group="General", tip="", angle=False):
    return Attr(name, kind, default, n, nmax, tuple(choices), ref, group, tip, angle)


AUTO_BOOL = ("false", "true", "auto")
TRANSFORM = "Transform"
PHYS = "Physics"
CONTACT = "Contact"
VISUAL = "Visual"
LIMITS = "Limits"
SOLVER = "Solver"

_ORIENT = [
    A("quat", VEC, "1 0 0 0", n=4, group=TRANSFORM, tip="방향 쿼터니언 (w x y z)"),
    A("axisangle", VEC, None, n=4, group=TRANSFORM, tip="축 (x y z)과 각도 (compiler의 각도 단위)"),
    A("xyaxes", VEC, None, n=6, group=TRANSFORM, tip="프레임의 X축과 Y축"),
    A("zaxis", VEC, None, n=3, group=TRANSFORM, tip="프레임의 Z축 (+Z에서 최소 회전)"),
    A("euler", VEC, None, n=3, group=TRANSFORM, tip="오일러 각 (compiler의 eulerseq/각도 단위)"),
]
_POS = A("pos", VEC, "0 0 0", n=3, group=TRANSFORM, tip="부모 기준 위치")
_USER = A("user", VEC, None, group="User", tip="사용자 데이터")
_SOLREF = "0.02 1"
_SOLIMP = "0.9 0.95 0.001 0.5 2"

BODY = [
    A("name", NAME), A("childclass", REF, ref="class", tip="자식 요소에 적용할 기본 클래스"),
    _POS, *_ORIENT,
    A("mocap", BOOL, "false", group=PHYS, tip="모캡 바디 (월드 바로 아래만)"),
    A("gravcomp", REAL, "0", group=PHYS, tip="중력 보상 (무게 대비 비율)"),
    _USER,
]

INERTIAL = [
    _POS, *_ORIENT,
    A("mass", REAL, None, group=PHYS, tip="질량 (kg)"),
    A("diaginertia", VEC, None, n=3, group=PHYS, tip="관성 프레임 기준 대각 관성"),
    A("fullinertia", VEC, None, n=6, group=PHYS, tip="전체 관성 Ixx Iyy Izz Ixy Ixz Iyz"),
]

JOINT = [
    A("name", NAME), A("class", REF, ref="class"),
    A("type", ENUM, "hinge", choices=("free", "ball", "slide", "hinge")),
    A("group", INT, "0", group=VISUAL),
    A("pos", VEC, "0 0 0", n=3, group=TRANSFORM, tip="바디 기준 관절 중심 위치"),
    A("axis", VEC, "0 0 1", n=3, group=TRANSFORM, tip="바디 기준 회전/이동 축"),
    A("limited", ENUM, "auto", choices=AUTO_BOOL, group=LIMITS),
    A("range", VEC, "0 0", n=2, group=LIMITS, tip="관절 범위 (경첩/볼: 도 또는 rad, 슬라이드: m)", angle=True),
    A("margin", REAL, "0", group=LIMITS),
    A("solreflimit", VEC, _SOLREF, nmax=2, group=LIMITS),
    A("solimplimit", VEC, _SOLIMP, nmax=5, group=LIMITS),
    A("ref", REAL, "0", group=PHYS, tip="기준 위치 (qpos0)", angle=True),
    A("springref", REAL, "0", group=PHYS, angle=True),
    A("stiffness", REAL, "0", group=PHYS),
    A("damping", REAL, "0", group=PHYS),
    A("armature", REAL, "0", group=PHYS),
    A("frictionloss", REAL, "0", group=PHYS),
    A("springdamper", VEC, "0 0", n=2, group=PHYS),
    A("solreffriction", VEC, _SOLREF, nmax=2, group=PHYS),
    A("solimpfriction", VEC, _SOLIMP, nmax=5, group=PHYS),
    A("actuatorfrclimited", ENUM, "auto", choices=AUTO_BOOL, group=LIMITS),
    A("actuatorfrcrange", VEC, "0 0", n=2, group=LIMITS),
    _USER,
]

FREEJOINT = [A("name", NAME), A("group", INT, "0", group=VISUAL)]

GEOM_TYPES = ("plane", "hfield", "sphere", "capsule", "ellipsoid", "cylinder", "box", "mesh", "sdf")
GEOM = [
    A("name", NAME), A("class", REF, ref="class"),
    A("type", ENUM, "sphere", choices=GEOM_TYPES),
    A("size", VEC, "0 0 0", nmax=3, tip="종류별 크기 값 (구: 반지름, 캡슐: 반지름 반길이, 박스: 반폭 3개 …)"),
    A("mesh", REF, ref="mesh"), A("hfield", REF, ref="hfield"),
    A("fromto", VEC, None, n=6, group=TRANSFORM, tip="대체 프레임: 선분 양 끝점"),
    _POS, *_ORIENT,
    A("material", REF, ref="material", group=VISUAL),
    A("rgba", RGBA, "0.5 0.5 0.5 1", n=4, group=VISUAL),
    A("group", INT, "0", group=VISUAL),
    A("contype", INT, "1", group=CONTACT, tip="접촉 종류 비트마스크"),
    A("conaffinity", INT, "1", group=CONTACT, tip="접촉 상대 비트마스크"),
    A("condim", ENUM, "3", choices=("1", "3", "4", "6"), group=CONTACT),
    A("priority", INT, "0", group=CONTACT),
    A("friction", VEC, "1 0.005 0.0001", nmax=3, group=CONTACT),
    A("solmix", REAL, "1", group=CONTACT),
    A("solref", VEC, _SOLREF, nmax=2, group=CONTACT),
    A("solimp", VEC, _SOLIMP, nmax=5, group=CONTACT),
    A("margin", REAL, "0", group=CONTACT), A("gap", REAL, "0", group=CONTACT),
    A("mass", REAL, None, group=PHYS, tip="질량 (지정하면 밀도 대신 사용)"),
    A("density", REAL, "1000", group=PHYS),
    A("shellinertia", BOOL, "false", group=PHYS),
    A("fitscale", REAL, "1", group=PHYS),
    A("fluidshape", ENUM, "none", choices=("none", "ellipsoid"), group=PHYS),
    A("fluidcoef", VEC, "0.5 0.25 1.5 1.0 1.0", n=5, group=PHYS),
    _USER,
]

SITE_TYPES = ("sphere", "capsule", "ellipsoid", "cylinder", "box")
SITE = [
    A("name", NAME), A("class", REF, ref="class"),
    A("type", ENUM, "sphere", choices=SITE_TYPES),
    A("size", VEC, "0.005 0.005 0.005", nmax=3),
    A("fromto", VEC, None, n=6, group=TRANSFORM),
    _POS, *_ORIENT,
    A("material", REF, ref="material", group=VISUAL),
    A("rgba", RGBA, "0.5 0.5 0.5 1", n=4, group=VISUAL),
    A("group", INT, "0", group=VISUAL),
    _USER,
]

CAM_MODES = ("fixed", "track", "trackcom", "targetbody", "targetbodycom")
CAMERA = [
    A("name", NAME), A("class", REF, ref="class"),
    A("mode", ENUM, "fixed", choices=CAM_MODES), A("target", REF, ref="body"),
    _POS, *_ORIENT,
    A("fovy", REAL, "45", group=VISUAL), A("ipd", REAL, "0.068", group=VISUAL),
    A("resolution", VEC, "1 1", n=2, group=VISUAL),
    _USER,
]

LIGHT = [
    A("name", NAME), A("class", REF, ref="class"),
    A("mode", ENUM, "fixed", choices=CAM_MODES), A("target", REF, ref="body"),
    _POS, A("dir", VEC, "0 0 -1", n=3, group=TRANSFORM),
    A("directional", BOOL, "false", group=VISUAL), A("castshadow", BOOL, "true", group=VISUAL),
    A("active", BOOL, "true", group=VISUAL),
    A("attenuation", VEC, "1 0 0", n=3, group=VISUAL), A("cutoff", REAL, "45", group=VISUAL),
    A("exponent", REAL, "10", group=VISUAL),
    A("ambient", VEC, "0 0 0", n=3, group=VISUAL), A("diffuse", VEC, "0.7 0.7 0.7", n=3, group=VISUAL),
    A("specular", VEC, "0.3 0.3 0.3", n=3, group=VISUAL),
]

FRAME = [A("name", NAME), A("childclass", REF, ref="class"), _POS, *_ORIENT]

# ------------------------------------------------------------------ assets
MESH = [
    A("name", NAME), A("class", REF, ref="class"),
    A("file", STR, tip="메시 파일 (STL/OBJ/MSH), meshdir 기준 경로"),
    A("scale", VEC, "1 1 1", n=3), A("refpos", VEC, "0 0 0", n=3), A("refquat", VEC, "1 0 0 0", n=4),
    A("smoothnormal", BOOL, "false"), A("maxhullvert", INT, "-1"),
    A("vertex", VEC), A("face", VEC), A("normal", VEC), A("texcoord", VEC),
    A("content_type", STR),
]
TEXTURE = [
    A("name", NAME), A("type", ENUM, "cube", choices=("2d", "cube", "skybox")),
    A("file", STR), A("builtin", ENUM, "none", choices=("none", "gradient", "checker", "flat")),
    A("rgb1", VEC, "0.8 0.8 0.8", n=3), A("rgb2", VEC, "0.5 0.5 0.5", n=3),
    A("mark", ENUM, "none", choices=("none", "edge", "cross", "random")), A("markrgb", VEC, "0 0 0", n=3),
    A("random", REAL, "0.01"), A("width", INT, "0"), A("height", INT, "0"),
    A("gridsize", VEC, "1 1", n=2), A("gridlayout", STR),
    A("fileright", STR), A("fileleft", STR), A("fileup", STR), A("filedown", STR),
    A("filefront", STR), A("fileback", STR), A("hflip", BOOL, "false"), A("vflip", BOOL, "false"),
    A("content_type", STR),
]
MATERIAL = [
    A("name", NAME), A("class", REF, ref="class"), A("texture", REF, ref="texture"),
    A("rgba", RGBA, "1 1 1 1", n=4), A("texrepeat", VEC, "1 1", n=2), A("texuniform", BOOL, "false"),
    A("emission", REAL, "0"), A("specular", REAL, "0.5"), A("shininess", REAL, "0.5"),
    A("reflectance", REAL, "0"),
]
HFIELD = [A("name", NAME), A("file", STR), A("nrow", INT, "0"), A("ncol", INT, "0"),
          A("size", VEC, None, n=4), A("elevation", VEC), A("content_type", STR)]
SKIN = [A("name", NAME), A("file", STR), A("material", REF, ref="material"),
        A("rgba", RGBA, "0.5 0.5 0.5 1", n=4), A("inflate", REAL, "0"), A("group", INT, "0"),
        A("vertex", VEC), A("texcoord", VEC), A("face", VEC)]
SKIN_BONE = [A("body", REF, ref="body"), A("bindpos", VEC, n=3), A("bindquat", VEC, n=4),
             A("vertid", VEC), A("vertweight", VEC)]

# --------------------------------------------------------------- actuators
_ACT_COMMON = [
    A("name", NAME), A("class", REF, ref="class"), A("group", INT, "0"),
    A("joint", REF, ref="joint", group="Transmission"),
    A("jointinparent", REF, ref="joint", group="Transmission"),
    A("tendon", REF, ref="tendon", group="Transmission"),
    A("site", REF, ref="site", group="Transmission"),
    A("refsite", REF, ref="site", group="Transmission"),
    A("cranksite", REF, ref="site", group="Transmission"),
    A("body", REF, ref="body", group="Transmission"),
    A("slidersite", REF, ref="site", group="Transmission"),
    A("cranklength", REAL, "0", group="Transmission"),
    A("gear", VEC, "1 0 0 0 0 0", nmax=6, group="Transmission"),
    A("ctrllimited", ENUM, "auto", choices=AUTO_BOOL, group=LIMITS),
    A("ctrlrange", VEC, "0 0", n=2, group=LIMITS),
    A("forcelimited", ENUM, "auto", choices=AUTO_BOOL, group=LIMITS),
    A("forcerange", VEC, "0 0", n=2, group=LIMITS),
    A("actlimited", ENUM, "auto", choices=AUTO_BOOL, group=LIMITS),
    A("actrange", VEC, "0 0", n=2, group=LIMITS),
    A("lengthrange", VEC, "0 0", n=2, group=LIMITS),
    _USER,
]
ACTUATORS = {
    "general": _ACT_COMMON + [
        A("dyntype", ENUM, "none", choices=("none", "integrator", "filter", "filterexact", "muscle", "user"), group="Dynamics"),
        A("gaintype", ENUM, "fixed", choices=("fixed", "affine", "muscle", "user"), group="Dynamics"),
        A("biastype", ENUM, "none", choices=("none", "affine", "muscle", "user"), group="Dynamics"),
        A("dynprm", VEC, "1 0 0 0 0 0 0 0 0 0", nmax=10, group="Dynamics"),
        A("gainprm", VEC, "1 0 0 0 0 0 0 0 0 0", nmax=10, group="Dynamics"),
        A("biasprm", VEC, "0 0 0 0 0 0 0 0 0 0", nmax=10, group="Dynamics"),
        A("actearly", BOOL, "false", group="Dynamics"), A("actdim", INT, "-1", group="Dynamics")],
    "motor": _ACT_COMMON,
    "position": _ACT_COMMON + [A("kp", REAL, "1", group="Dynamics"),
                               A("kv", REAL, "0", group="Dynamics", tip="MuJoCo 3.1 이상에서만 (3.0.x에는 없음)"),
                               A("timeconst", REAL, "0", group="Dynamics", tip="MuJoCo 3.1 이상에서만")],
    "velocity": _ACT_COMMON + [A("kv", REAL, "1", group="Dynamics")],
    "intvelocity": _ACT_COMMON + [A("kp", REAL, "1", group="Dynamics"), A("kv", REAL, "0", group="Dynamics")],
    "damper": _ACT_COMMON + [A("kv", REAL, "1", group="Dynamics")],
    "cylinder": _ACT_COMMON + [A("timeconst", REAL, "1", group="Dynamics"), A("area", REAL, "1", group="Dynamics"),
                               A("diameter", REAL, None, group="Dynamics"), A("bias", VEC, "0 0 0", n=3, group="Dynamics")],
    "muscle": _ACT_COMMON + [A("timeconst", VEC, "0.01 0.04", n=2, group="Dynamics"), A("range", VEC, "0.75 1.05", n=2, group="Dynamics"),
                             A("force", REAL, "-1", group="Dynamics"), A("scale", REAL, "200", group="Dynamics"),
                             A("lmin", REAL, "0.5", group="Dynamics"), A("lmax", REAL, "1.6", group="Dynamics"),
                             A("vmax", REAL, "1.5", group="Dynamics"), A("fpmax", REAL, "1.3", group="Dynamics"),
                             A("fvmax", REAL, "1.2", group="Dynamics")],
    "adhesion": [A("name", NAME), A("class", REF, ref="class"), A("group", INT, "0"),
                 A("body", REF, ref="body", group="Transmission"), A("gain", REAL, "1", group="Dynamics"),
                 A("ctrllimited", ENUM, "auto", choices=AUTO_BOOL, group=LIMITS),
                 A("ctrlrange", VEC, "0 0", n=2, group=LIMITS),
                 A("forcelimited", ENUM, "auto", choices=AUTO_BOOL, group=LIMITS),
                 A("forcerange", VEC, "0 0", n=2, group=LIMITS), _USER],
}

# ----------------------------------------------------------------- sensors
_SENS_COMMON = [A("name", NAME), A("noise", REAL, "0"), A("cutoff", REAL, "0"), _USER]
_OBJTYPES = ("body", "xbody", "geom", "site", "camera")
SENSOR_SITE = ("touch", "accelerometer", "velocimeter", "gyro", "force", "torque", "magnetometer", "rangefinder")
SENSOR_JOINT = ("jointpos", "jointvel", "ballquat", "ballangvel", "jointlimitpos", "jointlimitvel",
                "jointlimitfrc", "jointactuatorfrc")
SENSOR_TENDON = ("tendonpos", "tendonvel", "tendonlimitpos", "tendonlimitvel", "tendonlimitfrc")
SENSOR_ACT = ("actuatorpos", "actuatorvel", "actuatorfrc")
SENSOR_FRAME = ("framepos", "framequat", "framexaxis", "frameyaxis", "framezaxis", "framelinvel",
                "frameangvel", "framelinacc", "frameangacc")
SENSOR_SUBTREE = ("subtreecom", "subtreelinvel", "subtreeangmom")
SENSORS: dict[str, list[Attr]] = {}
for _t in SENSOR_SITE:
    SENSORS[_t] = _SENS_COMMON + [A("site", REF, ref="site")]
for _t in SENSOR_JOINT:
    SENSORS[_t] = _SENS_COMMON + [A("joint", REF, ref="joint")]
for _t in SENSOR_TENDON:
    SENSORS[_t] = _SENS_COMMON + [A("tendon", REF, ref="tendon")]
for _t in SENSOR_ACT:
    SENSORS[_t] = _SENS_COMMON + [A("actuator", REF, ref="actuator")]
for _t in SENSOR_FRAME:
    SENSORS[_t] = _SENS_COMMON + [A("objtype", ENUM, None, choices=_OBJTYPES), A("objname", REF, ref="@objtype"),
                                  A("reftype", ENUM, None, choices=_OBJTYPES), A("refname", REF, ref="@reftype")]
for _t in SENSOR_SUBTREE:
    SENSORS[_t] = _SENS_COMMON + [A("body", REF, ref="body")]
SENSORS["clock"] = list(_SENS_COMMON)
SENSORS["user"] = _SENS_COMMON + [A("objtype", STR), A("objname", STR), A("datatype", STR),
                                  A("needstage", STR), A("dim", INT)]

# ------------------------------------------------------- equality / tendon
_EQ_COMMON = [A("name", NAME), A("class", REF, ref="class"), A("active", BOOL, "true"),
              A("solref", VEC, _SOLREF, nmax=2, group=SOLVER), A("solimp", VEC, _SOLIMP, nmax=5, group=SOLVER)]
EQUALITY = {
    "connect": _EQ_COMMON + [A("body1", REF, ref="body"), A("body2", REF, ref="body"), A("anchor", VEC, n=3)],
    "weld": _EQ_COMMON + [A("body1", REF, ref="body"), A("body2", REF, ref="body"), A("relpose", VEC, n=7),
                          A("anchor", VEC, "0 0 0", n=3), A("torquescale", REAL, "1")],
    "joint": _EQ_COMMON + [A("joint1", REF, ref="joint"), A("joint2", REF, ref="joint"),
                           A("polycoef", VEC, "0 1 0 0 0", n=5, tip="joint1 = c0 + c1*j2 + c2*j2^2 + ...")],
    "tendon": _EQ_COMMON + [A("tendon1", REF, ref="tendon"), A("tendon2", REF, ref="tendon"),
                            A("polycoef", VEC, "0 1 0 0 0", n=5)],
}
_TEND_COMMON = [A("name", NAME), A("class", REF, ref="class"), A("group", INT, "0"),
                A("limited", ENUM, "auto", choices=AUTO_BOOL, group=LIMITS), A("range", VEC, "0 0", n=2, group=LIMITS),
                A("solreflimit", VEC, _SOLREF, nmax=2, group=LIMITS), A("solimplimit", VEC, _SOLIMP, nmax=5, group=LIMITS),
                A("frictionloss", REAL, "0", group=PHYS), A("springlength", VEC, "-1 -1", nmax=2, group=PHYS),
                A("stiffness", REAL, "0", group=PHYS), A("damping", REAL, "0", group=PHYS),
                A("margin", REAL, "0", group=LIMITS), _USER]
TENDON = {
    "spatial": _TEND_COMMON + [A("width", REAL, "0.003", group=VISUAL), A("material", REF, ref="material", group=VISUAL),
                               A("rgba", RGBA, "0.5 0.5 0.5 1", n=4, group=VISUAL)],
    "fixed": _TEND_COMMON,
}

SCHEMA: dict[str, list[Attr]] = {
    "body": BODY, "inertial": INERTIAL, "joint": JOINT, "freejoint": FREEJOINT, "geom": GEOM,
    "site": SITE, "camera": CAMERA, "light": LIGHT, "frame": FRAME,
    "mesh": MESH, "texture": TEXTURE, "material": MATERIAL, "hfield": HFIELD, "skin": SKIN,
    "spatial": TENDON["spatial"], "fixed": TENDON["fixed"],
    "pair": [A("name", NAME), A("class", REF, ref="class"), A("geom1", REF, ref="geom"), A("geom2", REF, ref="geom"),
             A("condim", ENUM, "3", choices=("1", "3", "4", "6")), A("friction", VEC, "1 1 0.005 0.0001 0.0001", n=5),
             A("solref", VEC, _SOLREF, nmax=2), A("solreffriction", VEC, "0 0", nmax=2),
             A("solimp", VEC, _SOLIMP, nmax=5), A("margin", REAL, "0"), A("gap", REAL, "0")],
    "exclude": [A("name", NAME), A("body1", REF, ref="body"), A("body2", REF, ref="body")],
    "key": [A("name", NAME), A("time", REAL, "0"), A("qpos", VEC), A("qvel", VEC), A("act", VEC),
            A("ctrl", VEC), A("mpos", VEC), A("mquat", VEC)],
    "mujoco": [A("model", STR, "MuJoCo Model")],
    "compiler": [A("angle", ENUM, "degree", choices=("degree", "radian")), A("eulerseq", STR, "xyz"),
                 A("autolimits", BOOL, "true"), A("meshdir", STR), A("texturedir", STR), A("assetdir", STR),
                 A("inertiafromgeom", ENUM, "auto", choices=AUTO_BOOL), A("boundmass", REAL, "0"),
                 A("boundinertia", REAL, "0"), A("settotalmass", REAL, "-1"), A("balanceinertia", BOOL, "false"),
                 A("strippath", BOOL, "false"), A("fitaabb", BOOL, "false"), A("discardvisual", BOOL, "false"),
                 A("convexhull", BOOL, "true"), A("usethread", BOOL, "true"), A("fusestatic", BOOL, "false"),
                 A("inertiagrouprange", VEC, "0 5", n=2), A("exactmeshinertia", BOOL, "false")],
    "option": [A("timestep", REAL, "0.002"), A("gravity", VEC, "0 0 -9.81", n=3),
               A("integrator", ENUM, "Euler", choices=("Euler", "RK4", "implicit", "implicitfast")),
               A("cone", ENUM, "pyramidal", choices=("pyramidal", "elliptic")),
               A("solver", ENUM, "Newton", choices=("PGS", "CG", "Newton")),
               A("iterations", INT, "100"), A("ls_iterations", INT, "50"), A("tolerance", REAL, "1e-8"),
               A("impratio", REAL, "1"), A("noslip_iterations", INT, "0"),
               A("density", REAL, "0"), A("viscosity", REAL, "0"), A("wind", VEC, "0 0 0", n=3),
               A("magnetic", VEC, "0 -0.5 0", n=3), A("jacobian", ENUM, "auto", choices=("dense", "sparse", "auto"))],
}
SCHEMA.update(ACTUATORS)
SCHEMA.update(SENSORS)
# equality children share tag names with joint/tendon -> resolved by parent section
EQUALITY_SCHEMA = EQUALITY


def attrs_for(tag: str, parent_tag: str | None = None) -> list[Attr]:
    """Schema attribute list for an element tag (context-sensitive for equality)."""
    if parent_tag == "equality" and tag in EQUALITY:
        return EQUALITY[tag]
    if parent_tag == "fixed" and tag == "joint":
        return [A("joint", REF, ref="joint"), A("coef", REAL)]
    if parent_tag == "spatial":
        if tag == "site":
            return [A("site", REF, ref="site")]
        if tag == "geom":
            return [A("geom", REF, ref="geom"), A("sidesite", REF, ref="site")]
        if tag == "pulley":
            return [A("divisor", REAL)]
    if parent_tag == "skin" and tag == "bone":
        return SKIN_BONE
    if parent_tag == "default":
        # defaults accept the same attributes as the element (minus name)
        return [a for a in SCHEMA.get(tag, []) if a.kind != NAME]
    return SCHEMA.get(tag, [])


def attr_spec(tag: str, name: str, parent_tag: str | None = None) -> Attr | None:
    for a in attrs_for(tag, parent_tag):
        if a.name == name:
            return a
    return None


# Element kinds that can be referenced by name, keyed by the MJCF tags that define them.
NAMED_KINDS = {
    "body": ("body",), "joint": ("joint", "freejoint"), "geom": ("geom",), "site": ("site",),
    "camera": ("camera",), "light": ("light",), "mesh": ("mesh",), "material": ("material",),
    "texture": ("texture",), "hfield": ("hfield",), "skin": ("skin",), "tendon": ("spatial", "fixed"),
    "actuator": tuple(ACTUATORS.keys()), "sensor": tuple(SENSORS.keys()),
}

# sections of the <mujoco> root that hold lists of elements
SECTIONS = ("compiler", "option", "size", "visual", "statistic", "default", "extension", "custom",
            "asset", "worldbody", "deformable", "contact", "equality", "tendon", "actuator",
            "sensor", "keyframe")

# tags living in the kinematic tree
TREE_TAGS = {"body", "geom", "joint", "freejoint", "site", "camera", "light", "inertial", "frame",
             "replicate", "composite", "flexcomp", "plugin", "attach"}

# MuJoCo object type codes (mjtObj) by tag, for tags that compile to objects
OBJ_TAGS = {"body": "mjOBJ_BODY", "joint": "mjOBJ_JOINT", "freejoint": "mjOBJ_JOINT",
            "geom": "mjOBJ_GEOM", "site": "mjOBJ_SITE", "camera": "mjOBJ_CAMERA",
            "light": "mjOBJ_LIGHT", "mesh": "mjOBJ_MESH", "material": "mjOBJ_MATERIAL",
            "texture": "mjOBJ_TEXTURE", "hfield": "mjOBJ_HFIELD", "skin": "mjOBJ_SKIN",
            "spatial": "mjOBJ_TENDON", "fixed": "mjOBJ_TENDON", "pair": "mjOBJ_PAIR",
            "exclude": "mjOBJ_EXCLUDE", "key": "mjOBJ_KEY"}
for _t in ACTUATORS:
    OBJ_TAGS[_t] = "mjOBJ_ACTUATOR"
for _t in SENSORS:
    OBJ_TAGS[_t] = "mjOBJ_SENSOR"
EQ_OBJ = "mjOBJ_EQUALITY"


@dataclass
class RefSite:
    """An attribute that references another named element."""
    tag: str
    attr: str
    kind: str
    parent_tag: str | None = None


def reference_attrs() -> list[RefSite]:
    out: list[RefSite] = []
    for tag, attrs in SCHEMA.items():
        for a in attrs:
            if a.kind == REF and a.ref and not a.ref.startswith("@") and a.ref != "class":
                out.append(RefSite(tag, a.name, a.ref))
    for tag, attrs in EQUALITY.items():
        for a in attrs:
            if a.kind == REF and a.ref != "class":
                out.append(RefSite(tag, a.name, a.ref, "equality"))
    out += [RefSite("joint", "joint", "joint", "fixed"), RefSite("site", "site", "site", "spatial"),
            RefSite("geom", "geom", "geom", "spatial"), RefSite("geom", "sidesite", "site", "spatial"),
            RefSite("bone", "body", "body", "skin")]
    return out
