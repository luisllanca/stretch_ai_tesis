"""
Offline DynaMem — Orbbec dataset, con export de bounding boxes en las consultas

Es una variante de offline_dynamem.py que además de la posición 3D de cada consulta,
exporta un PNG del frame usado con el bounding box del objeto detectado dibujado
encima (o un aviso si no hubo detección y se usó solo similitud de features).

También permite elegir la carpeta de salida en cada lanzamiento con --output-dir.
Dentro de esa carpeta, las cosas "pesadas" (rgb*.jpg/npy, depth*.npy, intrinsics*.npy,
pose*.npy, mapa.pkl, mapa_2d.png, bbox_*.png) quedan en una subcarpeta frames/, mientras
que query_results.json (el resultado de las consultas) queda al mismo nivel que frames/.

Estructura de salida:
    <output_dir>/
        query_results.json
        frames/
            rgb1.jpg  rgb1.npy  depth1.npy  intrinsics1.npy  pose1.npy  ...
            mapa.pkl
            mapa_2d.png
            bbox_<consulta>_frame<N>.png

Estructura esperada del dataset (igual que offline_dynamem.py):
    <data_dir>/
        rgb_data_XXXXXXXX_XXXXXX.csv   ← image_path, x, y, yaw
        <fecha>_<hora>/
            orbbec_rgb/    *.jpg
            orbbec_depth/  *.npz  (clave 'depth', uint16, valores en mm)

Uso:
    cd stretch_ai/src
    python ../examples/offline_dynamem_bbox.py \
        --data-dir ../robot_20251024_074000 \
        --output-dir ../salida_consulta1 \
        --fx 1079.5 --fy 1079.5 --cx 960 --cy 540
"""

import argparse
import csv
import datetime
import inspect
import json
import os
import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # sin display
import matplotlib.pyplot as plt
import cv2
import numpy as np
import torch
from PIL import Image

# ──────────────────────────────────────────────────────────────────────────────
# Intrínsecos por defecto — Orbbec Femto Mega a 1920×1080
# Cámbialos si tienes calibración propia
# ──────────────────────────────────────────────────────────────────────────────
DEFAULT_FX = 1079.5
DEFAULT_FY = 1079.5
DEFAULT_CX = 960.0
DEFAULT_CY = 540.0

# Resolución de salida: mantenemos aspecto 16:9 (1/3 de la original)
TARGET_W = 640
TARGET_H = 360

# Parámetros de montaje de cámara (Orbbec en un robot tipo Stretch)
DEFAULT_CAMERA_HEIGHT = 1.1      # metros sobre el suelo
DEFAULT_CAMERA_TILT   = -0.6    # radianes (negativo = mirando hacia abajo)
DEFAULT_CAMERA_FWD    = 0.1     # metros hacia adelante desde el centro del robot

DEFAULT_OUTPUT_DIR = "dynamem_log/offline"
# ──────────────────────────────────────────────────────────────────────────────


def _ts_from_path(path: str) -> float:
    """Extrae timestamp en segundos del nombre de archivo YYYYMMDD_HHMMSS.ffffff."""
    fname = Path(path).stem  # quita extensión
    dt = datetime.datetime.strptime(fname, "%Y%m%d_%H%M%S.%f")
    return dt.timestamp()


def load_dataset(
    data_dir: str,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """Lee el CSV, empareja RGB/depth por timestamp más cercano y devuelve listas."""
    data_dir = Path(data_dir)

    # Buscar CSV
    csv_files = list(data_dir.glob("rgb_data_*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No se encontró rgb_data_*.csv en {data_dir}")
    csv_path = csv_files[0]
    print(f"Usando CSV: {csv_path.name}")

    with open(csv_path) as f:
        rows = list(csv.DictReader(f))

    rgb_rows   = [r for r in rows if "orbbec_rgb"   in r["image_path"]]
    depth_rows = [r for r in rows if "orbbec_depth" in r["image_path"]]

    print(f"  Frames RGB: {len(rgb_rows)}  |  Depth: {len(depth_rows)}")

    # Parsear timestamps de depth para emparejar con RGB
    dep_times = np.array([_ts_from_path(r["image_path"].split("/")[-1]) for r in depth_rows])

    rgbs, depths, poses = [], [], []

    for rgb_row in rgb_rows:
        # Emparejar con el depth frame más cercano en tiempo
        rgb_ts = _ts_from_path(rgb_row["image_path"].split("/")[-1])
        idx = int(np.argmin(np.abs(dep_times - rgb_ts)))
        dep_row = depth_rows[idx]

        rgb_path   = data_dir / rgb_row["image_path"]
        depth_path = data_dir / dep_row["image_path"]

        if not rgb_path.exists() or not depth_path.exists():
            continue

        # Cargar RGB
        img = np.array(Image.open(rgb_path).convert("RGB"), dtype=np.uint8)

        # Cargar depth: uint16 mm → float32 metros
        depth_mm = np.load(depth_path)["depth"].astype(np.float32)
        depth_m  = depth_mm / 1000.0

        # Pose del robot: (x, y, yaw)
        xyt = np.array([float(rgb_row["x"]), float(rgb_row["y"]), float(rgb_row["yaw"])],
                       dtype=np.float32)

        rgbs.append(img)
        depths.append(depth_m)
        poses.append(xyt)

    print(f"  Pares válidos cargados: {len(rgbs)}")
    return rgbs, depths, poses


def scale_intrinsics(fx, fy, cx, cy, src_w, src_h, dst_w, dst_h) -> np.ndarray:
    """Escala la matriz K al cambiar de resolución."""
    return np.array([
        [fx * dst_w / src_w,  0.,                 cx * dst_w / src_w],
        [0.,                  fy * dst_h / src_h, cy * dst_h / src_h],
        [0.,                  0.,                 1.                 ],
    ], dtype=np.float32)


def resize_frame(rgb: np.ndarray, depth: np.ndarray, w: int, h: int):
    """Redimensiona RGB y depth al tamaño objetivo."""
    rgb_out   = cv2.resize(rgb,   (w, h), interpolation=cv2.INTER_LINEAR)
    depth_out = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)
    return rgb_out, depth_out


def base_pose_to_cam_world(
    xyt: np.ndarray,
    camera_height: float,
    camera_tilt: float,
    camera_fwd: float,
) -> np.ndarray:
    """
    Convierte pose base 2D (x, y, theta) → matriz 4×4 cam-to-world.

    Marco mundo: ROS convention (X adelante, Y izquierda, Z arriba).
    Marco cámara: OpenCV convention (X derecha, Y abajo, Z adelante).
    """
    x, y, theta = xyt

    # Posición de la cámara en el mundo
    t = np.array([
        x + camera_fwd * np.cos(theta),
        y + camera_fwd * np.sin(theta),
        camera_height,
    ], dtype=np.float32)

    # Rotación: heading del robot (alrededor de Z)
    cy, sy = np.cos(theta), np.sin(theta)
    R_yaw = np.array([[cy, -sy, 0.],
                      [sy,  cy, 0.],
                      [0.,  0., 1.]], dtype=np.float32)

    # Cámara OpenCV → cuerpo ROS (cuando heading=0, tilt=0)
    #   X_cam (derecha)   → -Y_world (derecha del robot)
    #   Y_cam (abajo)     → -Z_world
    #   Z_cam (adelante)  → +X_world
    R_cam2body = np.array([[0., 0., 1.],
                           [-1., 0., 0.],
                           [0., -1., 0.]], dtype=np.float32)

    # Tilt de la cámara alrededor de su eje X
    ct, st = np.cos(camera_tilt), np.sin(camera_tilt)
    R_tilt = np.array([[1., 0.,  0.],
                       [0., ct, -st],
                       [0., st,  ct]], dtype=np.float32)

    R = R_yaw @ R_cam2body @ R_tilt

    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R
    T[:3, 3]  = t
    return T


def build_voxel_map(device: str, log_dir: str):
    """Instancia el mapa voxel semántico de DynaMem. log_dir es donde se vuelcan
    todos los archivos por-frame (rgb/depth/intrinsics/pose) y el pickle del mapa."""
    from stretch.mapping.voxel.voxel_dynamem import SparseVoxelMap
    from stretch.perception.detection.owl import OwlPerception
    from stretch.perception.detection.yoloe import YoloEPerception
    from stretch.perception.encoders.clip_encoder import MaskClipEncoder
    from stretch.perception.encoders.siglip_encoder import MaskSiglipEncoder

    if device == "cuda":
        print("GPU: SigLIP-so400m + OWLv2-L")
        encoder  = MaskSiglipEncoder(version="so400m",    feature_matching_threshold=0.14, device=device)
        detector = OwlPerception(version="owlv2-L-p14-ensemble", device=device, confidence_threshold=0.15)
        sem_res  = 0.05
    else:
        print("CPU: CLIP ViT-B/16 + YoloE-L")
        encoder  = MaskClipEncoder(version="ViT-B/16",  feature_matching_threshold=0.35, device=device)
        detector = YoloEPerception(confidence_threshold=0.05, device=device, size="l")
        sem_res  = 0.05

    # image_shape=None porque ya redimensionamos fuera
    voxel_map = SparseVoxelMap(
        resolution=0.1,
        semantic_memory_resolution=sem_res,
        local_radius=0.5,
        obs_min_height=0.2,
        obs_max_height=1.5,
        obs_min_density=5,
        grid_resolution=0.1,
        min_depth=0.5,
        max_depth=8.0,           # Orbbec llega a ~12 m; usamos 8 m
        pad_obstacles=2,
        add_local_radius_points=True,
        remove_visited_from_obstacles=False,
        smooth_kernel_size=3,
        use_median_filter=True,
        median_filter_size=4,
        use_derivative_filter=True,
        derivative_filter_threshold=0.2,
        detection=detector,
        encoder=encoder,
        image_shape=None,        # ya venimos en TARGET_W × TARGET_H
        log=log_dir,
        mllm=False,
    )
    return voxel_map


def ingest_frames(voxel_map, rgbs, depths, poses, K, camera_height, camera_tilt, camera_fwd):
    n = len(rgbs)
    for i, (rgb, depth, xyt) in enumerate(zip(rgbs, depths, poses)):
        cam_pose = base_pose_to_cam_world(xyt, camera_height, camera_tilt, camera_fwd)
        print(f"  [{i+1:4d}/{n}]  x={xyt[0]:.2f}  y={xyt[1]:.2f}  yaw={np.degrees(xyt[2]):.1f}°")
        voxel_map.process_rgbd_images(rgb, depth, K, cam_pose)
    print(f"\nFrames procesados: {voxel_map.obs_count}")


def _sanitize_filename(text: str) -> str:
    """Convierte el texto de una consulta en un nombre de archivo seguro."""
    name = re.sub(r"[^a-zA-Z0-9_-]+", "_", text.strip())
    return name.strip("_") or "query"


def _to_numpy(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _select_detection_bbox(
    scores: Optional[np.ndarray],
    boxes: Optional[np.ndarray],
    depth: np.ndarray,
    depth_threshold: float = 3.0,
) -> Optional[Tuple[Tuple[int, int, int, int], float]]:
    """
    Replica la lógica de selección de compute_obj_coord (ver
    stretch/perception/detection/{yoloe,owl}): recorre las detecciones ordenadas
    por score y devuelve la primera cuyo bbox tiene profundidad válida, que es
    la misma caja que se usó para calcular el punto 3D devuelto por DynaMem.
    """
    if scores is None or boxes is None or len(scores) == 0:
        return None
    height, width = depth.shape
    order = np.argsort(-scores)
    for idx in order:
        tl_x, tl_y, br_x, br_y = boxes[idx]
        tl_x = int(max(0, tl_x))
        tl_y = int(max(0, tl_y))
        br_x = int(min(width, br_x))
        br_y = int(min(height, br_y))
        if br_x <= tl_x or br_y <= tl_y:
            continue
        if depth[tl_y:br_y, tl_x:br_x].min() < depth_threshold:
            return (tl_x, tl_y, br_x, br_y), float(scores[idx])
    return None


def export_bbox_frame(voxel_map, text: str, obs_id: int, frames_dir: Path) -> Optional[str]:
    """
    Vuelve a correr el detector sobre el frame donde DynaMem localizó `text` y
    guarda en frames_dir un PNG con el bounding box dibujado (o un aviso si no
    hubo detección y la localización vino solo de similitud de features).
    """
    if voxel_map.detection_model is None:
        return None
    if obs_id is None or obs_id <= 0 or obs_id > len(voxel_map.observations):
        return None

    frame = voxel_map.observations[obs_id - 1]
    rgb = frame.rgb.numpy().astype(np.uint8)
    depth = _to_numpy(frame.depth)

    detect_kwargs = {}
    if "output_mask" in inspect.signature(voxel_map.detection_model.detect_object).parameters:
        detect_kwargs["output_mask"] = False
    scores, boxes = voxel_map.detection_model.detect_object(rgb=rgb, text=text, **detect_kwargs)
    scores = _to_numpy(scores) if scores is not None else None
    boxes = _to_numpy(boxes) if boxes is not None else None

    vis = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
    selection = _select_detection_bbox(scores, boxes, depth)
    if selection is not None:
        (tl_x, tl_y, br_x, br_y), score = selection
        cv2.rectangle(vis, (tl_x, tl_y), (br_x, br_y), (0, 255, 0), 2)
        label = f"{text} ({score:.2f})"
        cv2.putText(
            vis, label, (tl_x, max(0, tl_y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA,
        )
    else:
        cv2.putText(
            vis, f"'{text}': sin deteccion (similitud de features)", (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA,
        )

    png_path = frames_dir / f"bbox_{_sanitize_filename(text)}_frame{obs_id}.png"
    cv2.imwrite(str(png_path), vis)
    return str(png_path)


def _localize(voxel_map, text: str, frames_dir: Path):
    """Localiza el texto, exporta el PNG con bbox y devuelve (pos, obs_id, png_path)."""
    result = voxel_map.localize_with_feature_similarity(text, debug=True, return_debug=True)
    if result is None:
        return None
    target_point, debug_text, obs_id, _ = result
    if target_point is None:
        return None
    obs_id = int(obs_id)
    pos = np.round(
        target_point.numpy() if hasattr(target_point, "numpy") else np.array(target_point), 3
    ).tolist()
    png_path = export_bbox_frame(voxel_map, text, obs_id, frames_dir)
    return pos, obs_id, png_path


def query_objects(voxel_map, queries: List[str], output_dir: Path, frames_dir: Path) -> dict:
    """Localiza una lista de objetos, exporta sus PNGs con bbox y guarda resultados en JSON."""
    results = {}
    for text in queries:
        found = _localize(voxel_map, text, frames_dir)
        if found is None:
            print(f"  '{text}': no encontrado")
            results[text] = None
        else:
            pos, obs_id, png_path = found
            print(f"  '{text}': pos={pos}  frame={obs_id}  bbox={png_path}")
            results[text] = {"position": pos, "frame_id": obs_id, "bbox_image": png_path}
    json_path = output_dir / "query_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResultados guardados en: {json_path}")
    return results


def query_loop(voxel_map, output_dir: Path, frames_dir: Path):
    print("\n" + "=" * 60)
    print("Mapa listo. Escribe un objeto para localizarlo en el mapa.")
    print("Escribe 'salir' para terminar.")
    print("=" * 60)
    session = {}
    while True:
        try:
            text = input("\nConsulta> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text or text.lower() in ("salir", "quit", "exit", "q"):
            break
        found = _localize(voxel_map, text, frames_dir)
        if found is None:
            print(f"  No se encontró '{text}' en el mapa.")
            session[text] = None
        else:
            pos, obs_id, png_path = found
            print(f"  '{text}' encontrado:")
            print(f"    Posición 3D: {pos}")
            print(f"    Frame ID:    {obs_id}")
            print(f"    BBox PNG:    {png_path}")
            session[text] = {"position": pos, "frame_id": obs_id, "bbox_image": png_path}
    if session:
        json_path = output_dir / "query_results.json"
        with open(json_path, "w") as f:
            json.dump(session, f, indent=2, ensure_ascii=False)
        print(f"Resultados guardados en: {json_path}")


def main():
    parser = argparse.ArgumentParser(description="Offline DynaMem — dataset Orbbec, con export de bboxes")
    parser.add_argument("--data-dir",  default="../robot_20251024_074000",
                        help="Carpeta raíz del dataset")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                        help="Carpeta de salida. query_results.json queda aquí; todo lo demás "
                             "(rgb/depth/npy por frame, mapa.pkl, mapa_2d.png, PNGs de bbox) "
                             "va en <output-dir>/frames/")
    parser.add_argument("--fx", type=float, default=DEFAULT_FX)
    parser.add_argument("--fy", type=float, default=DEFAULT_FY)
    parser.add_argument("--cx", type=float, default=DEFAULT_CX)
    parser.add_argument("--cy", type=float, default=DEFAULT_CY)
    parser.add_argument("--camera-height", type=float, default=DEFAULT_CAMERA_HEIGHT)
    parser.add_argument("--camera-tilt",   type=float, default=DEFAULT_CAMERA_TILT)
    parser.add_argument("--camera-fwd",    type=float, default=DEFAULT_CAMERA_FWD)
    parser.add_argument("--cpu",           action="store_true")
    parser.add_argument("--max-frames",    type=int,   default=0,
                        help="Limitar número de frames (0 = todos)")
    parser.add_argument("--queries",       type=str,   default="",
                        help="Objetos a buscar, separados por coma. Ej: 'silla,taza'. "
                             "Si está vacío, abre el loop interactivo.")
    args = parser.parse_args()

    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    print(f"Device: {device}")

    # 0. Preparar carpetas de salida
    output_dir = Path(args.output_dir)
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    print(f"Salida:\n  consultas          -> {output_dir}/query_results.json")
    print(f"  frames/mapa/bboxes  -> {frames_dir}/")

    # 1. Cargar dataset
    rgbs_orig, depths_orig, poses = load_dataset(args.data_dir)

    if args.max_frames > 0:
        rgbs_orig    = rgbs_orig[:args.max_frames]
        depths_orig  = depths_orig[:args.max_frames]
        poses        = poses[:args.max_frames]

    # 2. Redimensionar a TARGET_W×TARGET_H y escalar intrínsecos
    src_h, src_w = rgbs_orig[0].shape[:2]   # 1080, 1920
    K = scale_intrinsics(args.fx, args.fy, args.cx, args.cy,
                         src_w, src_h, TARGET_W, TARGET_H)
    print(f"\nResolución: {src_w}×{src_h} → {TARGET_W}×{TARGET_H}")
    print(f"Intrínsecos escalados:\n{np.round(K, 2)}")

    rgbs   = []
    depths = []
    for rgb, depth in zip(rgbs_orig, depths_orig):
        r, d = resize_frame(rgb, depth, TARGET_W, TARGET_H)
        rgbs.append(r)
        depths.append(d)

    # 3. Construir mapa voxel (los archivos por-frame se guardan en frames_dir)
    print("\nInicializando DynaMem (los modelos se descargan la primera vez)...")
    voxel_map = build_voxel_map(device, log_dir=str(frames_dir))

    # 4. Procesar frames
    print(f"\nProcesando {len(rgbs)} frames...")
    ingest_frames(voxel_map, rgbs, depths, poses, K,
                  args.camera_height, args.camera_tilt, args.camera_fwd)

    # 5. Mapa 2D de ocupación — recortado al área observada
    obstacles, explored = voxel_map.get_2d_map()
    obs_np  = obstacles.cpu().numpy()
    exp_np  = explored.cpu().numpy()

    # Bounding box de las celdas exploradas
    rows = np.any(exp_np > 0, axis=1)
    cols = np.any(exp_np > 0, axis=0)
    if rows.any():
        r0, r1 = np.where(rows)[0][[0, -1]]
        c0, c1 = np.where(cols)[0][[0, -1]]
        pad = 5  # celdas de margen
        r0, r1 = max(0, r0 - pad), min(obs_np.shape[0], r1 + pad)
        c0, c1 = max(0, c0 - pad), min(obs_np.shape[1], c1 + pad)
        obs_np = obs_np[r0:r1, c0:c1]
        exp_np = exp_np[r0:r1, c0:c1]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(obs_np, origin="lower", cmap="Reds")
    axes[0].set_title("Obstáculos")
    axes[1].imshow(exp_np, origin="lower", cmap="Greens")
    axes[1].set_title("Área explorada")
    plt.tight_layout()
    map_path = frames_dir / "mapa_2d.png"
    plt.savefig(map_path)
    plt.close()
    print(f"\nMapa 2D guardado en: {map_path}")

    # Guardar mapa semántico completo
    pkl_path = frames_dir / "mapa.pkl"
    voxel_map.write_to_pickle(str(pkl_path))
    print(f"Mapa semántico guardado en: {pkl_path}")

    # 6. Consultas
    if args.queries:
        query_list = [q.strip() for q in args.queries.split(",") if q.strip()]
        print(f"\nConsultando {len(query_list)} objetos...")
        query_objects(voxel_map, query_list, output_dir, frames_dir)
    else:
        query_loop(voxel_map, output_dir, frames_dir)

    print("Fin.")


if __name__ == "__main__":
    src_path = str(Path(__file__).resolve().parents[1] / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    main()
