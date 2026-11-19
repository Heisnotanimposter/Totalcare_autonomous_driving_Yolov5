#!/usr/bin/env python3
"""Road vs Non-Road Semantic Segmentation & BEV Lane Detection / Masking for KITTI v4.

Combines:
1. Calibrated P2 Inverse Perspective Mapping (IPM) between image plane (u, v)
   and Ego-centric ground plane (X_m right, Y_m forward).
2. SegFormer / Semantic Road & Ego-Lane mask post-processing:
   - Binary Road vs Non-Road mask + continuous probability P(Road)
   - Ego-Lane drivable corridor polygon in both Image and Ego BEV space
   - Temporal EMA smoothing across video frames
3. Curvilinear Lane-Line Marking Detector constrained strictly inside the
   segmented Road mask (HLS L-channel + yellow chroma + directional Sobel-X),
   with sliding-window quadratic polynomial fitting and road-edge fallback.
4. Per-object context classification: `ego_lane`, `adjacent_road`, `off_road`.
"""

from __future__ import annotations

import base64
import math
import zlib
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


ROAD_BGR = (60, 195, 75)          # Emerald green for drivable road
NON_ROAD_BGR = (95, 55, 165)      # Muted plum/rose tint for non-road background
EGO_LANE_BGR = (210, 175, 45)     # Cyan-teal highlight for Ego driving lane
LANE_LINE_LEFT_BGR = (0, 235, 255)
LANE_LINE_RIGHT_BGR = (255, 210, 60)
LANE_LINE_ADJ_BGR = (180, 180, 210)


class P2RoadMapper:
    """Projects between image pixels (u, v) and Ego ground plane (X_m, Y_m) using P2."""

    def __init__(
        self,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        camera_height_m: float = 1.65,
    ):
        self.fx = float(fx)
        self.fy = float(fy)
        self.cx = float(cx)
        self.cy = float(cy)
        self.camera_height_m = float(camera_height_m)

    def ground_xy_to_uv(self, x_m: float, y_m: float) -> Optional[Tuple[int, int]]:
        if y_m <= 0.5:
            return None
        u = int(round(self.fx * (x_m / y_m) + self.cx))
        v = int(round(self.fy * (self.camera_height_m / y_m) + self.cy))
        return u, v

    def uv_to_ground_xy(self, u: float, v: float) -> Optional[Tuple[float, float]]:
        dv = float(v) - self.cy
        if dv <= 1.5:
            return None
        y_m = (self.fy * self.camera_height_m) / dv
        if y_m <= 0.5 or y_m > 80.0:
            return None
        x_m = ((float(u) - self.cx) * y_m) / self.fx
        return float(x_m), float(y_m)

    def warp_to_bev(
        self,
        img_or_mask: np.ndarray,
        bev_w: int = 240,
        bev_h: int = 360,
        x_range: Tuple[float, float] = (-12.0, 12.0),
        y_range: Tuple[float, float] = (4.0, 48.0),
        flags: int = cv2.INTER_LINEAR,
    ) -> np.ndarray:
        """Inverse-map image/mask into Ego ground BEV grid via calibrated P2 pinhole geometry."""
        fh, fw = img_or_mask.shape[:2]
        x_min, x_max = x_range
        y_min, y_max = y_range

        xs = np.linspace(x_min, x_max, bev_w, dtype=np.float32)
        ys = np.linspace(y_max, y_min, bev_h, dtype=np.float32)
        grid_x, grid_y = np.meshgrid(xs, ys)

        map_u = (self.fx * (grid_x / np.maximum(grid_y, 0.5)) + self.cx).astype(np.float32)
        map_v = (self.fy * (self.camera_height_m / np.maximum(grid_y, 0.5)) + self.cy).astype(np.float32)

        valid = (map_u >= 0.0) & (map_u <= fw - 1.0) & (map_v >= 0.0) & (map_v <= fh - 1.0)
        map_u[~valid] = -1.0
        map_v[~valid] = -1.0
        return cv2.remap(
            img_or_mask,
            map_u,
            map_v,
            interpolation=flags,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )


def parse_kitti_road_gt_mask(gt_bgr: np.ndarray, is_um_lane: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Decode KITTI data_road ground-truth PNG into binary (is_road, is_ego_lane) masks.

    In KITTI data_road GT (`training/gt_image_2/*.png`):
      - Background / Non-road: (R=255, G=0, B=0) in `*_road_*` and `um_lane_*` (BGR: B=0, G=0, R=255)
        or (0, 0, 0) for ignore/black borders.
      - Road / Ego-lane foreground: magenta (R=255, G=0, B=255) -> in BGR: (B=255, G=0, R=255).
    """
    b = gt_bgr[:, :, 0].astype(np.int16)
    g = gt_bgr[:, :, 1].astype(np.int16)
    r = gt_bgr[:, :, 2].astype(np.int16)

    magenta = (r > 200) & (b > 200) & (g < 80)
    if is_um_lane:
        ego_lane = magenta.astype(np.uint8)
        return ego_lane, ego_lane
    return magenta.astype(np.uint8), np.zeros_like(magenta, dtype=np.uint8)


class TemporalRoadLaneProcessor:
    """Turns per-frame Road/Ego-Lane probability maps + BGR image into BEV geometry and smooth lanes."""

    def __init__(self, mapper: P2RoadMapper, ema_alpha: float = 0.45):
        self.mapper = mapper
        self.ema_alpha = float(ema_alpha)
        self.prev_left_poly: Optional[np.ndarray] = None
        self.prev_right_poly: Optional[np.ndarray] = None

    def _fit_lane_lines_bev(
        self,
        bgr_frame: np.ndarray,
        road_mask_u8: np.ndarray,
        ego_mask_u8: np.ndarray,
        bev_w: int = 240,
        bev_h: int = 360,
        x_range: Tuple[float, float] = (-12.0, 12.0),
        y_range: Tuple[float, float] = (4.0, 46.0),
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], str]:
        """Detect left/right ego-lane polynomials (x_m as quadratic function of y_m)."""
        bev_img = self.mapper.warp_to_bev(bgr_frame, bev_w=bev_w, bev_h=bev_h, x_range=x_range, y_range=y_range)
        bev_road = self.mapper.warp_to_bev(road_mask_u8 * 255, bev_w=bev_w, bev_h=bev_h, x_range=x_range, y_range=y_range, flags=cv2.INTER_NEAREST) > 127
        bev_ego = self.mapper.warp_to_bev(ego_mask_u8 * 255, bev_w=bev_w, bev_h=bev_h, x_range=x_range, y_range=y_range, flags=cv2.INTER_NEAREST) > 127

        # Restrict strictly to interior of detected Road mask (erode 3px to avoid curb edges)
        kernel = np.ones((5, 5), np.uint8)
        road_interior = cv2.erode(bev_road.astype(np.uint8), kernel, iterations=1) > 0

        hls = cv2.cvtColor(bev_img, cv2.COLOR_BGR2HLS)
        l_ch = hls[:, :, 1].astype(np.float32)
        s_ch = hls[:, :, 2].astype(np.float32)
        gray = cv2.cvtColor(bev_img, cv2.COLOR_BGR2GRAY).astype(np.float32)

        # Horizontal intensity gradient (bright vertical line markings)
        sobel_x = np.abs(cv2.Sobel(l_ch, cv2.CV_32F, 1, 0, ksize=3))
        if np.max(sobel_x) > 1e-6:
            sobel_x = (sobel_x / np.max(sobel_x)) * 255.0

        # White markings (high lightness) or yellow markings (high saturation + warm hue)
        h_ch = hls[:, :, 0]
        yellow = (h_ch >= 12) & (h_ch <= 38) & (s_ch >= 70) & (l_ch >= 90)
        white = (l_ch >= 165) & ((l_ch - cv2.GaussianBlur(gray, (15, 3), 0)) > 14.0)
        grad_mask = (sobel_x >= 45.0) & (l_ch >= 115)
        marking_mask = road_interior & (white | yellow | grad_mask)

        x_min, x_max = x_range
        y_min, y_max = y_range
        xs_m = np.linspace(x_min, x_max, bev_w, dtype=np.float64)
        ys_m = np.linspace(y_max, y_min, bev_h, dtype=np.float64)

        def _fit_side(x_lo_m: float, x_hi_m: float, prev_poly: Optional[np.ndarray]) -> Tuple[Optional[np.ndarray], int]:
            col_mask = (xs_m >= x_lo_m) & (xs_m <= x_hi_m)
            sub = marking_mask[:, col_mask]
            # Focus on near-to-mid rows for histogram peak
            near_rows = sub[bev_h // 3 :, :]
            if int(near_rows.sum()) < 28:
                return None, 0
            hist = near_rows.sum(axis=0)
            peak_sub_col = int(np.argmax(hist))
            if hist[peak_sub_col] < 6:
                return None, 0
            col_indices = np.where(col_mask)[0]
            center_col = int(col_indices[peak_sub_col])

            # Sliding window search from bottom to top in BEV
            margin_px = max(8, int(round(0.75 / (x_max - x_min) * bev_w)))
            n_windows = 12
            win_h = bev_h // n_windows
            cur_col = center_col
            pts_y: List[float] = []
            pts_x: List[float] = []
            nz_y, nz_x = np.nonzero(marking_mask)
            for w_idx in range(n_windows):
                y_hi_px = bev_h - w_idx * win_h
                y_lo_px = max(0, y_hi_px - win_h)
                c_lo = max(0, cur_col - margin_px)
                c_hi = min(bev_w - 1, cur_col + margin_px)
                in_win = (nz_y >= y_lo_px) & (nz_y < y_hi_px) & (nz_x >= c_lo) & (nz_x <= c_hi)
                if int(np.sum(in_win)) >= 5:
                    win_xs = nz_x[in_win]
                    win_ys = nz_y[in_win]
                    cur_col = int(np.mean(win_xs))
                    pts_x.extend(xs_m[win_xs].tolist())
                    pts_y.extend(ys_m[win_ys].tolist())
            if len(pts_x) < 25:
                return None, len(pts_x)
            py_arr = np.asarray(pts_y, dtype=np.float64)
            px_arr = np.asarray(pts_x, dtype=np.float64)
            poly = np.polyfit(py_arr, px_arr, deg=2)
            # Sanity clamp curvature so single spurious shadow never explodes laterally
            if abs(poly[0]) > 0.0045:
                poly = np.array([0.0, float(np.polyfit(py_arr, px_arr, deg=1)[0]), float(np.polyfit(py_arr, px_arr, deg=1)[1])])
            if prev_poly is not None:
                poly = self.ema_alpha * poly + (1.0 - self.ema_alpha) * prev_poly
            return poly, len(pts_x)

        left_poly, l_cnt = _fit_side(-3.6, -0.4, self.prev_left_poly)
        right_poly, r_cnt = _fit_side(0.4, 3.6, self.prev_right_poly)
        source = "marking_fit"

        # Fallback to SegFormer Ego-Lane mask boundaries (or Road boundaries) on unmarked KITTI roads
        if left_poly is None or right_poly is None:
            source_mask = bev_ego if int(bev_ego.sum()) > 150 else road_interior
            ego_ys: List[float] = []
            ego_left_xs: List[float] = []
            ego_right_xs: List[float] = []
            for r_idx in range(bev_h - 1, int(bev_h * 0.2), -2):
                cols = np.where(source_mask[r_idx])[0]
                if len(cols) >= 6:
                    lx = float(xs_m[cols[0]])
                    rx = float(xs_m[cols[-1]])
                    # Clip overly wide full-road span into plausible ego-lane boundaries if unmarked
                    if rx - lx > 4.6:
                        mid = max(-1.2, min(1.2, 0.5 * (lx + rx))) if (lx < 0 < rx) else 0.0
                        lx = max(lx, mid - 1.75)
                        rx = min(rx, mid + 1.75)
                    if -4.5 <= lx <= -0.4 and 0.4 <= rx <= 4.5:
                        ego_ys.append(float(ys_m[r_idx]))
                        ego_left_xs.append(lx)
                        ego_right_xs.append(rx)
            if len(ego_ys) >= 12:
                ey = np.asarray(ego_ys, dtype=np.float64)
                if left_poly is None:
                    lp = np.polyfit(ey, np.asarray(ego_left_xs, dtype=np.float64), deg=2)
                    if abs(lp[0]) > 0.0045:
                        lp = np.array([0.0, float(np.polyfit(ey, ego_left_xs, 1)[0]), float(np.polyfit(ey, ego_left_xs, 1)[1])])
                    left_poly = (self.ema_alpha * lp + (1.0 - self.ema_alpha) * self.prev_left_poly) if self.prev_left_poly is not None else lp
                    source = "ego_mask_boundary" if l_cnt < 25 and r_cnt < 25 else "hybrid_fit"
                if right_poly is None:
                    rp = np.polyfit(ey, np.asarray(ego_right_xs, dtype=np.float64), deg=2)
                    if abs(rp[0]) > 0.0045:
                        rp = np.array([0.0, float(np.polyfit(ey, ego_right_xs, 1)[0]), float(np.polyfit(ey, ego_right_xs, 1)[1])])
                    right_poly = (self.ema_alpha * rp + (1.0 - self.ema_alpha) * self.prev_right_poly) if self.prev_right_poly is not None else rp
                    source = "ego_mask_boundary" if l_cnt < 25 and r_cnt < 25 else "hybrid_fit"

        if left_poly is not None:
            self.prev_left_poly = left_poly
        if right_poly is not None:
            self.prev_right_poly = right_poly
        return left_poly, right_poly, source

    def extract_frame_road_lane_record(
        self,
        frame_idx: int,
        bgr_frame: np.ndarray,
        road_prob_hw: np.ndarray,
        ego_prob_hw: np.ndarray,
        rle_downsample: int = 4,
    ) -> Dict:
        """Build compact per-frame Road/Non-Road + Ego-Lane + Lane-Lines metadata for visualization & World Model."""
        fh, fw = bgr_frame.shape[:2]
        road_bin = (road_prob_hw >= 0.50).astype(np.uint8)
        ego_bin = ((ego_prob_hw >= 0.45) & (road_bin > 0)).astype(np.uint8)

        # Clean small morphological islands
        k_close = np.ones((7, 7), np.uint8)
        road_bin = cv2.morphologyEx(road_bin, cv2.MORPH_CLOSE, k_close)
        road_bin = cv2.morphologyEx(road_bin, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        if int(ego_bin.sum()) > 100:
            ego_bin = cv2.morphologyEx(ego_bin, cv2.MORPH_CLOSE, k_close)

        left_poly, right_poly, lane_src = self._fit_lane_lines_bev(bgr_frame, road_bin, ego_bin)

        # Sample smooth BEV road free-space polygon from contours
        bev_w, bev_h = 180, 260
        x_rng = (-18.0, 18.0)
        y_rng = (2.0, 50.0)
        bev_road = self.mapper.warp_to_bev(
            road_bin * 255, bev_w=bev_w, bev_h=bev_h, x_range=x_rng, y_range=y_rng, flags=cv2.INTER_NEAREST
        )
        road_bev_polys: List[List[List[float]]] = []
        contours, _ = cv2.findContours(bev_road, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:3]:
            if cv2.contourArea(cnt) < 40:
                continue
            eps = 0.015 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, eps, True)
            poly_m: List[List[float]] = []
            for pt in approx.reshape(-1, 2):
                xm = x_rng[0] + (float(pt[0]) / max(1, bev_w - 1)) * (x_rng[1] - x_rng[0])
                ym = y_rng[1] - (float(pt[1]) / max(1, bev_h - 1)) * (y_rng[1] - y_rng[0])
                poly_m.append([round(xm, 2), round(ym, 2)])
            if len(poly_m) >= 3:
                road_bev_polys.append(poly_m)

        # Extract image-space polygons for Left Panel overlay (Road & Ego-Lane)
        def _mask_to_polys(mask_u8: np.ndarray, max_polys: int = 3) -> List[List[List[int]]]:
            out: List[List[List[int]]] = []
            cnts, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in sorted(cnts, key=cv2.contourArea, reverse=True)[:max_polys]:
                if cv2.contourArea(cnt) < 350:
                    continue
                eps = 0.008 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, eps, True)
                pts = approx.reshape(-1, 2).tolist()
                if len(pts) >= 3:
                    out.append([[int(p[0]), int(p[1])] for p in pts])
            return out

        road_px_polys = _mask_to_polys(road_bin, max_polys=3)
        ego_lane_px_polys = _mask_to_polys(ego_bin, max_polys=2)

        # Build BEV & Image polylines for left/right lane lines
        lane_lines_bev: Dict[str, List[List[float]]] = {}
        lane_lines_px: Dict[str, List[List[int]]] = {}
        eval_ys = [4.0, 6.0, 9.0, 12.0, 16.0, 21.0, 27.0, 34.0, 42.0]
        for side_name, poly in (("left_ego", left_poly), ("right_ego", right_poly)):
            if poly is None:
                continue
            pts_bev: List[List[float]] = []
            pts_px: List[List[int]] = []
            for ym in eval_ys:
                xm = float(np.polyval(poly, ym))
                pts_bev.append([round(xm, 2), round(ym, 2)])
                uv = self.mapper.ground_xy_to_uv(xm, ym)
                if uv is not None and 0 <= uv[0] < fw and 0 <= uv[1] < fh:
                    pts_px.append([int(uv[0]), int(uv[1])])
            if len(pts_bev) >= 2:
                lane_lines_bev[side_name] = pts_bev
            if len(pts_px) >= 2:
                lane_lines_px[side_name] = pts_px

        # Downsampled & zlib-compressed uint8 probability grid (compact JSON lookup for exact object road_prob checks)
        ds = max(2, int(rle_downsample))
        sw, sh = max(16, fw // ds), max(8, fh // ds)
        road_small = cv2.resize((np.clip(road_prob_hw, 0.0, 1.0) * 100.0).astype(np.uint8), (sw, sh), interpolation=cv2.INTER_AREA)
        grid_b64 = base64.b64encode(zlib.compress(road_small.tobytes(), level=6)).decode("ascii")

        # Road statistics (lower 60% ground region + full frame)
        ground_band = road_bin[int(fh * 0.42) :, :]
        road_area_ratio = float(np.mean(road_bin))
        ground_road_ratio = float(np.mean(ground_band)) if ground_band.size else road_area_ratio
        road_conf = float(np.mean(road_prob_hw[road_bin > 0])) if int(road_bin.sum()) > 0 else 0.0

        return {
            "frame_idx": int(frame_idx),
            "img_wh": [int(fw), int(fh)],
            "road_area_ratio": round(road_area_ratio, 4),
            "ground_road_ratio": round(ground_road_ratio, 4),
            "road_mean_confidence": round(road_conf, 4),
            "lane_source": lane_src,
            "left_poly_coeffs": [round(float(v), 6) for v in left_poly] if left_poly is not None else None,
            "right_poly_coeffs": [round(float(v), 6) for v in right_poly] if right_poly is not None else None,
            "road_px_polys": road_px_polys,
            "ego_lane_px_polys": ego_lane_px_polys,
            "road_bev_polys": road_bev_polys,
            "lane_lines_bev": lane_lines_bev,
            "lane_lines_px": lane_lines_px,
            "road_prob_grid_wh": [int(sw), int(sh)],
            "road_prob_grid_ds": int(ds),
            "road_prob_grid_b64": grid_b64,
        }


def _decode_road_prob_grid(road_record: Dict) -> Optional[np.ndarray]:
    sw, sh = road_record["road_prob_grid_wh"]
    if "_decoded_grid" in road_record:
        return road_record["_decoded_grid"]
    if "road_prob_grid_b64" in road_record:
        raw = zlib.decompress(base64.b64decode(road_record["road_prob_grid_b64"]))
        arr = np.frombuffer(raw, dtype=np.uint8).reshape((sh, sw))
        road_record["_decoded_grid"] = arr
        return arr
    if "road_prob_grid_q100" in road_record:
        arr = np.asarray(road_record["road_prob_grid_q100"], dtype=np.uint8).reshape((sh, sw))
        road_record["_decoded_grid"] = arr
        return arr
    return None


def query_object_road_lane_context(
    road_record: Optional[Dict],
    contact_u: float,
    contact_v: float,
    x_ego_m: float,
    y_ego_m: float,
    default_half_width_m: float = 1.65,
) -> Dict:
    """Classify an object's road/non-road and lane status using frame SegFormer + BEV lane polynomials."""
    if not road_record:
        return {
            "surface_label": "Unknown",
            "on_road": True,
            "road_prob": 0.5,
            "in_ego_lane": abs(x_ego_m) <= default_half_width_m,
            "lane_zone": "corridor_nominal",
        }

    # Query SegFormer Road probability at object ground-contact pixel
    sw, sh = road_record["road_prob_grid_wh"]
    ds = float(road_record["road_prob_grid_ds"])
    grid = _decode_road_prob_grid(road_record)
    if grid is None:
        road_prob = 0.5
    else:
        gu = max(0, min(sw - 1, int(round(contact_u / ds))))
        gv = max(0, min(sh - 1, int(round(contact_v / ds))))
        probs = []
        for dv in (-1, 0):
            rv = max(0, min(sh - 1, gv + dv))
            probs.append(float(grid[rv, gu]) / 100.0)
        road_prob = max(probs)
    on_road = road_prob >= 0.42

    # Check curved Ego-Lane bounds at y_ego_m using detected polynomials if available
    lp = road_record.get("left_poly_coeffs")
    rp = road_record.get("right_poly_coeffs")
    y_q = max(2.0, min(48.0, float(y_ego_m)))
    x_left = float(np.polyval(lp, y_q)) if lp is not None else -default_half_width_m
    x_right = float(np.polyval(rp, y_q)) if rp is not None else default_half_width_m
    if x_right - x_left < 2.2:
        mid = 0.5 * (x_left + x_right)
        x_left, x_right = mid - default_half_width_m, mid + default_half_width_m

    margin = 0.25
    if (x_left - margin) <= x_ego_m <= (x_right + margin) and y_ego_m > 0.0:
        in_ego_lane = True
        lane_zone = "ego_lane"
        on_road = True
    elif on_road:
        in_ego_lane = False
        lane_zone = "adjacent_left_road" if x_ego_m < x_left else "adjacent_right_road"
    else:
        in_ego_lane = False
        lane_zone = "non_road"

    return {
        "surface_label": "Road" if on_road else "Non-Road",
        "on_road": bool(on_road),
        "road_prob": round(float(road_prob), 3),
        "in_ego_lane": bool(in_ego_lane),
        "lane_zone": lane_zone,
        "ego_lane_bounds_x_m": [round(x_left, 2), round(x_right, 2)],
    }
