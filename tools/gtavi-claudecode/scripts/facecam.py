"""Find the streamer's webcam box in a video.

usage: python facecam.py <video> [--start S --end E] [--json out.json] [--debug out.jpg]
importable: find_cam(video, start=None, end=None) -> dict

Returns one of
  {"mode":"cam",  "box":[x,y,w,h] (fractions of the frame), "face":[x,y,w,h], "confidence":0..1, "reason":...}
  {"mode":"face", "box":[...], ...}   no clean rectangle, tight 16:9-ish crop around the face
  {"mode":"none", "reason":...}       no stable face: callers should fall back to the blur layout
Fractions plug straight into cut.py --cam x:y:w:h.

Method: YuNet face detection on ~12 sampled frames, keep the face cluster that stays at the same spot in most
frames (in-game faces move). Then grow a rectangle around that face using a median-of-frames edge map:
the webcam border is a straight, static line, so its edge response survives the median while gameplay edges blur out.
"""
import argparse, json, os, subprocess, sys, urllib.request
from pathlib import Path

import cv2
import numpy as np

MODEL_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
MODEL = Path(os.environ.get("LOCALAPPDATA", Path.home() / ".cache")) / "gtavi-claudecode" / "models" / "face_detection_yunet_2023mar.onnx"
N_FRAMES = 12
MAX_W = 1280  # analysis width


def model_path():
    if not MODEL.exists() or MODEL.stat().st_size < 100_000:
        MODEL.parent.mkdir(parents=True, exist_ok=True)
        tmp = MODEL.with_suffix(".part")
        urllib.request.urlretrieve(MODEL_URL, tmp)
        tmp.replace(MODEL)
    return str(MODEL)


def duration(video):
    cap = cv2.VideoCapture(str(video))
    n = cap.get(cv2.CAP_PROP_FRAME_COUNT); fps = cap.get(cv2.CAP_PROP_FPS) or 30
    cap.release()
    return n / fps if n > 0 else 0


def sample_frames(video, start, end, n=N_FRAMES):
    """Grab n frames evenly across [start,end] with cv2 (seek by time)."""
    dur = duration(video)
    s = max(0.0, start or 0.0)
    e = min(end if end is not None else dur, dur) if dur else (end or s + 20)
    if e <= s:
        e = s + 1
    cap = cv2.VideoCapture(str(video))
    frames = []
    for i in range(n):
        t = s + (e - s) * (i + 0.5) / n
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, f = cap.read()
        if ok and f is not None:
            h, w = f.shape[:2]
            if w > MAX_W:
                f = cv2.resize(f, (MAX_W, int(h * MAX_W / w)), interpolation=cv2.INTER_AREA)
            frames.append(f)
    cap.release()
    return frames


def detect_faces(frames):
    det = cv2.FaceDetectorYN.create(model_path(), "", (320, 320), 0.7, 0.3, 5000)
    out = []
    for f in frames:
        h, w = f.shape[:2]
        det.setInputSize((w, h))
        _, faces = det.detect(f)
        out.append([] if faces is None else [tuple(map(float, r[:4])) + (float(r[14]),) for r in faces])
    return out


def iou(a, b):
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw = max(0, min(ax2, bx2) - max(a[0], b[0])); ih = max(0, min(ay2, by2) - max(a[1], b[1]))
    inter = iw * ih
    return inter / (a[2] * a[3] + b[2] * b[3] - inter + 1e-9)


def stable_face(dets, n):
    """Cluster detections by position; return (median face box, hits) of the best stable cluster."""
    best = None
    flat = [(i, d) for i, ds in enumerate(dets) for d in ds]
    for _, seed in flat:
        members = {}
        for i, d in flat:
            # same webcam face: centre within ~1.3 face widths (streamers sway) and similar size
            dx = (seed[0] + seed[2] / 2) - (d[0] + d[2] / 2); dy = (seed[1] + seed[3] / 2) - (d[1] + d[3] / 2)
            if (dx * dx + dy * dy) ** 0.5 < 1.3 * max(seed[2], d[2]) and 0.6 < d[2] / seed[2] < 1.67:
                if i not in members or d[4] > members[i][4]:
                    members[i] = d
        if best is None or len(members) > len(best):
            best = members
    if not best:
        return None, 0
    arr = np.array(list(best.values()))
    return np.median(arr[:, :4], axis=0), len(best)


def edge_map(frames):
    """Median over frames of gradient magnitude per frame: keeps only static edges."""
    gs = []
    for f in frames:
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        g = cv2.GaussianBlur(g, (3, 3), 0)
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3); gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
        gs.append((gx, gy))
    mx = np.median(np.abs([a for a, _ in gs]), axis=0)
    my = np.median(np.abs([b for _, b in gs]), axis=0)
    # edges are 2-3 px thick after scaling: take the max over 1 px neighbours across the edge direction
    mx = np.maximum(mx, np.maximum(np.roll(mx, 1, 1), np.roll(mx, -1, 1)))
    my = np.maximum(my, np.maximum(np.roll(my, 1, 0), np.roll(my, -1, 0)))
    return mx, my


def find_rect(frames, face):
    """Expand from the face outward to the strongest static straight edges on each side.
    Returns (x0,y0,x1,y1, score) or None. Static edges: vertical lines (|gx|) for left/right, horizontal (|gy|) for top/bottom."""
    h, w = frames[0].shape[:2]
    gx, gy = edge_map(frames)
    fx, fy, fw, fh = face
    cx, cy = fx + fw / 2, fy + fh / 2
    # also demand the line to be temporally static in intensity (low variance across frames) along its length
    gray = np.array([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]).astype(np.float32)
    # the cam interior (face) is also dynamic, so static means: the edge exists in the median, with a long run
    thr_v = max(20.0, np.percentile(gx, 99.0) * 0.35)
    thr_h = max(20.0, np.percentile(gy, 99.0) * 0.35)

    def run_len(line_mask):
        # longest continuous run (allowing small gaps) in a 1-D boolean array
        best = cur = gap = 0
        for v in line_mask:
            if v:
                cur += 1 + gap; gap = 0; best = max(best, cur)
            elif cur and gap < 4:
                gap += 1
            else:
                cur = gap = 0
        return best

    # Candidate extents: face-relative search windows
    def search(axis_edges, lo, hi, other_lo, other_hi, thr, vertical):
        """Best static edge position in [lo,hi) judged over the span [other_lo,other_hi)."""
        cands = []
        for p in range(max(lo, 1), min(hi, (w if vertical else h) - 1)):
            sl = axis_edges[other_lo:other_hi, p] if vertical else axis_edges[p, other_lo:other_hi]
            m = sl > thr
            if m.mean() < 0.4:
                continue
            frac = run_len(m) / max(1, other_hi - other_lo)
            if frac < 0.4:  # a real border is one long straight line, not text or furniture
                continue
            cands.append((p, frac * float(sl.mean())))
        if not cands:
            return None, 0
        top = max(sc for _, sc in cands)
        # near-best lines: take the outermost (the webcam frame is outside inner furniture lines)
        strong = [(p, sc) for p, sc in cands if sc >= 0.9 * top]
        ref = cx if vertical else cy
        after = lo > ref
        return max(strong, key=lambda t: t[0] if after else -t[0])

    # span used to judge vertical lines: the face's height band extended; horizontal lines: face's width band
    ys0, ys1 = int(max(0, cy - fh * 2.5)), int(min(h, cy + fh * 2.5))
    xs0, xs1 = int(max(0, cx - fw * 4)), int(min(w, cx + fw * 4))
    # sides limited to sane webcam size: box between 1.6x and 8x face width
    def sides(ys0, ys1, xs0, xs1):
        return (search(gx, int(cx - fw * 10), int(fx - fw * 0.1), ys0, ys1, thr_v, True),
                search(gx, int(fx + fw * 1.1), int(cx + fw * 10), ys0, ys1, thr_v, True),
                search(gy, int(cy - fh * 4.5), int(fy - fh * 0.1), xs0, xs1, thr_h, False),
                search(gy, int(fy + fh * 1.1), int(cy + fh * 4.5), xs0, xs1, thr_h, False))
    (L, sL), (R, sR), (T, sT), (B, sB) = sides(ys0, ys1, xs0, xs1)
    # refine: judge each line over the full extent of the box found so far (overlays and room furniture do not span it)
    for _ in range(2):
        if None in (L, R, T, B):
            break
        (L2, sL), (R2, sR), (T2, sT), (B2, sB) = sides(T, B, L, R)
        L, R, T, B = (L2 if L2 is not None else L), (R2 if R2 is not None else R), (T2 if T2 is not None else T), (B2 if B2 is not None else B)
    # frame borders count as edges (cam docked to a corner)
    notes = []
    if L is None and fx - fw * 10 < 3: L = 0; sL = 1; notes.append("L=frame")
    if R is None and cx + fw * 10 > w - 3: R = w; sR = 1; notes.append("R=frame")
    if T is None and cy - fh * 4.5 < 3: T = 0; sT = 1; notes.append("T=frame")
    if B is None and cy + fh * 4.5 > h - 3: B = h; sB = 1; notes.append("B=frame")
    found = [x is not None for x in (L, R, T, B)]
    weak = [n for n, s, t in (("L", sL, thr_v), ("R", sR, thr_v), ("T", sT, thr_h), ("B", sB, thr_h)) if s < 0.9 * t]
    return dict(L=L, R=R, T=T, B=B, found=found, notes=notes, weak=len(weak))


def refine_with_frame_edges(rect, w, h, face):
    """If cam is docked to a frame edge, snap sides that are within a few px or close to the frame edge."""
    return rect


SNAP_FRAMES = 20
SNAP_RANGE = 80      # px (at MAX_W analysis width) searched around each edge
SNAP_SHRINK = 2      # px moved inside so no sliver of game/background remains


def _edge_profile(M, S, p_lo, p_hi, span, vertical):
    """Score candidate border positions p in [p_lo,p_hi) of a vertical (x) or horizontal (y) edge.
    M = median colour image (float32 HxWx3), S = temporal std map (HxW), span = (a,b) along the line.
    p = first pixel after the change: step between [p-3,p) and [p,p+3).
    Evidence: (1) straight line: the colour step holds on most pixels along the span (static edge / constant colour line),
    (2) temporal change: cam region and game region differ in temporal behaviour across the line."""
    a, b = span
    if not vertical:
        M = M.transpose(1, 0, 2); S = S.T
    Ms = M[a:b]; Ss = S[a:b]
    n = Ms.shape[1]
    ps = np.arange(max(p_lo, 4), min(p_hi, n - 3))
    if len(ps) == 0:
        return ps, np.zeros(0)
    cs = np.concatenate([np.zeros((Ms.shape[0], 1, 3), np.float32), np.cumsum(Ms, axis=1)], axis=1)
    right = (cs[:, ps + 3] - cs[:, ps]) / 3; left = (cs[:, ps] - cs[:, ps - 3]) / 3
    step = np.abs(right - left).sum(axis=2)            # rows x candidates
    cons = (step > 24).mean(axis=0)                    # fraction of the line that shows the step
    mag = np.clip(np.median(step, axis=0) / 45.0, 0, 1)
    cs2 = np.concatenate([np.zeros((Ss.shape[0], 1), np.float32), np.cumsum(Ss, axis=1)], axis=1)
    k0 = np.clip(ps - 10, 0, n); k1 = np.clip(ps - 2, 0, n); k2 = np.clip(ps + 2, 0, n); k3 = np.clip(ps + 10, 0, n)
    sl = (cs2[:, k1] - cs2[:, k0]) / np.maximum(1, k1 - k0); sr = (cs2[:, k3] - cs2[:, k2]) / np.maximum(1, k3 - k2)
    rel = np.abs(sl - sr) / (sl + sr + 2.0)
    tcons = (rel > 0.35).mean(axis=0)
    tmag = np.clip(np.median(rel, axis=0) * 2, 0, 1)
    return ps, cons * mag + 0.6 * tcons * tmag


def snap_edges(frames, box, search=SNAP_RANGE, shrink=SNAP_SHRINK, face=None):
    """Snap [L,T,R,B] (px of the analysis frames) to the real webcam border. Returns (L,T,R,B,raw_edges)."""
    h, w = frames[0].shape[:2]
    L, T, R, B = [int(round(v)) for v in box]
    idx = np.linspace(0, len(frames) - 1, min(len(frames), SNAP_FRAMES)).astype(int)
    fs = [frames[i] for i in idx]
    M = np.median(np.array(fs), axis=0).astype(np.float32)
    S = np.array([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in fs]).astype(np.float32).std(axis=0)
    M = cv2.GaussianBlur(M, (3, 3), 0); S = cv2.GaussianBlur(S, (3, 3), 0)
    H, W = B - T, R - L
    yspan = (int(T + 0.15 * H), int(B - 0.15 * H)); xspan = (int(L + 0.15 * W), int(R - 0.15 * W))
    out = {}
    for name, p0, vertical, span in (("L", L, True, yspan), ("R", R, True, yspan), ("T", T, False, xspan), ("B", B, False, xspan)):
        lim = w if vertical else h
        if p0 <= 3 or p0 >= lim - 3:      # docked to the frame edge: nothing to snap
            out[name] = 0 if p0 <= 3 else lim
            continue
        ps, sc = _edge_profile(M, S, p0 - search, p0 + search + 1, span, vertical)
        if len(ps) == 0 or sc.max() < 0.25:
            out[name] = p0
            continue
        sc = sc - 0.05 * np.abs(ps - p0) / search            # weak prior: stay near the first estimate
        # strong plateaus (runs within 90% of the best); the cam border is the one nearest the face: a stronger line
        # further out (game UI card, window edge) lies beyond a flat dead strip that is not part of the cam
        strong = sc >= 0.9 * sc.max()
        if face is not None:
            fx, fy, fw_, fh_ = face
            lim_in = {"L": fx - 0.3 * fw_, "R": fx + 1.3 * fw_, "T": fy - 0.3 * fh_, "B": fy + 1.3 * fh_}[name]
            ok = (ps < lim_in) if name in ("L", "T") else (ps > lim_in)
            if (strong & ok).any():
                strong &= ok
        idx = np.flatnonzero(strong)
        runs = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
        run = runs[-1] if name in ("L", "T") else runs[0]
        # the step window is 3+3 px wide, so the true border is the centre of the plateau
        out[name] = int(round((ps[run[0]] + ps[run[-1]]) / 2))
    nL = out["L"] if out["L"] <= 3 else out["L"] + shrink
    nT = out["T"] if out["T"] <= 3 else out["T"] + shrink
    nR = out["R"] if out["R"] >= w - 3 else out["R"] - shrink
    nB = out["B"] if out["B"] >= h - 3 else out["B"] - shrink
    return nL, nT, nR, nB, out


FACE_MAX_DRIFT = 0.03   # of frame width per second (face centre), same relative rate for face size


def _median_filter(a, k=5):
    pad = k // 2
    p = np.pad(a, pad, mode="edge")
    return np.array([np.median(p[i:i + k]) for i in range(len(a))])


def track_face(video, start, end, face, fps=2.0):
    """Face centre and size over time -> [{t, cx, cy, size}] (fractions of the frame; size = face height / frame height;
    t = seconds from `start`). Detections near the stable face are kept, gaps are interpolated, then median + exponential
    smoothing + a slew limit (max drift FACE_MAX_DRIFT of frame width per second) so the crop never jumps."""
    dur = duration(video)
    s = max(0.0, start or 0.0)
    e = min(end if end is not None else dur, dur) if dur else (end or s + 20)
    if e <= s:
        e = s + 1
    n = int(min(240, max(2, round((e - s) * fps) + 1)))
    ts = np.linspace(s, e - min(0.05, (e - s) / 10), n)
    cap = cv2.VideoCapture(str(video))
    det = cv2.FaceDetectorYN.create(model_path(), "", (320, 320), 0.7, 0.3, 5000)
    fx, fy, fw, fh = face
    raw = []
    W = H = None
    ref = np.array([fx + fw / 2, fy + fh / 2, fh], float)   # px of the analysis frame
    for t in ts:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, f = cap.read()
        if not ok or f is None:
            raw.append(None); continue
        h, w = f.shape[:2]
        sc = min(1.0, MAX_W / w)
        if sc < 1:
            f = cv2.resize(f, (int(w * sc), int(h * sc)), interpolation=cv2.INTER_AREA); h, w = f.shape[:2]
        W, H = w, h
        det.setInputSize((w, h))
        _, fs = det.detect(f)
        best = None
        for r in ([] if fs is None else fs):
            cx, cy, hh = r[0] + r[2] / 2, r[1] + r[3] / 2, r[3]
            if np.hypot(cx - ref[0], cy - ref[1]) < 1.6 * ref[2] and 0.6 < hh / ref[2] < 1.67:
                d = np.hypot(cx - ref[0], cy - ref[1])
                if best is None or d < best[0]:
                    best = (d, cx, cy, hh)
        if best:
            raw.append(best[1:]); ref = 0.6 * ref + 0.4 * np.array(best[1:])
        else:
            raw.append(None)
    cap.release()
    if not W or sum(r is not None for r in raw) < 2:
        return []
    ok_i = [i for i, r in enumerate(raw) if r is not None]
    arr = np.array([raw[i] for i in ok_i], float)
    rel_t = ts - s
    cx = np.interp(np.arange(n), ok_i, arr[:, 0]); cy = np.interp(np.arange(n), ok_i, arr[:, 1]); sz = np.interp(np.arange(n), ok_i, arr[:, 2])
    out = []
    for v in (cx, cy, sz):
        v[:] = _median_filter(v, 5)
        for i in range(1, n):                                  # exponential smoothing
            v[i] = 0.25 * v[i] + 0.75 * v[i - 1]
    lim = FACE_MAX_DRIFT * W
    for i in range(1, n):                                      # slew limit: drift per second, in px
        dt = max(1e-3, rel_t[i] - rel_t[i - 1])
        for v, L_ in ((cx, lim), (cy, lim), (sz, 0.06 * max(sz[i - 1], 1))):
            d = v[i] - v[i - 1]; m = L_ * dt
            v[i] = v[i - 1] + max(-m, min(m, d))
    return [{"t": round(float(rel_t[i]), 3), "cx": round(float(cx[i] / W), 4), "cy": round(float(cy[i] / H), 4),
             "size": round(float(sz[i] / H), 4)} for i in range(n)]


def _pw(ts, vs):
    """Piecewise-linear ffmpeg expression in t (commas escaped for a filtergraph)."""
    if len(ts) == 1:
        return f"{vs[0]:.2f}"
    expr = f"{vs[-1]:.2f}"
    for i in range(len(ts) - 2, -1, -1):
        dt = max(1e-3, ts[i + 1] - ts[i])
        expr = f"if(lt(t\\,{ts[i + 1]:.3f})\\,{vs[i]:.2f}+({vs[i + 1] - vs[i]:.2f})*(t-{ts[i]:.3f})/{dt:.3f}\\,{expr})"
    return expr


def face_crop_expr(track, src_w, src_h, out_w, out_h, eyes_at=0.40, head_fill=0.38, bounds=None):
    """ffmpeg crop for a face panel of aspect out_w:out_h that follows the smoothed face.
    head_fill = face box height / crop height (0.38 shows head and shoulders). Eyes sit at eyes_at of the panel height,
    the face is horizontally centred, the face box never leaves the crop and the crop never leaves the frame.
    Returns {"w","h" (even ints, fixed: ffmpeg cannot animate crop size), "x","y" (expressions in t, commas escaped as \\,
    so they drop straight into a filtergraph), "filter": "crop=w:h:x:y"}. track t is seconds from the cut start (use after -ss)."""
    if not track:
        return None
    ar = out_w / out_h
    L, T, R, B = bounds if bounds else (0, 0, src_w, src_h)                 # crop must stay inside the webcam
    bw, bh = R - L, B - T
    size = float(np.median([k["size"] for k in track])) * src_h          # face height px
    ch = min(size / head_fill, bh, bw / ar)                               # too big for the cam: zoom in instead
    cw = ch * ar
    cw = max(2, int(cw) // 2 * 2); ch = max(2, int(round(cw / ar)) // 2 * 2)
    if ch > src_h:
        ch = src_h // 2 * 2; cw = max(2, int(ch * ar) // 2 * 2)
    xs, ys, ts = [], [], []
    for k in track:
        fcx, fcy = k["cx"] * src_w, k["cy"] * src_h
        fh_ = k["size"] * src_h; fw_ = 0.8 * fh_
        x = fcx - cw / 2
        y = (fcy - 0.12 * fh_) - eyes_at * ch                              # eyes sit ~0.12 face heights above the box centre
        lo, hi = max(L, fcx + fw_ / 2 - cw), min(R - cw, fcx - fw_ / 2)  # keep the whole face box inside
        if lo <= hi: x = min(max(x, lo), hi)
        lo, hi = max(T, fcy + fh_ / 2 - ch), min(B - ch, fcy - fh_ / 2)
        if lo <= hi: y = min(max(y, lo), hi)
        xs.append(min(max(x, L), R - cw)); ys.append(min(max(y, T), B - ch)); ts.append(k["t"])
    x_e, y_e = _pw(ts, xs), _pw(ts, ys)
    return {"w": cw, "h": ch, "x": x_e, "y": y_e, "filter": f"crop={cw}:{ch}:{x_e}:{y_e}"}


def has_gameplay(video, start, end, cam_box, info=False):
    """True when the area outside the cam box is game footage that differs from the cam scene.
    cam_box = [x,y,w,h] fractions. None, or a box covering most of the frame, means a full-screen camera -> False.
    A face-mode box is a crop of the same room, so its outside matches it (similar colour histograms) -> False."""
    def ret(v, d):
        return (v, d) if info else v
    if not cam_box:
        return ret(False, {"why": "no cam box"})
    frames = sample_frames(video, start, end, 8)
    if len(frames) < 3:
        return ret(False, {"why": "too few frames"})
    h, w = frames[0].shape[:2]
    x, y, bw, bh = cam_box
    if bw * bh > 0.55:
        return ret(False, {"why": "cam covers the frame"})
    x0, y0, x1, y1 = int(x * w), int(y * h), int((x + bw) * w), int((y + bh) * h)
    inside = np.zeros((h, w), np.uint8); inside[y0:y1, x0:x1] = 255
    outside = 255 - inside
    # ignore a thin margin so the cam border does not leak into the outside histogram
    outside = cv2.erode(outside, np.ones((9, 9), np.uint8))
    hist = lambda m, hsv: cv2.normalize(cv2.calcHist([hsv], [0, 1, 2], m, [12, 4, 6], [0, 180, 0, 256, 0, 256]), None, 1, 0, cv2.NORM_L1).flatten()
    sims = []
    for f in frames:
        hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
        sims.append(cv2.compareHist(hist(inside, hsv), hist(outside, hsv), cv2.HISTCMP_BHATTACHARYYA))
    dist = float(np.median(sims))                      # 0 = same scene, 1 = nothing in common
    gray = np.array([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]).astype(np.float32)
    om = outside > 0
    content = float(gray.mean(axis=0)[om].std())                          # spatial variety outside
    motion = float(np.abs(np.diff(gray, axis=0))[:, om].mean())           # frame-to-frame change outside
    # the picture just outside the cam is a continuation of it for IRL footage: a plain-colour or flat outside is not gameplay
    # an overlay has a hard straight border all around; a crop of one IRL scene is continuous across the box edge
    Mg = cv2.GaussianBlur(np.median(np.array(frames), axis=0).astype(np.float32), (3, 3), 0)
    fr_ = []
    for p, vert, (a, b) in ((x0, 1, (y0, y1)), (x1, 1, (y0, y1)), (y0, 0, (x0, x1)), (y1, 0, (x0, x1))):
        if p <= 4 or p >= (w if vert else h) - 4:
            continue                                   # docked to the frame edge: no border there
        m = int(0.1 * (b - a)); a, b = a + m, b - m
        best = 0.0
        for q in range(p - 3, p + 4):
            if vert: s = np.abs(Mg[a:b, q + 1:q + 4].mean(1) - Mg[a:b, q - 3:q].mean(1)).sum(1)
            else: s = np.abs(Mg[q + 1:q + 4, a:b].mean(0) - Mg[q - 3:q, a:b].mean(0)).sum(1)
            best = max(best, float((s > 30).mean()))
        fr_.append(best)
    border = float(np.median(fr_)) if fr_ else 0.0
    has_content = content > 12 or motion > 3
    res = has_content and ((border >= 0.65 and dist > 0.38) or dist > 0.6)
    return ret(bool(res), {"hist_dist": round(dist, 3), "border": round(border, 2), "content": round(content, 1), "motion": round(motion, 2)})


def where(box):
    x, y, bw, bh = box
    cx, cy = x + bw / 2, y + bh / 2
    v = "top" if cy < 0.4 else "bottom" if cy > 0.6 else "middle"
    hz = "left" if cx < 0.4 else "right" if cx > 0.6 else "center"
    return f"{v}-{hz} {round(bw * 100)}% x {round(bh * 100)}%"


def find_cam(video, start=None, end=None):
    frames = sample_frames(video, start, end)
    if len(frames) < 4:
        return {"mode": "none", "confidence": 0.0, "reason": f"could only read {len(frames)} frames"}
    h, w = frames[0].shape[:2]
    dets = detect_faces(frames)
    face, hits = stable_face(dets, len(frames))
    need = max(3, int(len(frames) * 0.5))
    total = sum(len(d) for d in dets)
    if face is None or hits < need:
        return {"mode": "none", "confidence": 0.0, "faces_seen": total,
                "reason": f"no face stable across frames (best cluster {hits}/{len(frames)}, need {need})"}
    fx, fy, fw, fh = face
    fracface = [round(fx / w, 4), round(fy / h, 4), round(fw / w, 4), round(fh / h, 4)]
    if fw / w > 0.5:
        try:
            track = track_face(video, start, end, face)
        except Exception:
            track = []
        return {"mode": "none", "confidence": 0.0, "face": fracface, "face_track": track,
                "reason": "stable face is huge (full-frame talking head, not a webcam overlay)"}
    stab = hits / len(frames)

    r = find_rect(frames, face)
    L, R, T, B = r["L"], r["R"], r["T"], r["B"]
    n_found = sum(r["found"])
    # snap sides near frame borders
    m = 0.01 * w
    if L is not None and L < m: L = 0
    if R is not None and R > w - m: R = w
    if T is not None and T < 0.01 * h: T = 0
    if B is not None and B > h - 0.01 * h: B = h
    ok = L is not None and R is not None and T is not None and B is not None
    if ok:
        bw, bh = R - L, B - T
        # sane webcam: smaller than 60% of frame width/height, contains the face
        contains = L <= fx and R >= fx + fw and T <= fy and B >= fy + fh
        sane = 0.08 < bw / w < 0.6 and 0.08 < bh / h < 0.7 and contains and bw > fw * 1.2 and bh > fh * 1.2
        if sane:
            try:
                sf = sample_frames(video, start, end, SNAP_FRAMES)
                nL, nT, nR, nB, raw = snap_edges(sf if len(sf) >= 8 else frames, (L, T, R, B), face=(fx, fy, fw, fh))
                if nR - nL > fw * 1.2 and nB - nT > fh * 1.2 and nL <= fx and nR >= fx + fw and nT <= fy and nB >= fy + fh:
                    L, T, R, B = nL, nT, nR, nB; bw, bh = R - L, B - T
                    r["notes"] = list(r["notes"]) + [f"snapped {raw}"]
            except Exception as ex:  # snapping is a refinement: never lose the cam over it
                r["notes"] = list(r["notes"]) + [f"snap failed: {ex}"]
            box =[round(L / w, 4), round(T / h, 4), round(bw / w, 4), round(bh / h, 4)]
            conf = round(min(1.0, 0.55 + 0.25 * stab + 0.2 * (1 if n_found == 4 else 0.5)), 2)
            return {"mode": "cam", "box": box, "face": fracface, "confidence": conf,
                    "reason": f"face stable in {hits}/{len(frames)} frames, rectangle edges found on {n_found}/4 sides {r['notes']}",
                    "summary": f"cam: {where(box)}"}
    # fallback: tight 16:9 crop around the face (about 3x face width)
    cw = min(w, fw * 3.2); ch = cw * 9 / 16
    if ch < fh * 1.8:
        ch = fh * 1.8; cw = ch * 16 / 9
    ccx, ccy = fx + fw / 2, fy + fh * 0.55
    x0 = min(max(0, ccx - cw / 2), w - cw); y0 = min(max(0, ccy - ch / 2), h - ch)
    box = [round(x0 / w, 4), round(y0 / h, 4), round(cw / w, 4), round(ch / h, 4)]
    conf = round(min(0.8, 0.35 + 0.4 * stab), 2)
    try:
        track = track_face(video, start, end, face)
    except Exception:
        track = []
    return {"mode": "face", "box": box, "face": fracface, "confidence": conf, "face_track": track,
            "reason": f"face stable in {hits}/{len(frames)} frames but no clean rectangle ({n_found}/4 sides found); face-centred crop",
            "summary": f"face (no cam border): {where(box)}"}


def draw_debug(video, res, out, start=None, end=None):
    frames = sample_frames(video, start, end, 3)
    if not frames:
        return
    f = frames[len(frames) // 2].copy()
    h, w = f.shape[:2]
    if res.get("box"):
        x, y, bw, bh = res["box"]
        col = (0, 255, 0) if res["mode"] == "cam" else (0, 200, 255)
        cv2.rectangle(f, (int(x * w), int(y * h)), (int((x + bw) * w), int((y + bh) * h)), col, 4)
    if res.get("face"):
        x, y, bw, bh = res["face"]
        cv2.rectangle(f, (int(x * w), int(y * h)), (int((x + bw) * w), int((y + bh) * h)), (255, 0, 0), 2)
    cv2.putText(f, f"{res['mode']} conf={res.get('confidence')}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)
    cv2.imwrite(str(out), f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video"); ap.add_argument("--start", type=float); ap.add_argument("--end", type=float)
    ap.add_argument("--json"); ap.add_argument("--debug", help="write a JPG with the box drawn")
    a = ap.parse_args()
    res = find_cam(a.video, a.start, a.end)
    if res["mode"] == "none":
        line = f"no webcam found ({res['reason']}) -> use the blur layout"
    else:
        line = f"{res['summary']} (confidence {res['confidence']}; {res['reason']}) -> --cam {':'.join(f'{v:.3f}' for v in res['box'])}"
    print(line)
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1), encoding="utf-8")
    if a.debug:
        draw_debug(a.video, res, a.debug, a.start, a.end)


if __name__ == "__main__":
    main()




def inner_edges(video, start, end, face, src_w, src_h, n=12):
    """For a webcam with no full frame (e.g. docked to the corner of the screen), find the inner borders
    that do exist: the strongest straight line on each side of the face, held over sampled frames.
    face = [x, y, w, h] fractions. Returns (L, T, R, B) in pixels; a side with no clear line is the frame edge."""
    cap = cv2.VideoCapture(str(video)); g = []
    for k in range(n):
        cap.set(cv2.CAP_PROP_POS_MSEC, ((start or 0) + ((end or 0) - (start or 0)) * (k + 0.5) / n) * 1000)
        ok, f = cap.read()
        if ok:
            g.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32))
    cap.release()
    if len(g) < 4:
        return (0, 0, src_w, src_h)
    med = np.median(np.stack(g), 0)
    H_, W_ = med.shape
    fx, fy, fw, fh = face[0] * W_, face[1] * H_, face[2] * W_, face[3] * H_
    dy = np.abs(cv2.Sobel(med, cv2.CV_32F, 0, 1, ksize=3)); dx = np.abs(cv2.Sobel(med, cv2.CV_32F, 1, 0, ksize=3))
    # temporal: game and cam change differently, so the border is also where std jumps
    sd = np.stack(g).std(0)
    def best_line(score, lo, hi):
        lo, hi = int(max(0, lo)), int(min(len(score) - 1, hi))
        if hi - lo < 3:
            return None
        seg = score[lo:hi]; i = int(np.argmax(seg)); m = np.median(score) + 1e-6
        return lo + i if seg[i] > 4 * m and seg[i] > 12 else None
    x0, x1 = int(max(0, fx - 2.5 * fw)), int(min(W_, fx + fw + 2.5 * fw))
    y0, y1 = int(max(0, fy - 2.0 * fh)), int(min(H_, fy + fh + 3.0 * fh))
    rows = dy[:, x0:x1].mean(1) + np.abs(np.diff(sd[:, x0:x1].mean(1), prepend=0)) * 0.5
    cols = dx[y0:y1, :].mean(0) + np.abs(np.diff(sd[y0:y1, :].mean(0), prepend=0)) * 0.5
    b = best_line(rows, fy + fh * 1.3, fy + fh * 4.0)
    t = best_line(rows, fy - fh * 2.5, fy - fh * 0.4)
    r = best_line(cols, fx + fw * 1.3, fx + fw * 4.0)
    l = best_line(cols, fx - fw * 3.0, fx - fw * 0.3)
    sx, sy = src_w / W_, src_h / H_
    L = (l + 3) * sx if l is not None else 0
    T = (t + 3) * sy if t is not None else 0
    R = (r - 2) * sx if r is not None else src_w
    B = (b - 2) * sy if b is not None else src_h
    return (int(L), int(T), int(R), int(B))
