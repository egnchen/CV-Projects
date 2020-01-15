import numpy as np
import cv2
import numpy as np
import math


def bottleCapDetect(img):
    scale_factor = min(1024 / img.shape[0], 768 / img.shape[1], 1)
    img = cv2.resize(img, (0, 0), fx=scale_factor, fy=scale_factor)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edge_threshold = 10
    edeg_threahold_max = 2 * edge_threshold
    blurred = cv2.medianBlur(gray, 3)
    edges = cv2.Canny(blurred, edge_threshold, edeg_threahold_max, apertureSize=3)
    struct_ele = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edges_closed = edges.copy()
    edges_closed = cv2.morphologyEx(edges_closed, cv2.MORPH_DILATE, struct_ele, iterations=1)
    edges_closed = cv2.morphologyEx(edges_closed, cv2.MORPH_CLOSE, struct_ele, iterations=5)
    mask = np.zeros((edges_closed.shape[0] + 2, edges_closed.shape[1] + 2), dtype="uint8")
    cv2.floodFill(edges_closed, mask, (0, 0), 255, flags=cv2.FLOODFILL_MASK_ONLY)[1]
    mask[...] = 255 * (1 - mask)
    cap_cnt = 10

    def get_markers(mask_orig):
        mask = mask_orig.copy()
        cnt = 0
        result = []
        while cnt < cap_cnt:
            mask = cv2.medianBlur(mask, 9)
            # do distance transform
            dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
            dist = dist.astype("uint8")
            ret, markers_binary = cv2.threshold(dist, 0.8 * dist.max(), 255, 0)
            # do marker labelling
            ret, markers = cv2.connectedComponents(markers_binary)
            cur_cnt = markers.max()
            print("Got", cur_cnt, "marker(s)")
            cnt += cur_cnt
            cur_result = []
            for i in range(1, cur_cnt + 1):
                pos = np.nonzero(markers == i)
                x, y = pos[1], pos[0]
                minx, maxx, miny, maxy = x.min(), x.max(), y.min(), y.max()
                w = np.max(dist[markers == i])
                cur_result.append(((minx, miny), (maxx, maxy), w))
            result.extend(cur_result)
            if cnt < cap_cnt:
                for i in range(cur_cnt):
                    (minx, miny), (maxx, maxy), w = cur_result[i]
                    radius = w + 20  # just in case :)
                    print("Removing", (minx, miny), (maxx, maxy), radius)
                    mask = cv2.circle(mask, ((minx + maxx) // 2, (miny + maxy) // 2), radius, 0, -1)
            elif cnt > cap_cnt:
                print("warning: 翻车啦")
        return result

    markers = get_markers(mask)
    # now prepare for watersheding
    ws = np.logical_not(mask).astype('int32')
    for i, (p1, p2, w) in enumerate(markers):
        center = (p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2
        axis = ((p2[0] - p1[0]) // 2, (p2[1] - p1[1]) // 2)
        cv2.ellipse(ws, center, axis, 0, 0, 360, i + 2, cv2.FILLED)
    flooded = ws.copy()
    flooded = cv2.watershed(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), flooded)
    flooded[...] = flooded - 1
    flooded[flooded <= 0] = 0
    # dilate a little to remove tiny edges
    flooded = flooded.astype("uint8")
    flooded = cv2.morphologyEx(flooded, cv2.MORPH_DILATE, struct_ele, iterations=1)
    ws = flooded
    boxed = img.copy()
    bounding_boxes = []
    minimal_bounding_boxes = []
    for i in range(cap_cnt):
        # value in marker image is i+1
        buf = (ws == i + 1).astype('uint8')
        contours, hierarchy = cv2.findContours(buf, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        assert (len(contours) == 1)
        x, y, w, h = cv2.boundingRect(contours[0])
        bounding_boxes.append(((x, y), (x + w, y + h)))
        minimal_bounding_boxes.append(cv2.minAreaRect(contours[0]))
        ellipse = cv2.fitEllipse(contours[0])
        cv2.ellipse(boxed, ellipse, (0, 255, 0), 2)
        box_points = cv2.boxPoints(minimal_bounding_boxes[-1])
        box_points = np.int0(box_points)
        cv2.rectangle(boxed, bounding_boxes[-1][0], bounding_boxes[-1][1], (255, 0, 0), 2)
        cv2.drawContours(boxed, [box_points], 0, (0, 0, 255), 2)
    real_caps = []
    cap_edges = []
    real_ans = []
    cord = []
    edges = cv2.Canny(img, 40, 80)
    for i, ((p1, p2), (center, (width, height), a)) in enumerate(zip(bounding_boxes, minimal_bounding_boxes)):
        g = gray[p1[1]:p2[1], p1[0]:p2[0]]
        e = edges[p1[1]:p2[1], p1[0]:p2[0]]
        cord.append(center)
        if width / height > 1.5 or height / width > 1.5:
            print(i, "is a side")
            real_ans.append('side')
        else:
            real_ans.append(None)
            # do circle approximation for better center mounting
            circles = cv2.HoughCircles(g, cv2.HOUGH_GRADIENT, 2, 40, 1, 20, 40, 200)
            if type(circles[0][0]) == np.ndarray:
                for i in circles[0, :]:
                    # draw the outer circle
                    cv2.circle(g, (i[0], i[1]), i[2], 255, 2)
                    # draw the center of the circle
                    cv2.circle(g, (i[0], i[1]), 2, 255, 3)
        real_caps.append(g)
        cap_edges.append(e)
    k = len(real_caps)
    t = cv2.getGaussianKernel(9, 1)
    gaussian_kernel = t * t.T
    gaussian_kernel *= 255 / np.max(gaussian_kernel)
    gaussian_kernel = np.uint8(gaussian_kernel)
    gaussian_kernel = cv2.resize(gaussian_kernel, (64, 64))
    for no, (cap, edge) in enumerate(zip(real_caps, cap_edges)):
        if real_ans[no] is not None:
            print("Skipping", no, "it is a side")
            continue
        c = cv2.resize(cap, (128, 128))
        e = cv2.resize(edge, (128, 128))
        rx, ry = c.shape[0] // 4, c.shape[1] // 4
        cx, cy = c.shape[0] // 2, c.shape[1] // 2
        score = np.sum(e[cx - rx:cx + rx, cy - ry:cy + ry] * gaussian_kernel)
        print(score)
        real_ans[no] = 'top' if score > 50000 else 'tail'
    labelled = img.copy()
    for i in range(len(real_ans)):
        if real_ans[i] == 'top':
            color = (255, 0, 0)
        elif real_ans[i] == 'tail':
            color = (0, 255, 0)
        else:
            color = (0, 0, 255)
        cv2.rectangle(labelled, bounding_boxes[i][0], bounding_boxes[i][1], color, 2)
        cv2.putText(labelled, real_ans[i], bounding_boxes[i][0],
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(labelled, "({},{})".format(int(cord[i][0]), int(cord[i][1])),
                    (int(cord[i][0]) - 50, int(cord[i][1])), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, color, 2)
    return labelled
