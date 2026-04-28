import numpy as np
import math
import os
import re
import xml.etree.ElementTree as ET


class SVGParser:
    """
    Parses SVG lines/polygons for orientation guidelines.
    ROBUST VERSION: Ignores namespaces and finds deeply nested IDs.
    """
    def __init__(self, svg_path, alignment_matrix_A=None):
        self.svg_path = svg_path
        self.orientation_segments = []
        self.M_align = np.identity(3)
        if alignment_matrix_A is not None:
            self.M_align[:2, :] = np.array(alignment_matrix_A)
            
        self.valid = False
        if os.path.exists(svg_path):
            try:
                self.tree = ET.parse(svg_path)
                self.root = self.tree.getroot()
                self.orientation_segments = self._extract_segments()
                self.valid = True
            except Exception as e:
                print(f"[SVG ERR] {e}")
        else:
            print(f"[SVG ERR] File not found: {svg_path}")

    def _parse_transform(self, txt):
        M = np.identity(3)
        if not txt: return M
        ops = re.findall(r'(\w+)\s*\(([^)]+)\)', txt)
        for name, args in ops:
            vals = list(map(float, filter(None, re.split(r'[ ,]+', args.strip()))))
            T = np.identity(3)
            if name == 'translate':
                T[0,2], T[1,2] = vals[0], vals[1] if len(vals) > 1 else 0
            elif name == 'rotate':
                rad = math.radians(vals[0])
                c, s = math.cos(rad), math.sin(rad)
                if len(vals) == 3:
                    cx, cy = vals[1], vals[2]
                    T1=np.eye(3); T1[0,2]=cx; T1[1,2]=cy
                    R=np.eye(3); R[:2,:2]=[[c,-s],[s,c]]
                    T2=np.eye(3); T2[0,2]=-cx; T2[1,2]=-cy
                    T = T1 @ R @ T2
                else:
                    T[:2,:2] = [[c,-s],[s,c]]
            elif name == 'matrix':
                T = np.array([[vals[0], vals[2], vals[4]],
                              [vals[1], vals[3], vals[5]],
                              [0, 0, 1]])
            M = M @ T
        return M

    def _extract_segments(self):
        segs = []
        target_ids = ['Guidelines', 'Physical']
        
        def get_tag(el):
            return el.tag.split('}')[-1]

        target_nodes = []
        for el in self.root.iter():
            if get_tag(el) == 'g' and el.get('id') in target_ids:
                target_nodes.append(el)

        for g in target_nodes:
            self._process_node(g, np.identity(3), segs)
        return segs

    def _process_node(self, element, parent_mat, seg_list):
        local_mat = self._parse_transform(element.get('transform'))
        curr_mat = parent_mat @ local_mat
        
        tag = element.tag.split('}')[-1]
        pts = []
        
        if tag == 'line':
            pts = np.array([[float(element.get('x1',0)), float(element.get('y1',0))],
                            [float(element.get('x2',0)), float(element.get('y2',0))]])
        elif tag == 'polygon' or tag == 'polyline':
            raw = re.split(r'[ ,]+', element.get('points','').strip())
            raw = [x for x in raw if x]
            if raw: pts = np.array(raw, dtype=float).reshape(-1,2)
        
        if len(pts) > 0:
            homo = np.hstack([pts, np.ones((len(pts), 1))])
            t_pts = (self.M_align @ (curr_mat @ homo.T)).T[:, :2]
            for i in range(len(t_pts)-1):
                seg_list.append((t_pts[i], t_pts[i+1]))
            if tag == 'polygon':
                seg_list.append((t_pts[-1], t_pts[0]))

        for child in element:
            self._process_node(child, curr_mat, seg_list)

    def get_nearest_heading(self, pt):
        if not self.valid or not self.orientation_segments: return None
        min_d = float('inf')
        best_ang = None
        pt = np.array(pt)
        for sp1, sp2 in self.orientation_segments:
            ab = sp2 - sp1
            ab_sq = np.dot(ab, ab)
            if ab_sq < 1e-6: continue
            ap = pt - sp1
            t = np.dot(ap, ab) / ab_sq
            closest = sp1 + np.clip(t, 0, 1) * ab
            d = np.linalg.norm(pt - closest)
            if d < min_d:
                min_d = d
                best_ang = math.degrees(math.atan2(ab[1], ab[0]))
        if best_ang is not None:
            return (best_ang + 360) % 360
        return None
