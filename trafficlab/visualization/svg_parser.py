"""
SVGLayoutParser — parses an SVG map file into Qt graphics items for satellite view rendering.

Each named layer (Background, Aesthetic, Guidelines, Physical, Anchors) is parsed into a
list of QGraphicsLineItem / QGraphicsPolygonItem, transformed by the supplied affine
alignment matrix so that SVG coordinates map to satellite-image pixel space.
"""

import math
import re
import xml.etree.ElementTree as ET

import numpy as np

from PyQt5.QtCore import Qt, QPointF, QLineF
from PyQt5.QtGui import QColor, QPen, QBrush, QPolygonF
from PyQt5.QtWidgets import QGraphicsLineItem, QGraphicsPolygonItem


class SVGLayoutParser:
    """Parse an SVG layout file and expose Qt graphics items keyed by layer name."""

    def __init__(self, svg_path, affine_matrix_array):
        self.tree = ET.parse(svg_path)
        self.root = self.tree.getroot()
        self.ns = {'svg': 'http://www.w3.org/2000/svg'}

        self.M_align = np.identity(3)
        if affine_matrix_array:
            self.M_align[:2, :] = np.array(affine_matrix_array)

        # Pre-load CSS classes
        self.css_classes = {
            'cls-1': {'fill': '#afafaf'},
            'cls-2': {'fill': '#939393'},
            'cls-3': {'fill': '#fff', 'stroke': 'none'},
            'cls-4': {'fill': 'none', 'stroke': '#ff0', 'stroke-width': '1px'},
            'cls-5': {'fill': 'none', 'stroke': 'lime', 'stroke-width': '1px'},
            'cls-6': {'fill': '#fff', 'stroke': '#000', 'stroke-width': '2px'},
            'cls-7': {'fill': 'red'}
        }
        self._parse_css_from_file()

        self.layer_items = {
            'Background': [], 'Aesthetic': [], 'Guidelines': [],
            'Physical': [], 'Anchors': []
        }
        self._parse_layers()

    # ------------------------------------------------------------------
    # CSS helpers
    # ------------------------------------------------------------------

    def _parse_css_from_file(self):
        style_elem = self.root.find('.//svg:style', self.ns)
        if style_elem is None:
            style_elem = self.root.find('.//style')

        if style_elem is not None and style_elem.text:
            clean_css = re.sub(r'/\*.*?\*/', '', style_elem.text, flags=re.DOTALL)
            for match in re.finditer(r'([^{]+)\{(.*?)\}', clean_css, re.DOTALL):
                selectors_str = match.group(1)
                content = match.group(2)
                props = {}
                for prop_match in re.finditer(r'([\w-]+)\s*:\s*([^;]+)', content):
                    key, val = prop_match.group(1).strip(), prop_match.group(2).strip()
                    props[key] = val
                for sel in selectors_str.split(','):
                    cls_name = sel.strip().lstrip('.')
                    if cls_name in self.css_classes:
                        self.css_classes[cls_name].update(props)
                    else:
                        self.css_classes[cls_name] = props

    # ------------------------------------------------------------------
    # Transform helpers
    # ------------------------------------------------------------------

    def _parse_transform_str(self, txt):
        M = np.identity(3)
        if not txt:
            return M
        ops = re.findall(r'(\w+)\s*\(([^)]+)\)', txt)
        for name, args in ops:
            vals = list(map(float, filter(None, re.split(r'[ ,]+', args.strip()))))
            T = np.identity(3)
            if name == 'translate':
                T[0, 2], T[1, 2] = vals[0], vals[1] if len(vals) > 1 else 0
            elif name == 'rotate':
                rad = math.radians(vals[0])
                c, s = math.cos(rad), math.sin(rad)
                if len(vals) == 3:
                    cx, cy = vals[1], vals[2]
                    T1 = np.eye(3); T1[0, 2] = cx; T1[1, 2] = cy
                    R = np.eye(3); R[:2, :2] = [[c, -s], [s, c]]
                    T2 = np.eye(3); T2[0, 2] = -cx; T2[1, 2] = -cy
                    T = T1 @ R @ T2
                else:
                    T[:2, :2] = [[c, -s], [s, c]]
            elif name == 'matrix' and len(vals) == 6:
                T = np.array([[vals[0], vals[2], vals[4]],
                              [vals[1], vals[3], vals[5]],
                              [0, 0, 1]])
            M = M @ T
        return M

    def _apply_transform(self, pts, elem_matrix):
        M_total = self.M_align @ elem_matrix
        homo = np.hstack([pts, np.ones((len(pts), 1))])
        return (M_total @ homo.T).T[:, :2]

    # ------------------------------------------------------------------
    # Style / paint helpers
    # ------------------------------------------------------------------

    def _get_qt_style(self, elem, layer_name):
        styles = {'stroke': 'none', 'stroke-width': '1px', 'fill': 'none'}
        if layer_name == 'Physical':
            styles.update({'fill': '#fff', 'stroke': '#000'})
        elif layer_name == 'Guidelines':
            styles.update({'stroke': '#ff0', 'fill': 'none'})

        cls_str = elem.get('class')
        if cls_str:
            for c in cls_str.split():
                if c in self.css_classes:
                    styles.update(self.css_classes[c])

        style_attr = elem.get('style')
        if style_attr:
            for prop in style_attr.split(';'):
                if ':' in prop:
                    k, v = prop.split(':', 1)
                    styles[k.strip()] = v.strip()

        for key in ['stroke', 'stroke-width', 'fill']:
            val = elem.get(key)
            if val:
                styles[key] = val

        pen = QPen(Qt.NoPen)
        brush = QBrush(Qt.NoBrush)
        s_col = styles.get('stroke')
        if s_col and s_col != 'none':
            try:
                c = QColor(s_col)
                if c.isValid():
                    pen = QPen(c, float(styles.get('stroke-width', '1').replace('px', '')))
            except Exception:
                pass
        f_col = styles.get('fill')
        if f_col and f_col != 'none':
            try:
                c = QColor(f_col)
                if c.isValid():
                    brush = QBrush(c)
            except Exception:
                pass
        return pen, brush

    # ------------------------------------------------------------------
    # Layer parsing
    # ------------------------------------------------------------------

    def _parse_layers(self):
        for layer_name in self.layer_items.keys():
            group = self.root.find(f".//svg:g[@id='{layer_name}']", self.ns)
            if group is None:
                group = self.root.find(f".//*[@id='{layer_name}']")
            if group is None:
                continue

            for elem in group:
                tag = elem.tag.split('}')[-1]
                mat = self._parse_transform_str(elem.get('transform'))
                pen, brush = self._get_qt_style(elem, layer_name)

                item = None
                if tag == 'line':
                    try:
                        p1 = [float(elem.get('x1', 0)), float(elem.get('y1', 0))]
                        p2 = [float(elem.get('x2', 0)), float(elem.get('y2', 0))]
                        t = self._apply_transform(np.array([p1, p2]), mat)
                        item = QGraphicsLineItem(QLineF(QPointF(*t[0]), QPointF(*t[1])))
                        item.setPen(pen)
                    except Exception:
                        pass
                elif tag in ['rect', 'polygon', 'polyline']:
                    pts = []
                    if tag == 'rect':
                        try:
                            x = float(elem.get('x', 0))
                            y = float(elem.get('y', 0))
                            w = float(elem.get('width', 0))
                            h = float(elem.get('height', 0))
                            pts = np.array([[x, y], [x+w, y], [x+w, y+h], [x, y+h]])
                        except Exception:
                            pass
                    else:
                        raw = re.split(r'[ ,]+', elem.get('points', '').strip())
                        raw = [x for x in raw if x]
                        if raw:
                            try:
                                pts = np.array(raw, dtype=float).reshape(-1, 2)
                            except Exception:
                                pass
                    if len(pts) > 0:
                        t = self._apply_transform(pts, mat)
                        qpoly = QPolygonF([QPointF(*p) for p in t])
                        item = QGraphicsPolygonItem(qpoly)
                        item.setPen(pen)
                        item.setBrush(brush)

                if item:
                    self.layer_items[layer_name].append(item)
