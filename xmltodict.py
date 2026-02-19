"""Minimal xmltodict-compatible subset used by Safety-Gymnasium.

This local shim is used when the external ``xmltodict`` package is unavailable.
It implements only the parse/unparse behavior needed by this repository.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any
from xml.etree import ElementTree as ET


def _element_to_obj(elem: ET.Element) -> OrderedDict[str, Any]:
    node: OrderedDict[str, Any] = OrderedDict()

    for key, value in elem.attrib.items():
        node[f"@{key}"] = value

    children = list(elem)
    if children:
        grouped: OrderedDict[str, Any] = OrderedDict()
        for child in children:
            child_obj = _element_to_obj(child)
            tag, value = next(iter(child_obj.items()))
            if tag in grouped:
                if not isinstance(grouped[tag], list):
                    grouped[tag] = [grouped[tag]]
                grouped[tag].append(value)
            else:
                grouped[tag] = value
        node.update(grouped)

        text = (elem.text or "").strip()
        if text:
            node["#text"] = text
        return OrderedDict([(elem.tag, node)])

    text = (elem.text or "").strip()
    if text:
        if node:
            node["#text"] = text
            return OrderedDict([(elem.tag, node)])
        return OrderedDict([(elem.tag, text)])

    if node:
        return OrderedDict([(elem.tag, node)])
    return OrderedDict([(elem.tag, OrderedDict())])


def parse(xml_input: str | bytes, *_, **__) -> OrderedDict[str, Any]:
    """Parse XML text into an OrderedDict using xmltodict-like conventions."""
    if isinstance(xml_input, bytes):
        xml_input = xml_input.decode("utf-8")
    root = ET.fromstring(xml_input)
    return _element_to_obj(root)


def _append_children(parent: ET.Element, key: str, value: Any) -> None:
    if isinstance(value, list):
        for item in value:
            _append_children(parent, key, item)
        return
    child = _obj_to_element(key, value)
    parent.append(child)


def _obj_to_element(tag: str, value: Any) -> ET.Element:
    elem = ET.Element(tag)

    if isinstance(value, (dict, OrderedDict)):
        for key, sub_value in value.items():
            if key.startswith("@"):
                elem.set(key[1:], str(sub_value))

        text_value = value.get("#text")
        if text_value is not None:
            elem.text = str(text_value)

        for key, sub_value in value.items():
            if key.startswith("@") or key == "#text":
                continue
            _append_children(elem, key, sub_value)
        return elem

    if isinstance(value, list):
        for item in value:
            child = _obj_to_element(tag, item)
            elem.append(child)
        return elem

    if value is not None:
        elem.text = str(value)
    return elem


def unparse(data: dict[str, Any], pretty: bool = False, *_, **__) -> str:
    """Serialize an xmltodict-style dictionary into XML text."""
    if not isinstance(data, dict) or len(data) != 1:
        raise ValueError("unparse expects a dict with exactly one root key")

    root_tag = next(iter(data))
    root_val = data[root_tag]
    root = _obj_to_element(root_tag, root_val)
    xml = ET.tostring(root, encoding="unicode")

    if pretty:
        try:
            from xml.dom import minidom

            return minidom.parseString(xml).toprettyxml(indent="  ")
        except Exception:
            return xml
    return xml
