import numpy as np
import base64

def xml_attr(xml, path, attr, default=None):
    result = xml.find(path, xml_namespace(xml)).get(attr)
    if result is None:
        result = default
    return result

def xml_attr_float(xml, path, attr, default=None):
    return float(xml_attr(xml, path, attr, default))

def xml_attr_bool(xml, path, attr, default=None):
    result = xml_attr(xml, path, attr, default)
    if result is None:
        return default
    if result == 'true' or result == "1":
        return True
    if result == 'false' or result == "0":
        return False
    return default 

def xml_float(xml, path, default=None):
    result = xml.find(path, xml_namespace(xml)).text
    if result is None:
        return default
    return float(result)

def xml_float_array(xml, path):
    return np.array(list(map(float, xml.find(path, xml_namespace(xml)).text.split())))

def xml_np_array(xml, path, type = np.double):
    return np.fromstring(xml.find(path, xml_namespace(xml)).text, sep=' ', dtype=type)

def xml_np_binary_array(xml, path, type = np.double):
    data = xml.find(path, xml_namespace(xml)).text
    dcode = base64.b64decode(data)
    return np.frombuffer(dcode, dtype=np.float64)

def xml_attr_float_array(xml, path, attr):
    return np.array(list(map(float, xml.find(path, xml_namespace(xml)).get(attr).split())))

def xml_namespace(xml):
    return {'qif' : xml.tag.split('}')[0][1:]}