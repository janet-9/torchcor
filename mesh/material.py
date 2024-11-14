

class Properties:
    def __init__(self):
        self.element = {}
        self.nodal = {}
        self.functions = {}

    def add_element_property(self, name, type, map):
        self.element[name] = {'type': type, 'idmap': map}

    def add_nodal_property(self, name, type, map):
        self.nodal[name] = {'type': type, 'idmap': map}

    def add_function(self, name, definition):
        self.functions[name] = definition

    def element_property(self, name, type, element_id, region_id):
        # properties.add_element_properties("sigma_l", "region", diffusl)
        elem_type = self.element[name][type]
        if elem_type == "uniform":
            return self.element[name]["idmap"]
        elif elem_type == "region":
            return self.element[name]["idmap"][region_id]
        elif elem_type == "element":
            return self.element[name]["idmap"][type][element_id]
        else:
            raise Exception("Unknown element property type")

    def nodal_property(self, name, point_id, region_id):
        type = self.nodal[name]['type']
        if type == "uniform":
            return self.nodal[name]['idmap']
        elif type == "region":
            return self.nodal[name]['idmap'][region_id]
        elif type == "nodal":
            return self.nodal[name]['idmap'][point_id]
        else:
            raise Exception("Unknown nodal property type")

    def element_property_names(self):
        return list(self.element.keys())

    def nodal_property_names(self):
        return list(self.nodal.keys())

    def element_property_types(self, name):
        return self.element[name]['type']

    def nodal_property_type(self, name):
        return self.nodal[name]['type']

    def remove_element_property(self, name):
        self.element.pop(name)

    def clear_element_properties(self):
        self.element.clear()

    def clear_nodal_properties(self):
        self.nodal.clear()


