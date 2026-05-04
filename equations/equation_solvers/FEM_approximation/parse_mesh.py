import gmshparser

def parse_mesh(mesh_path):

    nodes = []
    elements = []
    # Parse the .msh file
    mesh = gmshparser.parse(mesh_path)

    # Extract all nodes
    for entity in mesh.get_node_entities():
        for node in entity.get_nodes():
            nid = node.get_tag()
            coords = node.get_coordinates()
            print(f"Node {nid}: {coords}")
            nodes.append(coords)

    # Extract all elements
    for entity in mesh.get_element_entities():
        eltype = entity.get_element_type()
        for element in entity.get_elements():
            elid = element.get_tag()
            conn = element.get_connectivity()
            print(f"Element {elid} (type {eltype}): {conn}")   
            elements.append(conn)

    nodes = [[x, y] for x, y, z in nodes] # convert (x,y,z) to [x,y]
    return nodes, elements