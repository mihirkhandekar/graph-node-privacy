

def get_edges(node_id, edgelist):
    neighbours = []
    sublist = edgelist[edgelist['target'] == node_id]
    sublist = sublist.append(edgelist[edgelist['source'] == node_id])
    neighbours.extend(list(sublist['target']))
    neighbours.extend(list(sublist['source']))
    neighbours = list(set(neighbours) - set([node_id]))
    return sublist, neighbours


def get_k_edges(node_id, k):
    sublist, neighbours = get_edges(node_id)
    if k > 0:
        for neighbour in neighbours:
            sublist = sublist.append(get_k_edges(neighbour, k-1))
    return sublist.drop_duplicates()
