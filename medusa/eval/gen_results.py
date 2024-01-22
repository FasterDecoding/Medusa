import matplotlib.pyplot as plt
import copy
import networkx as nx
import torch
import argparse

def load_accuracy_table(path):
    test_accuracy = torch.load(path)
    accuracy_table = []
    for i in range(len(test_accuracy)):
        accuracy_table.append(test_accuracy[i].sum(0)/16100)
    return torch.stack(accuracy_table)

def get_node_expectation(accuracies, node):
    expectation = copy.deepcopy(accuracies[0, node[0]])
    for i in range(1, len(node)):
        expectation *= accuracies[i, node[i]]
    return expectation

def explore_graph(accuracies, max_depth, max_child, num_iterations):
    explored_nodes = {}
    accept_nodes = [tuple([0])]
    expectations = get_node_expectation(accuracies, accept_nodes[0])
    explored_nodes[tuple(accept_nodes[0])] = expectations
    
    for _ in range(num_iterations):
        # find neighbors
        neighbors = []
        for node in accept_nodes:
            if node[-1] < max_child[len(node) - 1] - 1:
                neighbor = list(copy.deepcopy(node))
                neighbor[-1] = neighbor[-1] + 1
                neighbors.append(neighbor)
            if len(node) < max_depth:
                neighbor = list(copy.deepcopy(node))
                neighbor.append(0)
                neighbors.append(neighbor)
                
        # find the best neighbor
        best_neighbor = None
        best_neighbor_expectation = 0
        for neighbor in neighbors:
            if tuple(neighbor) in accept_nodes:
                continue
            if tuple(neighbor) in explored_nodes:
                neighbor_expectation = explored_nodes[tuple(neighbor)]
            else:
                neighbor_expectation = get_node_expectation(accuracies, neighbor)
                explored_nodes[tuple(neighbor)] = neighbor_expectation
            if neighbor_expectation > best_neighbor_expectation:
                best_neighbor = neighbor
                best_neighbor_expectation = neighbor_expectation
        accept_nodes.append(tuple(best_neighbor))
        expectations += best_neighbor_expectation
        
    return accept_nodes

def plot_and_save_graph(accept_nodes, output_path):
    plt.figure(figsize=(40, 20)) 

    G = nx.DiGraph()

    for path in accept_nodes:
        for i in range(len(path)):
            if i == 0:
                parent = 'root'
            else:
                parent = tuple(path[:i])
            child = tuple(path[:i+1])
            G.add_edge(parent, child)

    pos = nx.nx_agraph.graphviz_layout(G, prog='dot')
    nx.draw(G, pos, with_labels=True, node_size=500, node_color="skyblue", font_size=10, width=2, edge_color="gray")
    plt.savefig(output_path)

def main():
    parser = argparse.ArgumentParser(description="Generate Results.")
    parser.add_argument('--accuracy-path', type=str, required=True, help="Path to load accuracy tensor.")
    parser.add_argument('--output-path', type=str, required=True, help="Path to save the generated graph.")
    parser.add_argument('--max-depth', type=int, default=5, help="Maximum depth of the graph.")
    parser.add_argument('--num-iterations', type=int, default=62, help="Number of exploration iterations.")
    parser.add_argument('--max-child', nargs='+', type=int, default=[10, 10, 10, 10, 10], help="Maximum number of children per depth.")

    args = parser.parse_args()

    accuracies = load_accuracy_table(args.accuracy_path)
    accept_nodes = explore_graph(accuracies, args.max_depth, args.max_child, args.num_iterations)
    
    print("Accepted Nodes:", accept_nodes)
    
    try:
        plot_and_save_graph(accept_nodes, args.output_path)
        print(f"Graph saved to {args.output_path}.")
    except Exception as e:
        print(f"Failed to save the graph due to the following error: {e}")
        print("Ensure that Graphviz and pygraphviz are installed and set up correctly.")

if __name__ == "__main__":
    main()
