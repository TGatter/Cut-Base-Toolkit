import yeh
import argparse

def print_cut_enumeration(cut_list):
    col_widths = [8, 40]
    # Print the header with proper alignment
    print(f"{'Value':<{col_widths[0]}} {'Partition':<{col_widths[1]}}")
    print("-" * (sum(col_widths) + 2))
    # Print each row of the table
    for cut in cut_list:
        print(f"{cut[0]:<{col_widths[0]}} {str(cut[4]):<{col_widths[1]}}")

def run_algorithm(implementation, graph_path, show_all):
    algorithm_module = importlib.import_module(implementation)
    return yeh.MinCutEnumerator().run(graph_path, show_all)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--graph", help="Specifies the graph to be used.", required=True)
    parser.add_argument("-a", "--all", help="Specifies if all cuts should be examined (or just 2-partition-cuts)", action=argparse.BooleanOptionalAction)
    args = parser.parse_args()

    result = yeh.MinCutEnumerator().run(args.graph, args.all)
    print_cut_enumeration(result)
