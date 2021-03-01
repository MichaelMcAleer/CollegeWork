# Assignment 2 - Planning & Scheduling
# Problem 3 - Timetabling
# Michael McAleer (R00143621)
import csv
import itertools
import random
from timeit import default_timer as timer


def build_timetable_graph(file_name):
    """
    Read the contents of a CSV file to get timetable data, from data build a
    graph containing all nodes with their connected neighbours.

    :param file_name: (str) The path to CSV file containing timetable data
    :return: tt_graph: (dict) All nodes with a list of connected neighbours
    """
    # Open the file
    with open(file_name) as csv_file:
        # Read the contents of the CSV
        csv_reader = csv.reader(csv_file, delimiter=',')
        # Initialise the timetable graph dict
        tt_graph = dict()
        # For each row in the CSV file
        for row in csv_reader:
            # Initialise list of meetings
            meetings = list()
            # If the row details the meetings
            if 'Meetings' in row[0]:
                # Add the meeting to the graph and initialise contents as list,
                # this will hold all the node IDs connected to the current node
                for i in range(1, len(row)):
                    tt_graph[int(row[i])] = list()
            # Else the row contains meeting schedule info
            else:
                # For each of the columns in the row
                for n in range(1, len(row)):
                    # If the value is equal to 1 (meeting is attended)...
                    if int(row[n]):
                        # Append the meeting ID to the meeting list
                        meetings.append(n)

            # For each edge in all possible edge combinations with the meetings
            # selected in this row, if the edge does not have the same meeting
            # ID for both vertices, and the meeting is not already added, add
            # the node/neighbour edge to the graph
            for node, neighbour in [x for x in (itertools.product(
                    meetings, repeat=2)) if (x[0] != x[1])]:
                if neighbour not in tt_graph[node]:
                    tt_graph[node].append(neighbour)

        # For each node in the graph, sort their connected neighbour meeting
        # IDs in ascending order
        for node in tt_graph:
            tt_graph[node] = sorted(tt_graph[node])

        return tt_graph


def colour_graph(tt_graph, largest=False, first_fit=False):
    """
    Given a timetable graph, perform 'largest degree first' or 'first fit'
    colouring to solve timetable scheduling problem.

    :param tt_graph: (dict) All nodes with a list of connected neighbours
    :param largest: (bool) If the colouring mode is 'largest degree first'
    :param first_fit: (bool) If the colouring mode is 'first fit'
    :return: mode: (str) The graph colouring mode
    :return: node_colour_map: (dict) Dict containing all nodes and their colour
    """
    # Initialise the colour map dict
    node_colour_map = dict()
    # Initialise the colour order list
    node_order = list()
    # Initialise mode placeholder
    mode = None

    # >> 'Largest Degree First' mode
    if largest:
        # Order nodes in descending order by amount of edges
        node_order = sorted(list(graph.keys()), key=lambda x: len(graph[x]),
                            reverse=True)
        mode = 'LDO'
    # >> 'First Fit' mode - randomise the order of the nodes to simulate
    # unpredictable order
    elif first_fit:
        # Create a list of IDs that match the nodes in the graph
        node_order = list(range(1, len(tt_graph) + 1))
        # Shuffle the node order
        random.shuffle(node_order)
        mode = 'FF'

    # For each node in the order of nodes...
    for node in node_order:
        # Initialise list of colours equal to length of amount of nodes, this
        # is the absolute maximum amount of colours possible
        colours = [True] * len(node_order)
        # For every neigbour of the current node
        for neighbour in tt_graph[node]:
            # If the neighbour has been assigned a colour and added to the
            # colour map...
            if neighbour in node_colour_map:
                # Get the colour of the neighbour
                colour = node_colour_map[neighbour]
                # Mark that colour as false in the list of available colours
                colours[colour] = False
        # >> First fit colour allocation
        # For each colour and availability status in the list of colours
        for colour, is_available in enumerate(colours):
            # If the colour is available (first available colour)
            if is_available:
                # Assign the current node the colour
                node_colour_map[node] = colour
                # Break out of for loop and repeat for next node until all
                # nodes are coloured
                break

    return mode, node_colour_map


if __name__ == '__main__':
    # Run timeTable graph colouring solution & time each
    start_build = timer()
    meetings_file_name = 'pt3_meetings.csv'
    graph = build_timetable_graph(meetings_file_name)
    start_colour = timer()
    print(graph)
    colour_mode, graph_colour = colour_graph(
        graph, largest=True, first_fit=False)
    finish_colour = timer()

    # Calculate times in sub-millisecond accuracy and round to 5 decimal places
    graph_build_time = round((start_colour - start_build) * 1000, 5)
    graph_colour_time = round((finish_colour - start_colour) * 1000, 5)
    total_time = round((finish_colour - start_build) * 1000, 5)

    # Output results
    print("{} Colour Solution: {}".format(colour_mode, graph_colour))
    print("Graph Build Time: \t{}ms".format(graph_build_time))
    print("Graph Colour Time: \t{}ms".format(graph_colour_time))
    print("Total time: \t\t{}ms".format(total_time))
