import numpy as np
from math import ceil
from numpy import sqrt
from random import randint


class Vertex:

    def __init__(self, key):
        self.id = key
        self.connectedTo = {}

    def addNeighbor(self, nbr, weight=0):
        self.connectedTo[nbr] = weight

    def __str__(self):
        return f"{str(self.id)} connected to: {str([x.id for x in self.connectedTo])}"

    def getConnections(self):
        return self.connectedTo.keys()

    def getId(self):
        return self.id

    def getWeight(self, nbr):
        return self.connectedTo.get(nbr)


class Graph:

    def __init__(self, numVertices):
        self.vertList = {}
        self.numVertices = numVertices
        self.adjMatrix = []
        for i in range(numVertices):
            self.adjMatrix.append([0 for i in range(numVertices)])
        self.create_random_graph()

    def addVertex(self, key):
        """
        Add a vertex to the Graph network with the id of key
        Time complexity is O(1) as we are only adding a single
        new vertex which does not affect any other vertex
        """

        # add 1 to the number of vertices attribute
        # self.numVertices += 1

        # instantiate a new Vertex class
        newVertex = Vertex(key)

        # add the vertex with the key to the vertList dictionary
        self.vertList[key] = newVertex

        # return the NewVertex created
        return newVertex

    def create_random_graph(self):
        for i in range(0, self.numVertices):
            self.addVertex(i)

        for i in range(0, int(self.numVertices/10)):
            start = randint(0, self.numVertices-1)
            finish = randint(0, self.numVertices-1)
            if (start != finish):
                self.addEdge(start, finish)
            elif start != i:
                self.addEdge(start, i)
            else:
                self.addEdge(i, finish)

    def getVertex(self, key):
        """
        If vertex with key is in Graph then return the Vertex
        Time complexity is O(1) as we are simply checking whether
        the key exists or not in a dictionary and returning it
        """

        # use the get method to return the Vertex if it exists
        # otherwise it will return None
        return self.vertList.get(key)

    def __contains__(self, key):
        """
        Check whether vertex with key is in the Graph
        Time complexity is O(1) as we are simply checking whether
        the key is in in the dictrionary or not
        """

        # returns True or False depending if in list
        return key in self.vertList

    def addEdge(self, f, t, weight=0):
        """
        Add an edge to connect two vertices of t and f with weight
        assuming directed graph

        Time complexity of O(1) as adding vertices if they don't
        exist and then add neighbor
        """

        # add vertices if they do not exist
        if f not in self.vertList:
            nv = self.addVertex(f)
        if t not in self.vertList:
            nv = self.addVertex(t)

        # then add Neighbor from f to t with weight
        self.vertList[f].addNeighbor(self.vertList[t], weight)
        self.vertList[t].addNeighbor(self.vertList[f], weight)

        self.adjMatrix[f][t] = 1
        self.adjMatrix[t][f] = 1

    def printVertice(self, v):
        print(self.vertList[v].__str__())

    def print_adjacency(self):
        for i in range(0, self.numVertices):
            self.printVertice(i)

    def print_matrix(self):
        # for row in self.adjMatrix:
        #     print(row)
        print(np.array(self.adjMatrix))

    def get_binary_array(self):
        binary_array = []
        for i in range(0, self.numVertices):
            for j in range(i+1, self.numVertices):
                if (i < j):
                    binary_array.append(self.adjMatrix[i][j])
        print("binary", binary_array)
        return binary_array

    def compacted_array(self):
        binArray = self.get_binary_array()
        compacted = []
        for i in range(0, int(self.numVertices*(self.numVertices-1)/2)):
            if(binArray[i]):
                compacted.append(i)
        print("compa", compacted)
        return compacted

    def get_matrix_from_compacted_array(self):
        compacted = self.compacted_array()
        matrix = []
        index = 0
        pos = 0
        for i in range(self.numVertices):
            matrix.append([0 for i in range(self.numVertices)])
        for i in range(0, self.numVertices):
            for j in range(i+1, self.numVertices):
                if(index == compacted[pos]):
                    matrix[i][j] = 1
                    matrix[j][i] = 1
                    pos += 1
                    print(i, j)
                index += 1
        print("matrix", matrix)
        return matrix

    def q6_global(self, flag):
        matrix = self.get_matrix_from_compacted_array()
        array = self.compacted_array()
        for i in range(0, self.numVertices):
            for j in range(i+1, self.numVertices):
                if matrix[i][j]:
                    if flag == 1:
                        self.q6_analitica(i, j, array)
                    elif flag == 2:
                        self.q6_iterativa(i, j, array, matrix)
                    else:
                        self.q6_recursiva(i, j, array, matrix)

    def q6_analitica(self, i, j, array):
        iStart = (int)(self.numVertices*i - i*(i+1)/2)
        indexArray = (int)(iStart+j-(i+1))
        for k in range(0, len(array)):
            if(array[k] == indexArray):
                return array[k]
        return -1

    def q6_iterativa(self, i, j, array, matrix):
        index = -1
        for k in range(0, self.numVertices):
            for l in range(k+1, self.numVertices):
                index += matrix[k][l]
                if(k == i and l == j):
                    return array[index]
        return index

    def utilCompactedIndexFromMatrix(self, i, j, k, l, index, matrix, array):
        index += matrix[k][l]
        if(k == i and l == j):
            return array[index]
        else:
            l += 1
            if(l >= self.numVertices):
                k += 1
                l = k + 1
            return self.utilCompactedIndexFromMatrix(i, j, k, l, index, matrix, array)

    def q6_recursiva(self, i, j, array, matrix):
        index = -1
        return self.utilCompactedIndexFromMatrix(i, j, 0, 1, index, matrix, array)

    def q7_global(self, flag):
        array_size = len(self.compacted_array())
        for i in range(0, array_size):
            if flag == 1:
                self.q7_getMatrixIndexFromCompacted_A(i)
            else:
                self.q7_getMatrixIndexFromCompacted_I(i)

    def q7_getMatrixIndexFromCompacted_A(self, indexCompacted):
        matrix = self.get_matrix_from_compacted_array()
        array = self.compacted_array()
        index = array[indexCompacted]
        Ic = (int)(self.numVertices*(self.numVertices-1)/2 - index)
        Ie = ceil((1+sqrt(8*Ic+1))/2.0)
        i = (int)(self.numVertices - int(Ie))
        j = (int)(-self.numVertices*i + i*(i+1)/2 + (i+1) + index)
        print("(", i, ",", j, ")#")
        return matrix[i][j]

    def q7_getMatrixIndexFromCompacted_I(self, indexCompacted):
        matrix = self.get_matrix_from_compacted_array()
        array = self.compacted_array()
        index = array[indexCompacted]
        count = 0
        for i in range(0, self.numVertices):
            for j in range(i+1, self.numVertices):
                if count == index:
                    print(index, "#")
                    return matrix[i][j]
                count += 1
        return -1

    def getVertices(self):
        """
        Return all the vertices in the graph
        Time complexity is O(1) as we simply return all the keys
        in the vertList dictionary
        """

        return self.vertList.keys()

    def getCount(self):
        """
        Return a count of all vertices in the Graph

        Time complexity O(1) because we just return the count
        attribute
        """
        return self.numVertices


def sum_matrix(vet1, vet2):
    res = []
    for i in range(len(vet1)):
        res.append(vet1[i] + vet2[i])
    return res


def sum_matrix_uniao(vet1, vet2):
    res = []
    for i in range(len(vet1)):
        res.append(vet1[i] or vet2[i])
    return res


def intersection_between_matrix(vet1, vet2):
    res = []
    for i in range(len(vet1)):
        res.append(vet1[i] and vet2[i])
    return res


# graph.print_matrix()
# graph.get_binary_array()
# graph.compacted_array()
# graph.get_matrix_from_compacted_array()
graph1 = Graph(10)
graph2 = Graph(10)

graph1.print_adjacency()
graph2.print_adjacency()


#  QUESTAO 8
vet1 = graph1.get_binary_array()
vet2 = graph2.get_binary_array()
print(sum_matrix_uniao(vet1, vet2))
# print(intersection_between_matrix(vet1, vet2))
