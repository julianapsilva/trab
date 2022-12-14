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
        return f"{str(self.id)} conectado a: {str([x.id for x in self.connectedTo])}"

    def getConnections(self):
        return self.connectedTo.keys()


class Graph:

    # O(3n + 5) = O(n)
    def __init__(self, numVertices):
        self.vertList = {}
        self.numVertices = numVertices
        self.adjMatrix = []
        for i in range(numVertices):
            self.adjMatrix.append([0 for i in range(numVertices)])
        self.create_random_graph()

    def addVertex(self, key):
        newVertex = Vertex(key)
        self.vertList[key] = newVertex
        return newVertex

    # O(5n + 4) = O(n)
    def create_random_graph(self):
        for i in range(0, self.numVertices):
            self.addVertex(i)

        for i in range(0, int(self.numVertices/2)):
            start = randint(0, self.numVertices-1)
            finish = randint(0, self.numVertices-1)
            if (start != finish):
                self.addEdge(start, finish)
            elif start != i:
                self.addEdge(start, i)
            else:
                self.addEdge(i, finish)

    def getVertex(self, key):
        return self.vertList.get(key)

    def __contains__(self, key):
        return key in self.vertList

    def addEdge(self, f, t, weight=0):
        if f not in self.vertList:
            nv = self.addVertex(f)
        if t not in self.vertList:
            nv = self.addVertex(t)

        self.vertList[f].addNeighbor(self.vertList[t], weight)
        self.vertList[t].addNeighbor(self.vertList[f], weight)

        self.adjMatrix[f][t] = 1
        self.adjMatrix[t][f] = 1

    def printVertice(self, v):
        print(self.vertList[v].__str__())

    # O(2n + 2) = O(n)
    def print_adjacency(self):
        for i in range(0, self.numVertices):
            self.printVertice(i)

    def print_matrix(self):
        # for row in self.adjMatrix:
        #     print(row)
        print(np.array(self.adjMatrix))

    # O(2n?? + 3n + 5) = O(n??)
    def get_binary_array(self):
        binary_array = []
        for i in range(0, self.numVertices):
            for j in range(i+1, self.numVertices):
                if (i < j):
                    binary_array.append(self.adjMatrix[i][j])
        return binary_array

    # O(3n + 3) = O(n)
    def compacted_array(self):
        binArray = self.get_binary_array()
        compacted = []
        for i in range(0, int(self.numVertices*(self.numVertices-1)/2)):
            if(binArray[i]):
                compacted.append(i)
        return compacted

    # O(2n?? + 5n + 13) = O(n??)
    def get_matrix_from_compacted_array(self):
        compacted = self.compacted_array()
        matrix = []
        index = 0
        pos = 0
        for q in range(self.numVertices):
            matrix.append([0 for t in range(self.numVertices)])
        for i in range(0, self.numVertices):
            for j in range(i+1, self.numVertices):
                if(pos < len(compacted) and index == compacted[pos]):
                    matrix[i][j] = 1
                    matrix[j][i] = 1
                    pos += 1
                index += 1
        return matrix

    # O(2^n)
    def q6_global(self, flag):
        matrix = self.adjMatrix
        array = self.compacted_array()
        for i in range(0, self.numVertices):
            for j in range(i+1, self.numVertices):
                if matrix[i][j]:
                    if flag == 1:
                        print(self.q6_analitica(i, j, array))
                    elif flag == 2:
                        print(self.q6_iterativa(i, j, array, matrix))
                    else:
                        print(self.q6_recursiva(i, j, array, matrix))

    # O(3n + 4) = O(n)
    def q6_analitica(self, i, j, array):
        indexArray = (int)((self.numVertices*i - i*(i+1)/2)+j-(i+1))
        for k in range(0, len(array)):
            if(array[k] == indexArray):
                return array[k]
        return -1

    # O(2n?? + 4n + 3) = O(n??)
    def q6_iterativa(self, i, j, array, matrix):
        index = -1
        for k in range(0, self.numVertices):
            for l in range(k+1, self.numVertices):
                index += matrix[k][l]
                if(k == i and l == j):
                    return array[index]
        return index

    # O(2^n)
    def util_q6_recursiva(self, i, j, k, l, index, matrix, array):
        index += matrix[k][l]
        if(k == i and l == j):
            return array[index]
        else:
            l += 1
            if(l >= self.numVertices):
                k += 1
                l = k + 1
            return self.util_q6_recursiva(i, j, k, l, index, matrix, array)

    # O(2^n)
    def q6_recursiva(self, i, j, array, matrix):
        index = -1
        return self.util_q6_recursiva(i, j, 0, 1, index, matrix, array)

    # O(n??)
    def q7_global(self, flag):
        array_size = len(self.compacted_array())
        for i in range(0, array_size):
            if flag == 1:
                self.q7_analitica(i)
            else:
                self.q7_iterativa(i)

    def q7_analitica(self, indexCompacted):
        matrix = self.adjMatrix
        array = self.compacted_array()
        index = array[indexCompacted]
        k = (int)(self.numVertices*(self.numVertices-1)/2 - index)
        i = (int)(self.numVertices - int(np.ceil((1+sqrt(8*k+1))/2.0)))
        j = (int)(-self.numVertices*i + i*(i+1)/2 + (i+1) + index)
        print("(", i, ",", j, ") -> ", matrix[i][j])

        # O(3n?? + n + 9) = O(n??)
    def q7_iterativa(self, indexCompacted):
        matrix = self.adjMatrix
        array = self.compacted_array()
        index = array[indexCompacted]
        count = 0
        for i in range(0, self.numVertices):
            for j in range(i+1, self.numVertices):
                if count == index:
                    print("(", i, ",", j, ") -> ", matrix[i][j])
                    return
                count += 1
        print("-1")

    def getVertices(self):
        return self.vertList.keys()

    def getCount(self):
        return self.numVertices

# O(3n + 3) = O(n)


def sum_matrix(vet1, vet2):
    res = []
    for i in range(len(vet1)):
        res.append(vet1[i] + vet2[i])
    return res

# O(3n + 3) = O(n)


def sum_matrix_uniao(vet1, vet2):
    res = []
    for i in range(len(vet1)):
        res.append(vet1[i] or vet2[i])
    return res

# O(3n + 3) = O(n)


def intersection_between_matrix(vet1, vet2):
    res = []
    for i in range(len(vet1)):
        res.append(vet1[i] and vet2[i])
    return res


if __name__ == "__main__":
    print("\n================ QUEST??O 1 ================\n")
    graph1 = Graph(10)
    graph1.print_adjacency()

    print("\n================ QUEST??O 2 ================\n")
    graph1.print_matrix()

    print("\n================ QUEST??O 3 ================\n")
    print(graph1.get_binary_array())

    print("\n================ QUEST??O 4 ================\n")
    print(graph1.compacted_array())

    print("\n================ QUEST??O 5 ================\n")
    print(np.array(graph1.get_matrix_from_compacted_array()))

    print("\n================ QUEST??O 6 ================\n")
    print("Anal??tica")
    graph1.q6_global(1)

    print("\nIterativa")
    graph1.q6_global(2)

    print("\nRecursiva")
    graph1.q6_global(3)

    print("\n================ QUEST??O 7 ================\n")
    print("Anal??tica")
    graph1.q7_global(1)

    print("\nIterativa")
    graph1.q7_global(2)

    print("\n================ QUEST??O 8 ================\n")
    graph2 = Graph(10)
    graph3 = Graph(10)

    vet1 = graph2.get_binary_array()
    vet2 = graph3.get_binary_array()
    print("vet1 =>", graph2.get_binary_array())
    print("vet2 =>", graph3.get_binary_array())
    print("soma =>", sum_matrix_uniao(vet1, vet2))

    print("\n\nvet1  =>", graph2.get_binary_array())
    print("vet2  =>", graph3.get_binary_array())
    print("inter =>", intersection_between_matrix(vet1, vet2))
