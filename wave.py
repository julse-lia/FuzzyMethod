import os

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as cntrl
from skfuzzy.control import Rule as rule
import sys
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QDialog, QApplication, QGraphicsScene, QMainWindow, \
    QGraphicsItem,  QGraphicsEllipseItem, QGraphicsLineItem, QFileDialog
from PyQt5.QtGui import  QPen, QFont, QColor
from PyQt5 import uic
from PyQt5.QtCore import Qt, QRectF, QLineF, QPoint
import random

# from py2puml.py2puml import py2puml

class Node:
    def __init__(self, number):
        self.number = number
        self.neighbors = {}

    def get_number(self):
        return self.number

    def add_adjacent(self, node, value=0):
        self.neighbors[node] = value

    def remove_adjacent(self, node):
        del self.neighbors[node]

    def get_cost(self, node):
        return self.neighbors[node]


class Graph:
    def __init__(self):
        self.nodes = {}

    def add_node(self, node):
        self.nodes[node] = Node(node)

    def add_edge(self, from_node, to_node, distance, load):
        if from_node not in self.nodes:
            self.add_node(from_node)
        if to_node not in self.nodes:
            self.add_node(to_node)

        self.nodes[from_node].add_adjacent(self.nodes[to_node], [distance, load])
        self.nodes[to_node].add_adjacent(self.nodes[from_node], [distance, load])

    def remove_edge(self, from_node, to_node):
        self.nodes[from_node].remove_adjacent(self.nodes[to_node])

    def get_nodes(self):
        return self.nodes.values()

    def get_node(self, node):
        return self.nodes[node]


class WaveAlgorithm:
    def __init__(self, graph, start, end):
        self.graph = graph
        self.start = start
        self.end = end
        # self.optimal_path = None
        self.optimal_paths = []
        self.opt_paths_params = []
        # self.disjoint = []

    def paths_finder(self, start_node, end_node, p=[]):
        p = p + [start_node]
        if start_node == end_node:
            return [p]
        if start_node not in self.graph.get_nodes():
            return []
        paths_list = []
        for neighbor in start_node.neighbors:
            if neighbor not in p:
                found_new_paths = self.paths_finder(neighbor, end_node, p)
                for new in found_new_paths:
                    paths_list.append(new)
        return paths_list

    def find_opt_paths(self, pathway=[]):
        paths_list = self.paths_finder(self.start, self.end, pathway)
        min_load = sys.maxsize
        min_metr = sys.maxsize
        hop_am = sys.maxsize
        min_path = None
        path_metr_load = []
        # optimal_paths = []

        for pathway in paths_list:
            metric = sum(i.get_cost(j)[0] for i, j in zip(pathway, pathway[1::]))
            #print('time = ', metric)
            loads = [i.get_cost(j)[1] for i, j in zip(pathway, pathway[1::])]
            #print('loads = ', loads)
            hop = len(loads)
            # print('\t\tevaluating:', path, t)
            max_load = max(loads)
            path_metr_load.append((pathway, metric, max_load))
            # print(max_t)
            # if (max_load < min_load and hop < hop_am) or metric < min_metr:
            #     min_load = max_load
            #     min_metr = metric
            #     hop_am = hop
            #     min_path = pathway
        # self.optimal_paths.append(min_path)
        # print("OPT = ", mpath, ' ', min_load, min_metr, hop_am)

        def intersect_check(path, paths):
            count = 0
            for p in paths:
                if set(path[1:-1]).intersection(set(p[1:-1])) == set():
                    count += 1
            return count

        sort_list = sorted(path_metr_load, key=lambda element: (element[2], element[1]))
        self.optimal_paths.append(sort_list[0][0])
        for path, metr, load in sort_list:
            if path not in self.optimal_paths:
                count = intersect_check(path, self.optimal_paths)
                if count == len(self.optimal_paths):
                    self.optimal_paths.append(path)

        # self.optimal_path = min_path
        self.show_path_params(self.optimal_paths)

    def show_path_params(self, paths):
        for path in paths:
            total_metric = sum(i.get_cost(j)[0] for i, j in zip(path, path[1::]))  # calc total metric sum
            path_loads = [i.get_cost(j)[1] for i, j in zip(path, path[1::])]       # list of loads on the path
            max_load = max(path_loads)                           # find max load on the path
            integer_path_view = [i.get_number() for i in path]   # turn nodes into their identifiers
            self.opt_paths_params.append((integer_path_view, (round(total_metric, 2), max_load)))  # paths and params


class FuzzyLogic:
    def __init__(self):
        self.time = 0
        self.load_coef = 0
        self.rating = 0

    def fuzzy_controler(self):
        time = cntrl.Antecedent(np.arange(0, 10.1, 0.1), 'Time')
        load_coef = cntrl.Antecedent(np.arange(0, 1.1, 0.1), 'Load')
        rating = cntrl.Consequent(np.arange(0, 101, 1), 'Rating')

        load_coef_names = ['low', 'middle', 'high']
        load_coef.automf(names=load_coef_names)

        time_names = ['short', 'average', 'long']
        time.automf(names=time_names)

        rating_names = ['very_small', 'small', 'medium', 'big', 'very_big']
        rating.automf(names=rating_names)

        rating['very_small'] = fuzz.trapmf(rating.universe, [0, 0, 17, 34])
        rating['small'] = fuzz.trimf(rating.universe, [17, 34, 50])
        rating['medium'] = fuzz.trimf(rating.universe, [34, 50, 68])
        rating['big'] = fuzz.trimf(rating.universe, [50, 68, 84])
        rating['very_big'] = fuzz.trapmf(rating.universe, [68, 84, 100, 100])

        # You can see how these look with .view()
        #distance['middle'].view()

        #load_coef.view()
        #time.view()
        #rating.view()
        # tip.view()
        r1 = rule(load_coef['low'] & time['short'], rating['very_big'])
        r2 = rule(load_coef['low'] & time['average'], rating['very_big'])
        r3 = rule(load_coef['low'] & time['long'], rating['big'])
        r4 = rule(load_coef['middle'] & time['short'], rating['big'])
        r5 = rule(load_coef['middle'] & time['average'], rating['medium'])
        r6 = rule(load_coef['middle'] & time['long'], rating['small'])
        r7 = rule(load_coef['high'] & time['short'], rating['small'])
        r8 = rule(load_coef['high'] & time['average'], rating['very_small'])
        r9 = rule(load_coef['high'] & time['long'], rating['very_small'])

        #rule1.view()
        rating_ctrl = cntrl.ControlSystem([r1, r2, r3, r4, r5, r6, r7, r8, r9])

        rating_show = cntrl.ControlSystemSimulation(rating_ctrl)
        rating_show.input['Load'] = self.load_coef
        rating_show.input['Time'] = self.time
        # rule9.view()

        # Crisp numbers
        rating_show.compute()

        #load_coef.view(sim=rating_show)
        #time.view(sim=rating_show)
        self.rating = rating_show.output['Rating']
        #print(rating_show.output['rating'])
        #rating.view(sim=rating_show)
        #plt.show()

class Weight_Add_Dialog(QDialog):
    def __init__(self):
        super(Weight_Add_Dialog, self).__init__()
        uic.loadUi("edgeweight_dialog.ui", self)
        self.setWindowTitle("QSetWeightDialog")

class Source_End_Dialog(QDialog):
    def __init__(self):
        super(Source_End_Dialog, self).__init__()
        uic.loadUi("source_end_input.ui", self)
        self.setWindowTitle("QSetSourceEndDialog")

class QGraph_Vertex(QGraphicsEllipseItem):
    def __init__(self, rect=QRectF(-75, -15, 70, 70), parent=None):

        QGraphicsEllipseItem.__init__(self, rect, parent)
        self.id = None
        self.vertex_edges = []
        self.setZValue(1)
        self.setPen(QPen(Qt.black, 2.4))
        self.setBrush(Qt.white)
        self.setFlags(QGraphicsItem.ItemIsMovable |
                      QGraphicsItem.ItemIsSelectable |
                      QGraphicsItem.ItemSendsGeometryChanges)

    def addEdge(self, edge):
        self.vertex_edges.append(edge)


    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemSelectedChange:
            self.setBrush(QColor(255,239,213) if value else Qt.white)

        if change == QGraphicsItem.ItemPositionHasChanged:
            for edge in self.vertex_edges:
                edge.adjust()

        return QGraphicsItem.itemChange(self, change, value)


class QGraph_Edge(QGraphicsLineItem):
    def __init__(self, source, dest, parent=None, label=None):
        QGraphicsLineItem.__init__(self, parent)
        self.setFlag(QGraphicsItem.ItemIsSelectable)
        self.source_vert = source
        self.dest_vert = dest
        self.source_vert.addEdge(self)
        self.dest_vert.addEdge(self)
        self.setPen(QPen(Qt.black, 2.4))
        self.adjust()

    def adjust(self):
        self.prepareGeometryChange()
        x_offset_dest = self.dest_vert.rect().x() + self.dest_vert.rect().width()/2
        y_offset_dest = self.dest_vert.rect().y() + self.dest_vert.rect().height()/2

        x_offset_source = self.source_vert.rect().x() + self.source_vert.rect().width()/2
        y_offset_source = self.source_vert.rect().y() + self.source_vert.rect().height()/2

        self.setLine(QLineF(self.dest_vert.pos().x() + x_offset_dest, self.dest_vert.pos().y() + y_offset_dest,
                            self.source_vert.x() + x_offset_source, self.source_vert.pos().y() + y_offset_source))
        # print(self.source.number, self.dest.number)

class QGraph_Scene(QGraphicsScene):

    def __init__(self, parent=None):
        QGraphicsScene.__init__(self, parent)
        self.node_number = 1
        # self.scene_vertexes = []
        self.verteces = []
        self.list_nodes_weights = []
        self.setBackgroundBrush(QColor(230,230,250))

class Main(QMainWindow):
    def __init__(self):
        super(Main, self).__init__()
        uic.loadUi("maket.ui", self)
        self.sd_dialog = Source_End_Dialog()
        self.main_scene = QGraph_Scene()
        self.dialog = Weight_Add_Dialog()
        self.graphicsView.setScene(self.main_scene)
        self.sd_dialog.sub_but.clicked.connect(self.set_source_end)
        self.dialog.submit_button.clicked.connect(self.nodes_weights_lists)
        self.actionAdd_Node.triggered.connect(self.add_node)
        # self.actionAdd_Edge.triggered.connect(self.add_edge)
        self.actionAdd_Edge.triggered.connect(self.add_weight)
        # self.actionAdd_Weight.triggered.connect(self.add_weight)
        self.actionDelete_Node.triggered.connect(self.delete_node)
        self.actionDelete_Edge.triggered.connect(self.delete_edge)
        self.actionClear_All.triggered.connect(self.clear_scene)
        self.actionSave.triggered.connect(self.save_file)
        self.actionOpen.triggered.connect(self.open_file)
        self.actionNew.triggered.connect(self.new_window)
        self.actionExit.triggered.connect(self.exit_window)
        self.actionInput_source_and_end.triggered.connect(self.input_source_dest)
        self.actionMake_Full_Modeling.triggered.connect(self.show_paths)
        self.actionSave.setShortcut("Ctrl+S")
        self.actionAdd_Node.setShortcut("Ctrl+N")
        self.actionAdd_Edge.setShortcut("Ctrl+E")
        self.actionAdd_Weight.setShortcut("Ctrl+W")
        self.actionDelete_Node.setShortcut("Ctrl+R")
        self.actionDelete_Edge.setShortcut("Ctrl+D")
        self.actionClear_All.setShortcut("Ctrl+C")
        self.actionOpen.setShortcut("Ctrl+O")
        self.actionNew.setShortcut("Ctrl+F")
        self.actionExit.setShortcut("Ctrl+X")
        self.actionInput_source_and_end.setShortcut("Ctrl+P")
        self.actionMake_Full_Modeling.setShortcut("Ctrl+M")
        self.start_node = None
        self.end_node = None

    def nodes_weights_lists(self):

        n1 = self.main_scene.selectedItems()[0].id
        n2 = self.main_scene.selectedItems()[1].id
        w1 = self.dialog.time_input.text()
        w2 = self.dialog.load_input.text()
        n1_x = self.main_scene.selectedItems()[0].x()
        n1_y = self.main_scene.selectedItems()[0].y()
        n2_x = self.main_scene.selectedItems()[1].x()
        n2_y = self.main_scene.selectedItems()[1].y()

        # if not self.main_scene.list_nodes_weights == []:
        #     for item in self.main_scene.list_nodes_weights:
        #         if n1 in item and n2 in item:
        #             self.main_scene.list_nodes_weights.remove(item)
        #             self.main_scene.list_nodes_weights.append([n1, n2, float(w1), float(w2),n1_x, n1_y,n2_x,n2_y])
        #         else:
        #             self.main_scene.list_nodes_weights.append([n1, n2, float(w1), float(w2),n1_x, n1_y,n2_x,n2_y])
        # else:
        self.main_scene.list_nodes_weights.append([n1, n2, float(w1), float(w2),n1_x, n1_y,n2_x,n2_y])
        w1 = self.dialog.time_input.setText("")
        w2 = self.dialog.load_input.setText("")
        self.dialog.close()

    def add_node(self):
        q_vertex = QGraph_Vertex()
        q_vertex.id = self.main_scene.node_number
        self.main_scene.verteces.append(q_vertex.id)
        node_label = QtWidgets.QLabel()
        node_label.setText('R'+str(self.main_scene.node_number))
        node_label.move(q_vertex.pos().x() + q_vertex.rect().x() + q_vertex.rect().width() / 2 - 20,
                   q_vertex.pos().y() + q_vertex.rect().y() + q_vertex.rect().height() / 2 - 14)
        node_label.setStyleSheet('background-color: transparent')
        node_label.setFont(QFont('Arial', 19))

        prox = QtWidgets.QGraphicsProxyWidget(q_vertex)
        prox.setWidget(node_label)
        self.main_scene.node_number += 1
        self.main_scene.addItem(q_vertex)

    def add_weight(self):
        if len(self.main_scene.selectedItems()) == 2:
            new_edge = QGraph_Edge(self.main_scene.selectedItems()[0], self.main_scene.selectedItems()[1])
            # print(self.selectedItems()[0], self.selectedItems()[1])
            self.main_scene.addItem(new_edge)
        if len(self.main_scene.selectedItems()) == 2:
            self.dialog.exec_()

    def delete_node(self):
        if len(self.main_scene.selectedItems()) == 1:
            if type(self.main_scene.selectedItems()[0]) == QGraph_Vertex:
                self.main_scene.verteces.remove(self.main_scene.selectedItems()[0].id)
                for elem in self.main_scene.items():
                    if type(elem) == QGraph_Edge:
                        if elem.source_vert.id == self.main_scene.selectedItems()[0].id:
                            for i in self.main_scene.list_nodes_weights:
                                if elem.source_vert.id in i and elem.dest_vert.id in i:
                                    self.main_scene.list_nodes_weights.remove(i)
                            self.main_scene.removeItem(elem)
                        elif elem.dest_vert.id == self.main_scene.selectedItems()[0].id:
                            for i in self.main_scene.list_nodes_weights:
                                if elem.source_vert.id in i and elem.dest_vert.id in i:
                                    self.main_scene.list_nodes_weights.remove(i)
                            self.main_scene.removeItem(elem)
                self.main_scene.removeItem(self.main_scene.selectedItems()[0])


    def delete_edge(self):
        if len(self.main_scene.selectedItems()) == 1:
            if type(self.main_scene.selectedItems()[0]) == QGraph_Edge:
                for list_elem in self.main_scene.list_nodes_weights:
                    if self.main_scene.selectedItems()[0].source_vert.id in list_elem and self.main_scene.selectedItems()[0].dest_vert.id in list_elem:
                        self.main_scene.list_nodes_weights.remove(list_elem)
                self.main_scene.removeItem(self.main_scene.selectedItems()[0])

    def clear_scene(self):

        self.main_scene.clear()
        self.main_scene.node_number = 1

    def save_file(self):
        file_options = QFileDialog.Options()
        file_options |= QFileDialog.DontUseNativeDialog
        file_path, _ = QFileDialog.getSaveFileName(self, "QSaveFileDialog", "",
                                                  "All Files (*);;Text Files (*.txt)", options=file_options)
        if file_path == "":
            return

        if os.path.exists(file_path):
            os.remove(file_path)

        for n1, n2, w1, w2, n1_x, n1_y, n2_x, n2_y in self.main_scene.list_nodes_weights:
            with open(file_path, 'a') as file:
                file.write(str(n1)+','+str(n2)+','+str(w1)+','+str(w2)+','+str(n1_x)+','+str(n1_y)+','+str(n2_x)+','+str(n2_y)+'\n')
        file.close()

    def open_file(self):
        file_options = QFileDialog.Options()
        file_options |= QFileDialog.DontUseNativeDialog
        file_path, _ = QFileDialog.getOpenFileName(self, "QOpenFileDialog", "",
                                                  "All Files (*);;Python Files (*.py)", options=file_options)
        if file_path == "":
            return

        with open(file_path) as f:
            lines = f.readlines()
            for line in lines:
                n1, n2, w1, w2, n1_x, n1_y, n2_x, n2_y = (line.strip().split(','))
                n_l = [(n1, n1_x, n1_y), (n2, n2_x, n2_y)]
                for n_i in n_l:
                    if int(n_i[0]) not in self.main_scene.verteces:
                        # print(int(s_i))
                        vert = QGraph_Vertex()
                        vert.id = int(n_i[0])
                        self.main_scene.verteces.append(vert.id)
                        # print(self.verteces)
                        node_label = QtWidgets.QLabel()
                        node_label.setText('R'+n_i[0])
                        node_label.move(vert.pos().x() + vert.rect().x() + vert.rect().width() / 2 - 16,
                                   vert.pos().y() + vert.rect().y() + vert.rect().height() / 2 - 15)
                        node_label.setStyleSheet('background-color: transparent')
                        node_label.setFont(QFont('Arial', 18))
                        proxy = QtWidgets.QGraphicsProxyWidget(vert)
                        proxy.setWidget(node_label)
                        self.main_scene.node_number += 1
                        point = QPoint(float(n_i[1]), float(n_i[2]))
                        point_item = vert.mapFromScene(point)
                        vert.setPos(point_item)
                        self.main_scene.addItem(vert)
                vert1 = None
                vert2 = None
                scene_elems = self.main_scene.items()
                # print(scene_items)
                for elem in scene_elems:
                    if type(elem) == QGraph_Vertex:
                        if elem.id == int(n1):
                            vert1 = elem
                        elif elem.id == int(n2):
                            vert2 = elem
                new_edge = QGraph_Edge(vert1, vert2)
                # print(self.selectedItems()[0], self.selectedItems()[1])
                self.main_scene.addItem(new_edge)
                self.main_scene.list_nodes_weights.append([int(n1), int(n2), float(w1), float(w2), float(n1_x),float(n1_y),float(n2_x),float(n2_y)])

    def new_window(self):

        self.my_app = QApplication(sys.argv)
        self.my_app.setApplicationName("Network Project")
        self.main_win = Main()
        self.my_widget = QtWidgets.QStackedWidget()
        self.my_widget.addWidget(self.main_win)
        self.my_widget.show()
        self.my_app.exec_()

    def exit_window(self):
        sys.exit(self.close())

    def set_source_end(self):
        # global s, d
        self.start_node = int(self.sd_dialog.source_inp.text())
        self.end_node = int(self.sd_dialog.end_inp.text())
        # print(self.start_node, self.end_node)
        self.sd_dialog.close()

    def input_source_dest(self):
        self.sd_dialog.exec_()

    def show_paths(self):
        # global nodes_number
        if self.start_node == None and self.end_node == None:
            self.disjoint_paths_field.setText("No start node and end node were entered!!!")
            return

        start_node = self.start_node
        end_node = self.end_node
        graph = Graph()

        for l in self.main_scene.list_nodes_weights:
            node1 = l[0]
            node2 = l[1]
            time_coef = l[2]
            load_coef = l[3]
            graph.add_edge(node1, node2, time_coef, load_coef)
        # start = graph.get_node(start_node)

        global start, end
        if end_node in self.main_scene.verteces:
            end = graph.get_node(end_node)
        else:
            self.disjoint_paths_field.setText("End node is not in graph!!!")
            return

        if start_node in self.main_scene.verteces:
            start = graph.get_node(start_node)
        else:
            self.disjoint_paths_field.setText("Start node is not in graph!!!")
            return

        str_alg = WaveAlgorithm(graph, start, end)

        str_alg.find_opt_paths()

        paths_params = str_alg.opt_paths_params  # list of tuples of paths and their params

        edges = []
        for path, params in paths_params:
            path_edges = [[i, j] for i, j in zip(path, path[1::])]
            for edge in path_edges:
                edges.append(edge)

        scen_items = self.main_scene.items()
        i = 0
        # colors = [QColor(177, 255, 99), QColor(255, 255, 80), QColor(255, 192, 203), QColor(235, 181, 235),
        #           QColor(255, 210, 0)]
        #
        # random.shuffle(colors)
        # used_col = []
        #
        # def rand_col():
        #
        #     c = random.choice(colors)
        #     if c not in used_col:
        #         used_col.append(c)
        #         return c
        #     else:
        #         return rand_col()

        # for path, params in paths_params:
            # st = path[0]
            # ed = path[-1]
            # col = QColor(144+i,238,144+i)
            # col = rand_col()

            # for item in scen_items:
            #     if type(item) == QGraph_Vertex and item.id in path and not item.id == self.start_node and not item.id == self.end_node:
            #         item.setBrush(col)
            #         item.color = col
            #     elif type(item) == QGraph_Vertex and item.id == self.start_node:
            #         item.setBrush(QColor(240, 230, 140))
            #         item.color = QColor(240, 230, 140)
            #     elif type(item) == QGraph_Vertex and item.id == self.end_node:
            #         item.setBrush(QColor(240, 230, 140))
            #         item.color = QColor(240, 230, 140)

        for item in scen_items:
            if type(item) == QGraph_Edge:
                for e in edges:
                    if item.source_vert.id in e and item.dest_vert.id in e:
                        item.setPen(QPen(Qt.black, 4))

        paths_and_rating = {}
        op_path = []
        op_rtng = 0
        for path, params in paths_params:
            # print('t =', params[0], 'l =', params[1])
            fl = FuzzyLogic()
            fl.time, fl.load_coef = params[0], params[1]
            fl.fuzzy_controler()
            paths_and_rating[str(path)] = fl.rating
            if fl.rating > op_rtng:
                op_rtng = fl.rating
                op_path = path
        # print(paths_and_rating)
        text_paths = 'Found Disjoint Paths'.center(73, ' ')+'-'*69+'\n\n'
        i = 0

        for path, params in paths_params:
            time, load = params
            # print(t, l)
            i += 1
            n = '-'*69
            R_in_paths = ['R'+str(i)+'-> ' for i in path]
            p = ''
            for k in R_in_paths:
                p = p + k
            f = f'Path_{i} :\n  {p[:-3]}\n\nRating_{i} = {round(paths_and_rating[str(path)], 2)}%\nTime_Metric_{i} = {time}\nLoad_{i} = {load}\n\n'
            text_paths += f
            # text_paths += 'Path'+ str(i) + ': ' + str(path) +'  ' + 'R' + '_' + str(i) + ' = ' + str(round(paths_and_rating[str(path)], 2)) + '% ' + '\n' + \
            #               ' M' + str(i) + ' = ' + str(time) + '\n' + ' D' + str(i) + ' = ' + str(load) + '\n\n'
        op = ['R'+str(i)+'-> ' for i in op_path]
        m = ''
        for t in op:
            m = m + t
        text_paths += '-'*69 + '\n' + 'Optimal path:  ' + m[:-3]

        self.disjoint_paths_field.setText(text_paths)

if __name__ == '__main__':
    my_app = QApplication(sys.argv)
    my_app.setApplicationName("Network Project")
    main_window = Main()
    wid_get = QtWidgets.QStackedWidget()
    wid_get.addWidget(main_window)
    wid_get.show()
    my_app.exec_()

