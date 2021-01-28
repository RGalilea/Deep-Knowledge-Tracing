import argparse
import matplotlib
import matplotlib.pyplot
import tensorflow as tf
import numpy as np
import networkx as nx
import pandas as pd
from deepkt import deepkt, data_util, metrics

def run(args):
    #Load the right Dataset, but really, just to get it's shape and labels
    _, _, nb_features, label_key, color_key = data_util.load_dataset_w_difficulty(fn=args.f,
                                                                                  fn2=args.classes,
                                                                                  batch_size=args.batch_size,
                                                                                  shuffle=True,
                                                                                  level=args.l)

    print("[----- LOADING MODEL ------]")
    model = deepkt.DKTModel(nb_features=nb_features,
                            extra_inputs=5,
                            hidden_units=args.hidden_units,
                            dropout_rate=args.dropout_rate)
    #Although we're not going to train, the model needs to compile
    model.compile( optimizer='adam',   metrics=[ metrics.BinaryAccuracy(), metrics.AUC(), metrics.Precision(), metrics.Recall() ])

    model.load_weights(args.w)
    print(model.summary())

    # Making the inputs to probe the network and calculate the Influences
    system_responce=np.zeros((nb_features, nb_features))
    for i in range(nb_features):
        probe_input = np.zeros((1, 1, 2 * nb_features + 5))
        probe_input[0][0][2*i+1] = 1
        probe_input[0][0][-1] = 1 # -1 : Muy Difícil, -2:Difícil, -3:Media, -4:Fácil, -5:Muy Fácil
        system_responce[i,:] = model(tf.convert_to_tensor(probe_input,dtype=tf.float32),training=False).numpy()
        model.reset_states()

    column_totals=system_responce.sum(axis=1)
    influence_matrix=np.zeros((nb_features,nb_features))

    for i in range(nb_features):
        influence_matrix[i,:]=np.divide(system_responce[i,:],column_totals)

    #Parameters for the graph
    fact0=1.5
    d_fact=0.01
    color_dictionary = {-1: '#000000',
                        0: '#1E8449',
                        1: '#CB4335',
                        2: '#F107F5',
                        3: '#481A87',
                        4: '#3498DB',
                        5: '#F5077E',
                        6: '#AAD617',
                        7: '#F1C40F',
                        8: '#D47510',
                        9: '#43CED1',
                       10: '#0E19CA',
                       11: '#B89C00',
                       12: '#42C914',
                       13: '#7F98B0'}

    #Calculating the matrix that represents the Graph
    auxiliar=np.ones((nb_features,nb_features))-np.eye(nb_features,nb_features)
    norm_factor=1/np.max( (np.multiply(influence_matrix,auxiliar)) )
    print("Normalize Factor: ",norm_factor)
    graph_matrix=np.floor( fact0*norm_factor*(np.multiply(influence_matrix,auxiliar)) )#np.floor( fact0*(np.multiply(influence_matrix,auxiliar)) )
    influence_graph=nx.from_numpy_matrix( graph_matrix,parallel_edges=False,create_using=nx.MultiDiGraph() )
    print(args.graph)
    if args.graph:
        # Getting just the most relevant elements, depends on the slider
        nonzero_indexes = np.nonzero( graph_matrix.sum(axis=1) + graph_matrix.sum(axis=0) )
        reduced_graph = influence_graph.subgraph( nonzero_indexes[0].tolist() )
        positions = nx.spring_layout( reduced_graph )

        # Drawing
        fig, ax = matplotlib.pyplot.subplots()

        #' ''
        ax.legend(handles=[matplotlib.patches.Patch(color='#1E8449'),
                           matplotlib.patches.Patch(color='#CB4335'),
                           matplotlib.patches.Patch(color='#F107F5'),
                           matplotlib.patches.Patch(color='#481A87'),
                           matplotlib.patches.Patch(color='#3498DB'),
                           matplotlib.patches.Patch(color='#F5077E'),
                           matplotlib.patches.Patch(color='#AAD617'),
                           matplotlib.patches.Patch(color='#F1C40F'),
                           matplotlib.patches.Patch(color='#D47510'),
                           matplotlib.patches.Patch(color='#43CED1'),
                           matplotlib.patches.Patch(color='#0E19CA'),
                           matplotlib.patches.Patch(color='#B89C00'),
                           matplotlib.patches.Patch(color='#42C914'),
                           matplotlib.patches.Patch(color='#7F98B0')],
                  labels=['Biología', 'Economía y Sociedad', 'Formación Ciudadana', 'Física', 'Geometría',
                          'Historia en perspectiva: Mundo, América y Chile', 'Literarios: Narraciones',
                          'Literarios: Obras Dramáticas', 'No Literarios: con finalidad expositiva y argumentativa',
                          'Números', 'Probabilidades y Estadística',
                          'Proveniente de los medios masivos de Comunicación',
                          'Química', 'Álgebra y Funciones'],
                  loc='upper right')
        #' ''

        #color_series = list()


        color_series=color_key[positions.keys()].replace(color_dictionary)
        #'''
        plotter = nx.draw_networkx_nodes( reduced_graph, pos=positions,  node_size=50, node_color=color_series)#pd.Series(index=nonzero_indexes[0], data=tuple(color_series)) )
        nx.draw_networkx_labels( reduced_graph,
                                 pos=positions,
                                 font_size=8,
                                 labels=pd.Series(index=nonzero_indexes[0],
                                                  data=tuple(label_key.values[nonzero_indexes[0]]))
                                 )
        nx.draw_networkx_edges( reduced_graph,
                                pos=positions,
                                arrowstyle='->')
        #'''

        '''
        plotter = nx.draw_networkx(reduced_graph, pos=positions,
                                   node_color=color_series,
                                   node_size=50 + 10 * graph_matrix[nonzero_indexes[0].tolist()].sum(axis=1),
                                   arrows=True,
                                   arrowstyle='->',
                                   font_size=6,
                                   labels=pd.Series(index=nonzero_indexes[0],data=tuple(label_key.values[nonzero_indexes[0]])) )
        '''

        ax.margins(x=0)
        axfact = matplotlib.pyplot.axes([0.125, 0.05, 0.775, 0.03], facecolor='lightgoldenrodyellow')
        slider_fact = matplotlib.widgets.Slider(axfact, 'A', 0.01, 2.0, valinit=fact0, valstep=d_fact)

        # Redraws and re-calculates when the slider changes
        def update(val):
            factor = val
            ax.clear()
            graph_matrix = np.floor(factor * norm_factor*(np.multiply(influence_matrix, auxiliar)))
            graph = nx.from_numpy_matrix(graph_matrix, parallel_edges=False, create_using=nx.MultiDiGraph())
            nonzero_indexes = np.nonzero(graph_matrix.sum(axis=1) + graph_matrix.sum(axis=0))
            reduced_graph = influence_graph.subgraph(nonzero_indexes[0].tolist())
            positions = nx.spring_layout(reduced_graph)

            ax.legend(handles=[matplotlib.patches.Patch(color='#1E8449'),
                               matplotlib.patches.Patch(color='#CB4335'),
                               matplotlib.patches.Patch(color='#F107F5'),
                               matplotlib.patches.Patch(color='#481A87'),
                               matplotlib.patches.Patch(color='#3498DB'),
                               matplotlib.patches.Patch(color='#F5077E'),
                               matplotlib.patches.Patch(color='#AAD617'),
                               matplotlib.patches.Patch(color='#F1C40F'),
                               matplotlib.patches.Patch(color='#D47510'),
                               matplotlib.patches.Patch(color='#43CED1'),
                               matplotlib.patches.Patch(color='#0E19CA'),
                               matplotlib.patches.Patch(color='#B89C00'),
                               matplotlib.patches.Patch(color='#42C914'),
                               matplotlib.patches.Patch(color='#7F98B0')],
                      labels=['Biología', 'Economía y Sociedad', 'Formación Ciudadana', 'Física', 'Geometría',
                              'Historia en perspectiva: Mundo, América y Chile', 'Literarios: Narraciones',
                              'Literarios: Obras Dramáticas', 'No Literarios: con finalidad expositiva y argumentativa',
                              'Números', 'Probabilidades y Estadística',
                              'Proveniente de los medios masivos de Comunicación', 'Química', 'Álgebra y Funciones'],
                      loc='upper right')


            plotter = nx.draw_networkx(reduced_graph,
                                       pos=positions,
                                       node_color=color_key[positions.keys()].replace(color_dictionary),
                                       node_size=50 + 10 * graph_matrix[nonzero_indexes[0].tolist()].sum(axis=1),
                                       arrows=True,
                                       arrowstyle='->',
                                       font_size=6,
                                       labels=pd.Series(index=nonzero_indexes[0],
                                                        data=tuple(label_key.values[nonzero_indexes[0]])
                                                        )
                                       )



            fig.canvas.draw_idle()

        # Last instructions to actually draw and show the plot
        slider_fact.on_changed(update)
        matplotlib.pyplot.sca(ax)
        matplotlib.pyplot.show()
    else:
        matplotlib.pyplot.matshow(graph_matrix)  # prediction_result[0].numpy() )
        matplotlib.pyplot.colorbar()
        # imshow(prediction_result[0].numpy(), interpolation='nearest')
        # grid(True)
        matplotlib.pyplot.show()




def parse_args():
    parser = argparse.ArgumentParser(prog="Use Trained DeepKT")

    parser.add_argument("-graph",
                        type=bool,
                        default=False,
                        help="Show a Influence Graph(True) or a Matrix(False)")

    # Just relevan for the [demo_dkt], nivel 2 and 3 are more insightful, nivel 1 has too few classes (4)
    parser.add_argument("-l",
                        type=str,
                        default='nivel 1 prueba de transición',
                        help="nivel 1, 2 ó 3 prueba de transición ")

    parser.add_argument("-f",
                        type=str,
                        default="data/ASSISTments_skill_builder_data.csv",#"data/[demo_dkt] Respuestas.csv",#
                        help="the path to the data")

    parser.add_argument("-classes",
                        type=str,
                        default="data/[DATOS_DKT] Clasificaciones.csv",
                        help="the path to the data")

    parser.add_argument("-v",
                        type=int,
                        default=1,
                        help="verbosity mode [0, 1, 2].")

    parser.add_argument("-w",
                        type=str,
                        default="weights/bestmodel",
                        help="model weights file.")

    parser.add_argument("--log_dir",
                        type=str,
                        default="logs/",
                        help="log dir.")

    model_group = parser.add_argument_group(title="Model arguments.")
    model_group.add_argument("--dropout_rate",
                             type=float,
                             default=.3,
                             help="fraction of the units to drop.")

    model_group.add_argument("--hidden_units",
                             type=int,
                             default=100,
                             help="number of units of the LSTM layer.")

    #the training is already done
    train_group = parser.add_argument_group(title="Training arguments.")
    train_group.add_argument("--batch_size",
                             type=int,
                             default=1,
                             help="number of elements to combine in a single batch.")

    train_group.add_argument("--epochs",
                             type=int,
                             default=50,
                             help="number of epochs to train.")

    train_group.add_argument("--test_split",
                             type=float,
                             default=.2,
                             help="fraction of data to be used for testing (0, 1).")

    train_group.add_argument("--val_split",
                             type=float,
                             default=.2,
                             help="fraction of data to be used for validation (0, 1).")

    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())