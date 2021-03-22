import PySimpleGUI as sg
import argparse
import tensorflow as tf
import numpy as np
#from pylab import * #??
#import pandas as pd
from deepkt import deepkt#, metrics#, data_util

def input_vector(clase,correct,length=2*15+5,dificulty='Muy Fácil'):
    vector=np.zeros((1,1,length))
    diff_dict={'Muy Fácil':-5,'Fácil':-4,'Media':-3,'Difícil':-2,'Muy Difícil':-1}

    # Turn to 1s the right Stuff
    vector[0][0][diff_dict[dificulty]] = 1
    vector[0][0][2*clase+correct] = 1
    return tf.convert_to_tensor(vector,dtype=tf.float32)

def update_values(handle,model_output):
    for i in range(len(model_output)):
        handle['output'+str(i)].update('{:.1f}'.format( 100*model_output[i] )+'%')


def run(args):
    math_labels = ['Conjunto de los números enteros, racionales y reales',
                   'Ecuaciones de segundo grado',
                   'Ecuaciones e inecuaciones de primer grado',
                   'Expresiones algebraicas',
                   'Función cuadrática',
                   'Función lineal y afín',
                   'Geometría analítica en 2D', 'Medidas de posición',
                   'Medidas de tendencia central y rango',
                   'Potencias, raíces enésimas y logaritmos',
                   'Reglas de las probabilidades y probabilidad condicional',
                   'Representación de datos a través de tablas y gráficos',
                   'Semejanza, proporcionalidad y homotecia de figuras planas',
                   'Sistemas de ecuaciones lineales (2x2)',
                   'Transformaciones isométricas']

    science_labels = ['Herencia y evolución',
                      'Organismo y ambiente',
                      'Organización, estructura y actividad celular',
                      'Procesos y funciones biológicas',
                      'Química orgánica',
                      'Reacciones químicas y estequiometría',
                      'Estructura atómica',
                      'Mecánica',
                      'Energía',
                      'Electricidad y magnetismo',
                      'Ondas']

    language_labels = ['Comprensión de lectura (textos narrativos)',
                       'Vocabulario (textos narrativos)',
                       'Comprensión de lectura (obras dramáticas)',
                       'Vocabulario (obras dramáticas)',
                       'Comprensión de lectura (medios masivos)',
                       'Vocabulario (medios masivos)',
                       'Vocabulario (textos argumentativos y expositivos)',
                       'Comprensión de lectura (textos argumentativos y expositivos)']

    history_labels = ['Chile en el contexto de la Guerra Fría: transformaciones estructurales, polarización política.',
                      'Configuración del territorio chileno y sus dinámicas geográficas en el siglo XIX.',
                      'Crisis, totalitarismo y guerra en la primera mitad del siglo XX.',
                      'De un mundo bipolar a un mundo globalizado.',
                      'Dictadura militar, transición política y consenso en torno a la democracia en Chile.',
                      'El desafío de consolidar el orden republicano y la idea de nación: Chile en el siglo XIX.',
                      'El orden liberal y las transformaciones políticas y sociales a fines del siglo XIX y comienzos del siglo XX en Chile.',
                      'Estado de derecho, acceso a la justicia y garantías ciudadanas.',
                      'Estado nación y sociedad burguesa en Europa y América en el siglo XIX.',
                      'Funcionamiento del sistema económico.',
                      'Participación ciudadana y su importancia en el funcionamiento del sistema político.']

    labels_to_use = []
    if "Matemáticas" in args.Asignatura:
        labels_to_use += math_labels
    if "Ciencias" in args.Asignatura:
        labels_to_use += science_labels
    if "Lenguaje" in args.Asignatura:
        labels_to_use += language_labels
    if "Historia" in args.Asignatura:
        labels_to_use += history_labels


    n_classes=len(labels_to_use)

    model = deepkt.DKTModel(nb_features=n_classes,
                            extra_inputs=5,
                            hidden_units=args.hidden_units
                            ,stateful=True
                            )
    # Although we're not going to train, the model needs to compile
    model.compile(optimizer='adam', loss_func= lambda a,b: a+b,#totally pointless here
                  metrics=None)#[metrics.BinaryAccuracy(), metrics.AUC(), metrics.Precision(), metrics.Recall()])

    model.load_weights(args.Asignatura+'/'+args.w)
    print(model.summary())

    #test1 = []
    #test2 = []
    #for i in range(4):
    #    test1.append([i])
    #    test2+=[i]






    outputs=[[sg.Text('OUT')]]
    incorrect_buttons = [[sg.Text("")]]
    correct_buttons = [[sg.Text("")]]
    class_labels = [[sg.Text('Temas:')]]

    for i in range(n_classes):
        outputs.append([sg.Text(' 0.0%',key='output'+str(i), pad=(0,6))])
        incorrect_buttons.append( [sg.Button("MF",key='I_MF_'+str(i),button_color="black on red"),sg.Button("F",key='I_F_'+str(i),button_color="black on red"),sg.Button("M",key='I_M_'+str(i),button_color="black on red"),sg.Button("D",key='I_D_'+str(i),button_color="black on red"),sg.Button("MD",key='I_MD_'+str(i),button_color="black on red")] )
        correct_buttons.append([sg.Button("MF", key='C_MF_' + str(i),button_color="black on green"), sg.Button("F", key='C_F_' + str(i),button_color="black on green"),sg.Button("M", key='C_M_' + str(i),button_color="black on green"), sg.Button("D", key='C_D_' + str(i),button_color="black on green"),sg.Button("MD", key='C_MD_' + str(i),button_color="black on green")])
        class_labels.append( [sg.Text(labels_to_use[i], pad=(0,6))] )


    layout =[[sg.Column(class_labels),sg.Column(incorrect_buttons),sg.Column(outputs),sg.Column(correct_buttons),sg.Button('Reset')]]

    # Create the window
    window = sg.Window("Demo", layout)

    # Create an event loop
    while True:
        event, values = window.read()
        # End program if user closes window or
        # presses the OK button
        if  event == sg.WIN_CLOSED:
            break
        if '_MD_' in event:
            #print('Comparation works '+ str(len(event)))
            input=[]
            if len(event)>6:
                if 'I' in event:
                    input = input_vector(int(event[-2:]), 0, 2 * n_classes + 5, 'Muy Difícil')
                if 'C' in event:
                    input = input_vector(int(event[-2:]), 1, 2 * n_classes + 5, 'Muy Difícil')
            else:
                if 'I' in event:
                    input = input_vector(int(event[-1]), 0, 2 * n_classes + 5, 'Muy Difícil')
                if 'C' in event:
                    input = input_vector(int(event[-1]), 1, 2 * n_classes + 5, 'Muy Difícil')

            output = model(input, training=False).numpy()
            update_values(window, output[0][0])
        if '_D_' in event:
            #print('Comparation works '+ str(len(event)))
            input=[]
            if len(event)>6:
                if 'I' in event:
                    input = input_vector(int(event[-2:]), 0, 2 * n_classes + 5, 'Difícil')
                if 'C' in event:
                    input = input_vector(int(event[-2:]), 1, 2 * n_classes + 5, 'Difícil')
            else:
                if 'I' in event:
                    input = input_vector(int(event[-1]), 0, 2 * n_classes + 5, 'Difícil')
                if 'C' in event:
                    input = input_vector(int(event[-1]), 1, 2 * n_classes + 5, 'Difícil')

            output = model(input, training=False).numpy()
            update_values(window, output[0][0])
        if '_M_' in event:
            #print('Comparation works '+ str(len(event)))
            input=[]
            if len(event)>6:
                if 'I' in event:
                    input = input_vector(int(event[-2:]), 0, 2 * n_classes + 5, 'Media')
                if 'C' in event:
                    input = input_vector(int(event[-2:]), 1, 2 * n_classes + 5, 'Media')
            else:
                if 'I' in event:
                    input = input_vector(int(event[-1]), 0, 2 * n_classes + 5, 'Media')
                if 'C' in event:
                    input = input_vector(int(event[-1]), 1, 2 * n_classes + 5, 'Media')
            output = model(input, training=False).numpy()
            update_values(window, output[0][0])
        if '_F_' in event:
            #print('Comparation works '+ str(len(event)))
            input=[]
            if len(event)>6:
                if 'I' in event:
                    input = input_vector(int(event[-2:]), 0, 2 * n_classes + 5, 'Fácil')
                if 'C' in event:
                    input = input_vector(int(event[-2:]), 1, 2 * n_classes + 5, 'Fácil')
            else:
                if 'I' in event:
                    input = input_vector(int(event[-1]), 0, 2 * n_classes + 5, 'Fácil')
                if 'C' in event:
                    input = input_vector(int(event[-1]), 1, 2 * n_classes + 5, 'Fácil')
            output = model(input, training=False).numpy()
            update_values(window, output[0][0])
        if '_MF_' in event:
            #print('Comparation works '+ str(len(event)))
            input=[]
            if len(event)>6:
                if 'I' in event:
                    input = input_vector(int(event[-2:]), 0, 2 * n_classes + 5, 'Muy Fácil')
                if 'C' in event:
                    input = input_vector(int(event[-2:]), 1, 2 * n_classes + 5, 'Muy Fácil')
            else:
                if 'I' in event:
                    input = input_vector(int(event[-1]), 0, 2 * n_classes + 5, 'Muy Fácil')
                if 'C' in event:
                    input = input_vector(int(event[-1]), 1, 2 * n_classes + 5, 'Muy Fácil')
            output = model(input, training=False).numpy()
            update_values(window, output[0][0])

        if event=='Reset':
            model.reset_states()
            for i in range(n_classes):
                window['output'+str(i)].update('{}'.format(' 0%'))



    window.close()



def parse_args():
    parser = argparse.ArgumentParser(prog="DeepKT Example")

    # Select the category to apply the network to, 1 es more general, while 3 is very particular
    parser.add_argument("-l",
                        type=str,
                        default='nivel 1 prueba de transición',
                        help="nivel 1, 2 ó 3 prueba de transición ")

    parser.add_argument("-Asignatura",
                        type=str,
                        default='Todas',
                        help="Matemáticas, Ciencias, Lenguaje, Historia o Todas")

    # Folder to read the weights from
    parser.add_argument("-w",
                        type=str,
                        default="weights/bestmodel",
                        help="model weights file.")


    parser.add_argument("-hidden_units",
                          type=int,
                          default=100,
                          help="number of units of the LSTM layer.")



    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())