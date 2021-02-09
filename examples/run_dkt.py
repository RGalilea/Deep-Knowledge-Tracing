import argparse
import tensorflow as tf
from deepkt import deepkt, data_util, metrics
import time


def custom_loss(l_r,l_w1,l_w2):
    def true_loss(y_true, y_pred):
        #print(holder)
        y_true2, y_pred2 = data_util.get_target(y_true[:, :-1, :], y_pred[:, 1:, :])#is this shit right??
        y_true2 = tf.concat([tf.zeros([1,1,1],tf.float32),y_true2],axis=-2)
        y_pred2 = tf.concat([tf.zeros([1,1, 1], tf.float32),y_pred2], axis=-2)

        n_batch,n_inter,n_outputs = y_true.shape

        waviness_norm_1 = tf.reduce_sum(tf.abs(y_pred[:, 1:, :] - y_pred[:, :-1, :]))/(n_inter*n_outputs )

        waviness_norm_2 = tf.reduce_sum(tf.square(y_pred[:, 1:, :] - y_pred[:, :-1, :]))/(n_inter*n_outputs )


        y_true, y_pred = data_util.get_target(y_true, y_pred)

        def get_loss_value(a,b,c,d,e,f):
            return tf.keras.losses.binary_crossentropy(a,b) + l_r*tf.keras.losses.binary_crossentropy(c, d) +l_w1*e +l_w2*f#(y_true, y_pred)
        return get_loss_value(y_true, y_pred,y_true2, y_pred2,waviness_norm_1,waviness_norm_2)

    return true_loss


def run(args):
    start = time.time()
    print("[----- LOADING DATASET  ------]")
    #dataset, length, nb_features,_,_  = data_util.load_dataset_w_difficulty(fn=args.f, fn2=args.classes,
    #                                                                          batch_size=args.batch_size,
    #                                                                          level=args.l,
    #                                                                          shuffle=True)



    dataset, length, nb_features,_,_  = data_util.load_dataset_w_difficulty_filter(fn=args.f, fn2=args.classes, asignatura="Matemáticas",
                                                                                   batch_size=args.batch_size, shuffle=True, level=args.l)


    print("[----- DIVIDING DATASET  ------]")
    train_set, test_set, val_set = data_util.split_dataset(dataset=dataset,
                                                           total_size=length,
                                                           test_fraction=args.test_split,
                                                           val_fraction=args.val_split)

    print("[----- COMPILING  ------]")
    model = deepkt.DKTModel(nb_features=nb_features,
                            hidden_units=args.hidden_units,
                            extra_inputs=5,#for the 5 different dificulties
                            dropout_rate=args.dropout_rate)



    model.compile(optimizer='adam',
                  loss_func= custom_loss(0.1,0.03,0.3),# i need to fix this value asignation
                  metrics=[metrics.BinaryAccuracy(),
                                             metrics.AUC(),
                                             metrics.Precision(),
                                             metrics.Recall()])
    if args.re_train:
        print("\n[-- Loading Weights --]")
        model.load_weights(args.w)

    print(model.summary())
    print("\n[-- COMPILING DONE  --]")

    print("\n[----- TRAINING ------]")
    model.fit(
        dataset=train_set,
        epochs=args.epochs,
        verbose=args.v,
        validation_data=val_set,
        callbacks=[
            tf.keras.callbacks.CSVLogger(f"{args.log_dir}/train.log"),
            tf.keras.callbacks.ModelCheckpoint(args.w,
                                               save_best_only=True,
                                               save_weights_only=True),
            tf.keras.callbacks.TensorBoard(log_dir=args.log_dir)
        ])
    print("\n[--- TRAINING DONE ---]")

    print("[----- TESTING  ------]")
    model.load_weights(args.w)
    model.evaluate(dataset=test_set, verbose=args.v)
    print("\n[--- TESTING DONE  ---]")
    end = time.time()
    print("\nElapsed Time: ",end - start)



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

    # True will load weights and train them with the provided data, mind that the network arquitectures match
    parser.add_argument("-re_train",
                        type=bool,
                        default=False,
                        help="load weights to continue training True/False")

    parser.add_argument("-f",
                        type=str,
                        default="data/ASSISTments_skill_builder_data.csv",
                        help="the path to the data")

    parser.add_argument("-classes",
                        type=str,
                        default="data/[DATOS_DKT] Clasificaciones.csv",
                        help="the path to the data's info file")

    parser.add_argument("-v",
                        type=int,
                        default=1,
                        help="verbosity mode [0, 1, 2].")

    # Folder to save the weights
    parser.add_argument("-w",
                        type=str,
                        default="weights/bestmodel",
                        help="model weights file.")

    parser.add_argument("--log_dir",
                        type=str,
                        default="logs",
                        help="log dir.")

    model_group = parser.add_argument_group(title="Model arguments.")
    model_group.add_argument("--dropout_rate",
                             type=float,
                             default=.275,
                             help="fraction of the units to drop.")

    model_group.add_argument("--hidden_units",
                             type=int,
                             default=100,
                             help="number of units of the LSTM layer.")

    train_group = parser.add_argument_group(title="Training arguments.")
    train_group.add_argument("--batch_size",
                             type=int,
                             default=1,
                             help="number of elements to combine in a single batch.")

    train_group.add_argument("--epochs",
                             type=int,
                             default=30,
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


#-------| GRAVEYARD |--------#


    '''    
    # This argument will be used to decide which methos to use, so i can keep the same code working with it's original Database and the [demo_dkt] one
    parser.add_argument("-criolla",
                        type=bool,
                        default=True,
                        help="Is this the database from Puntaje?")
    '''