import pandas as pd
import tensorflow as tf
import numpy as np


MASK_VALUE = -1.  # The masking value cannot be zero.


def load_testset_w_difficulty(fn, fn2, valid_features, batch_size=32, shuffle=True, level='nivel 1 prueba de transición'):
    # To load a dataset to be tested, valid features will be selected, while others discarded.
    # Valid features should come from the dataset the network was trained with.

    df=pd.read_csv(fn)
    df2 = pd.read_csv(fn2)

    if "pregunta_id" not in df.columns:
        raise KeyError(f"The column 'pregunta_id' was not found on {fn}")
    if "correcta" not in df.columns:
        raise KeyError(f"The column 'correct' was not found on {fn}")
    if "usuario_id" not in df.columns:
        raise KeyError(f"The column 'usuario_id' was not found on {fn}")
    if "pregunta_id" not in df2.columns:
        raise KeyError(f"The column 'pregunta_id' was not found on {fn2}")
    if "pregunta_id" not in df2.columns:
        raise KeyError(f"The column 'clasificacion_tipo' was not found on {fn2}")
    if "pregunta_id" not in df2.columns:
        raise KeyError(f"The column 'clasificacion' was not found on {fn2}")

    # Right or wrong must be coded as 1s or 0s respectively
    if not (df['correcta'].isin([0, 1])).all():
        raise KeyError(f"The values of the column 'correcta' must be 0 or 1.")

    # Build dictionaries with the labels from the 3 cathegories and dificulty
    n1_dict = {}
    n2_dict = {}
    n3_dict = {}
    diff_dict = {}

    for i in range(len(df2['pregunta_id'])):
        if df2['clasificacion_tipo'][i] == 'nivel 1 prueba de transición':
            n1_dict.update({df2['pregunta_id'][i]: df2['clasificacion'][i]})
        elif df2['clasificacion_tipo'][i] == 'nivel 2 prueba de transición':
            n2_dict.update({df2['pregunta_id'][i]: df2['clasificacion'][i]})
        elif df2['clasificacion_tipo'][i] == 'nivel 3 prueba de transición':
            n3_dict.update({df2['pregunta_id'][i]: df2['clasificacion'][i]})
        elif df2['clasificacion_tipo'][i] == 'dificultad':
            diff_dict.update({df2['pregunta_id'][i]: df2['clasificacion'][i]})


    # Apply the dictionaties to have a straightforward way to a question's category
    df['nivel 1 prueba de transición'] = df['pregunta_id'].map(n1_dict)
    df['nivel 2 prueba de transición'] = df['pregunta_id'].map(n2_dict)
    df['nivel 3 prueba de transición'] = df['pregunta_id'].map(n3_dict)
    df['dificultad'] = df['pregunta_id'].map(diff_dict)

    # Turn the Nans into something more useful
    df['dificultad'] = df['dificultad'].fillna('Muy Fácil')

    #Checking what is compatible with the network
    data_to_keep=[]
    for i in range(df.shape[0]):
        data_to_keep.append( any(x in valid_features for x in [df[level][i]]) )
    df['colador'] = data_to_keep

    # Filters out what had a False in df['colador']
    df=df.loc[df['colador']]

    # Remove users with a single answer
    df = df.groupby('usuario_id').filter(lambda q: len(q) > 1).copy()

    # Remove rows with mising values in the column to clasify
    df.dropna(axis=0, how="any", subset=[level], inplace=True)

    # Prepare the labels from valid_features
    features_dict={}
    for i in range(len(valid_features)):
        features_dict.update({i:valid_features[i]})
    inv_features = {v: k for k, v in features_dict.items()}
    #  Enumerate skill id, instead of factorize, the order from valid_features is used
    df['pregunta'] = df[level].map(inv_features)


    # Lets clasify the nivel 1's to color the nodes according to this
    df['colors'], color_label = pd.factorize(df['nivel 1 prueba de transición'], sort=True)
    hierarchy_key = df.groupby('pregunta').apply( lambda r: r['colors'].values[0] )

    # Replaces the dificultad keywords with numbres, for easier handling
    df['dificultad'] = df['dificultad'].replace(to_replace=['Muy Fácil', 'Fácil', 'Media', 'Difícil', 'Muy Difícil'], value=[0, 1, 2, 3, 4])

    # Cross skill id with answer to form a synthetic feature
    df['pregunta+correcta'] = df['pregunta'] * 2 + df['correcta']

    # Convert to a sequence per user id and shift features 1 timestep
    seq = df.groupby('usuario_id').apply(
        lambda r: (
            r['pregunta+correcta'].values[:-1],
            r['dificultad'].values[:-1],
            r['pregunta'].values[1:],
            r['correcta'].values[1:],
        )
    )
    nb_users = len(seq)

    # Get Tensorflow Dataset
    dataset = tf.data.Dataset.from_generator(
        generator=lambda: seq,
        output_types=(tf.int32, tf.int32, tf.int32, tf.float32)  #
    )

    # If u want to shuffle, let's shuffle
    if shuffle:
        dataset = dataset.shuffle(buffer_size=nb_users)

    # Prepares things to build inputs and outputs
    skill_depth = df['pregunta'].max() + 1
    features_depth = int(df['pregunta+correcta'].max() + 1)

    # Building inputs and targets (Input_[n by 2*skill_depth+dificulties] , output_[n by skill_depth] )
    dataset = dataset.map(
        lambda feat, diff, skill, label: (
            #Input
            tf.concat(values=[tf.one_hot(feat, depth=features_depth),
                              tf.one_hot(diff, depth=5)
                             ],
                      axis=-1),
            # Output and targets
            tf.concat(values=[tf.one_hot(skill, depth=skill_depth),
                              tf.expand_dims(label, -1)
                             ],
                      axis=-1)
            )
        )

    # Pad sequences per batch
    dataset = dataset.padded_batch(
        batch_size=batch_size,
        padding_values=(MASK_VALUE, MASK_VALUE),
        padded_shapes=([None, None], [None, None]),
        drop_remainder=True
    )

    #length = nb_users // batch_size

    return dataset, nb_users, skill_depth, hierarchy_key

def load_dataset_w_difficulty_filter(fn, fn2, asignatura="Todas" , batch_size=32, shuffle=True, level='nivel 1 prueba de transición'):
    df = pd.read_csv(fn) #should load a dataset simmilar to [demo_dkt] Respuestas.csv
    df2 = pd.read_csv(fn2)#should load [DATOS_DKT] Clasificaciones.csv

    # Just checking that all the needed things are there
    if "pregunta_id" not in df.columns:
        raise KeyError(f"The column 'pregunta_id' was not found on {fn}")
    if "correcta" not in df.columns:
        raise KeyError(f"The column 'correct' was not found on {fn}")
    if "usuario_id" not in df.columns:
        raise KeyError(f"The column 'usuario_id' was not found on {fn}")
    if "pregunta_id" not in df2.columns:
        raise KeyError(f"The column 'pregunta_id' was not found on {fn2}")
    if "pregunta_id" not in df2.columns:
        raise KeyError(f"The column 'clasificacion_tipo' was not found on {fn2}")
    if "pregunta_id" not in df2.columns:
        raise KeyError(f"The column 'clasificacion' was not found on {fn2}")

    # Right or wrong must be coded as 1s or 0s respectively
    if not (df['correcta'].isin([0, 1])).all():
        raise KeyError(f"The values of the column 'correcta' must be 0 or 1.")

    # Build dictionaries with the labels from the 3 categories and the dificulty
    n1_dict = {}
    n2_dict = {}
    n3_dict = {}
    diff_dict = {}

    for i in range( len( df2['pregunta_id'] )  ):
        if df2['clasificacion_tipo'][i]=='nivel 1 prueba de transición' :
            n1_dict.update({df2['pregunta_id'][i] : df2['clasificacion'][i]})
        elif df2['clasificacion_tipo'][i]=='nivel 2 prueba de transición' :
            n2_dict.update({df2['pregunta_id'][i] : df2['clasificacion'][i]})
        elif df2['clasificacion_tipo'][i]=='nivel 3 prueba de transición' :
            n3_dict.update({df2['pregunta_id'][i] : df2['clasificacion'][i]})
        elif df2['clasificacion_tipo'][i]=='dificultad' :
            diff_dict.update({df2['pregunta_id'][i] : df2['clasificacion'][i]})

    # Apply the dictionaties to have a straightforward way to a question's category
    df['nivel 1 prueba de transición'] = df['pregunta_id'].map(n1_dict)
    df['nivel 2 prueba de transición'] = df['pregunta_id'].map(n2_dict)
    df['nivel 3 prueba de transición'] = df['pregunta_id'].map(n3_dict)
    df['dificultad'] = df['pregunta_id'].map(diff_dict)

    # Remove rows with mising values in the column to clasify
    df.dropna( axis=0, how="any",subset=[level], inplace=True)
    # turn the Nans into something more useful
    df['dificultad'] = df['dificultad'].fillna('Muy Fácil')

    if asignatura!="Todas":
        if asignatura=="Matemáticas":
            print('Matemáticas')
            valid_features=['Geometría','Números','Probabilidades y estadísticas', 'Álgebra y funciones']
        elif asignatura=="Ciencias":
            print('Ciencias')
            valid_features = ['Biología', 'Física', 'Química']
        elif asignatura == "Lenguaje":
            print('Lenguaje')
            valid_features = ['Provenientes de los medios masivos de comunicación', 'Literarios: Narraciones', 'Literarios: Obras dramáticas', 'No Literarios: Con finalidad expositiva y argumentativa']
        elif asignatura=="Historia":
            print('Historia')
            valid_features = ['Economía y sociedad', 'Formación ciudadana', 'Historia en perspectiva: Mundo, América y Chile.']
        # Checking what is compatible with the network
        data_to_keep = []
        for i in df.index:
            data_to_keep.append(any(x in valid_features for x in [df['nivel 1 prueba de transición'][i]]))
        df['colador'] = data_to_keep

        # Filters out what had a False in df['colador']
        df = df.loc[df['colador']]




    # Remove users with a single answer
    df = df.groupby('usuario_id').filter(lambda q: len(q) > 1).copy()

    # Enumerate skill id
    df['pregunta'], label_key = pd.factorize(df[level], sort=True)
    # Replaces the dificultad keywords with numbres, for easier handling
    df['dificultad'] = df['dificultad'].replace(to_replace=['Muy Fácil','Fácil','Media','Difícil','Muy Difícil'],value=[0,1,2,3,4])

    # Lets clasify the "nivel 1"s to color the nodes according to this
    df['colors'],color_label = pd.factorize(df['nivel 1 prueba de transición'], sort=True)
    hierarchy_key = df.groupby('pregunta').apply( lambda r: r['colors'].values[0] ) # this finaly relates the current clasification to the flavors from Nivel 1
    # Cross skill id with answer to form a synthetic feature
    df['pregunta+correcta'] = df['pregunta'] * 2 + df['correcta']


    # Convert to a sequence per user id and shift features 1 timestep
    seq = df.groupby('usuario_id').apply(
        lambda r: (
            # the inputs loose their last item
            r['pregunta+correcta'].values[:-1],
            r['dificultad'].values[:-1],
            # the outputs loose the first one
            r['pregunta'].values[1:],
            r['correcta'].values[1:],
        )
    )
    nb_users = len(seq)

    # Get Tensorflow Dataset
    dataset = tf.data.Dataset.from_generator(
        generator=lambda: seq,
        output_types=(tf.int32,tf.int32,tf.int32, tf.float32)
    )

    #if u want to shuffle, let's shuffle
    if shuffle:
        dataset = dataset.shuffle(buffer_size=nb_users)

    #prepares things to build inputs and outputs
    skill_depth = df['pregunta'].max() + 1
    features_depth = int(df['pregunta+correcta'].max() + 1)

    # Building inputs and targets (Input_[n by 2*skill_depth+dificulties] , output_[n by skill_depth] )
    dataset = dataset.map(
        lambda feat, diff, skill, label: (
            # Inputs
            tf.concat(values=[tf.one_hot(feat, depth=features_depth),
                              tf.one_hot(diff, depth=5)
                              ],
                      axis=-1),
            # Outputs and targets
            tf.concat(values=[tf.one_hot(skill, depth=skill_depth),
                              tf.expand_dims(label, -1)
                             ],
                      axis=-1)
            )
        )

    # Pad sequences per batch
    dataset = dataset.padded_batch(
        batch_size=batch_size,
        padding_values=(MASK_VALUE, MASK_VALUE),
        padded_shapes=([None, None], [None, None]),
        drop_remainder=True
    )

    length = nb_users // batch_size

    return dataset, length, skill_depth, label_key, hierarchy_key


def load_dataset_w_difficulty(fn, fn2, batch_size=32, shuffle=True, level='nivel 1 prueba de transición'):
    df = pd.read_csv(fn) #should load a dataset simmilar to [demo_dkt] Respuestas.csv
    df2 = pd.read_csv(fn2)#should load [DATOS_DKT] Clasificaciones.csv

    # Just checking that all the needed things are there
    if "pregunta_id" not in df.columns:
        raise KeyError(f"The column 'pregunta_id' was not found on {fn}")
    if "correcta" not in df.columns:
        raise KeyError(f"The column 'correct' was not found on {fn}")
    if "usuario_id" not in df.columns:
        raise KeyError(f"The column 'usuario_id' was not found on {fn}")
    if "pregunta_id" not in df2.columns:
        raise KeyError(f"The column 'pregunta_id' was not found on {fn2}")
    if "pregunta_id" not in df2.columns:
        raise KeyError(f"The column 'clasificacion_tipo' was not found on {fn2}")
    if "pregunta_id" not in df2.columns:
        raise KeyError(f"The column 'clasificacion' was not found on {fn2}")

    # Right or wrong must be coded as 1s or 0s respectively
    if not (df['correcta'].isin([0, 1])).all():
        raise KeyError(f"The values of the column 'correcta' must be 0 or 1.")

    # Build dictionaries with the labels from the 3 categories and the dificulty
    n1_dict = {}
    n2_dict = {}
    n3_dict = {}
    diff_dict = {}

    for i in range( len( df2['pregunta_id'] )  ):
        if df2['clasificacion_tipo'][i]=='nivel 1 prueba de transición' :
            n1_dict.update({df2['pregunta_id'][i] : df2['clasificacion'][i]})
        elif df2['clasificacion_tipo'][i]=='nivel 2 prueba de transición' :
            n2_dict.update({df2['pregunta_id'][i] : df2['clasificacion'][i]})
        elif df2['clasificacion_tipo'][i]=='nivel 3 prueba de transición' :
            n3_dict.update({df2['pregunta_id'][i] : df2['clasificacion'][i]})
        elif df2['clasificacion_tipo'][i]=='dificultad' :
            diff_dict.update({df2['pregunta_id'][i] : df2['clasificacion'][i]})

    # Apply the dictionaties to have a straightforward way to a question's category
    df['nivel 1 prueba de transición'] = df['pregunta_id'].map(n1_dict)
    df['nivel 2 prueba de transición'] = df['pregunta_id'].map(n2_dict)
    df['nivel 3 prueba de transición'] = df['pregunta_id'].map(n3_dict)
    df['dificultad'] = df['pregunta_id'].map(diff_dict)

    # Remove rows with mising values in the column to clasify
    df.dropna( axis=0, how="any",subset=[level], inplace=True)
    # turn the Nans into something more useful
    df['dificultad'] = df['dificultad'].fillna('Muy Fácil')

    # Remove users with a single answer
    df = df.groupby('usuario_id').filter(lambda q: len(q) > 1).copy()

    # Enumerate skill id
    df['pregunta'], label_key = pd.factorize(df[level], sort=True)
    # Replaces the dificultad keywords with numbres, for easier handling
    df['dificultad'] = df['dificultad'].replace(to_replace=['Muy Fácil','Fácil','Media','Difícil','Muy Difícil'],value=[0,1,2,3,4])

    # Lets clasify the "nivel 1"s to color the nodes according to this
    df['colors'],color_label = pd.factorize(df['nivel 1 prueba de transición'], sort=True)
    hierarchy_key = df.groupby('pregunta').apply( lambda r: r['colors'].values[0] ) # this finaly relates the current clasification to the flavors from Nivel 1
    # Cross skill id with answer to form a synthetic feature
    df['pregunta+correcta'] = df['pregunta'] * 2 + df['correcta']


    # Convert to a sequence per user id and shift features 1 timestep
    seq = df.groupby('usuario_id').apply(
        lambda r: (
            # the inputs loose their last item
            r['pregunta+correcta'].values[:-1],
            r['dificultad'].values[:-1],
            # the outputs loose the first one
            r['pregunta'].values[1:],
            r['correcta'].values[1:],
        )
    )
    nb_users = len(seq)

    # Get Tensorflow Dataset
    dataset = tf.data.Dataset.from_generator(
        generator=lambda: seq,
        output_types=(tf.int32,tf.int32,tf.int32, tf.float32)
    )

    #if u want to shuffle, let's shuffle
    if shuffle:
        dataset = dataset.shuffle(buffer_size=nb_users)

    #prepares things to build inputs and outputs
    skill_depth = df['pregunta'].max() + 1
    features_depth = int(df['pregunta+correcta'].max() + 1)

    # Building inputs and targets (Input_[n by 2*skill_depth+dificulties] , output_[n by skill_depth] )
    dataset = dataset.map(
        lambda feat, diff, skill, label: (
            # Inputs
            tf.concat(values=[tf.one_hot(feat, depth=features_depth),
                              tf.one_hot(diff, depth=5)
                              ],
                      axis=-1),
            # Outputs and targets
            tf.concat(values=[tf.one_hot(skill, depth=skill_depth),
                              tf.expand_dims(label, -1)
                             ],
                      axis=-1)
            )
        )

    # Pad sequences per batch
    dataset = dataset.padded_batch(
        batch_size=batch_size,
        padding_values=(MASK_VALUE, MASK_VALUE),
        padded_shapes=([None, None], [None, None]),
        drop_remainder=True
    )

    length = nb_users // batch_size

    return dataset, length, skill_depth, label_key, hierarchy_key


def load_dataset(fn, batch_size=32, shuffle=True):
    df = pd.read_csv(fn)

    if "skill_id" not in df.columns:
        raise KeyError(f"The column 'skill_id' was not found on {fn}")
    if "correct" not in df.columns:
        raise KeyError(f"The column 'correct' was not found on {fn}")
    if "user_id" not in df.columns:
        raise KeyError(f"The column 'user_id' was not found on {fn}")

    if not (df['correct'].isin([0, 1])).all():
        raise KeyError(f"The values of the column 'correct' must be 0 or 1.")

    # Step 1.1 - Remove questions without skill
    df.dropna(subset=['skill_id'], inplace=True)

    # Step 1.2 - Remove users with a single answer
    df = df.groupby('user_id').filter(lambda q: len(q) > 1).copy()

    # Step 2 - Enumerate skill id
    df['skill'], _ = pd.factorize(df['skill_id'], sort=True)

    # Step 3 - Cross skill id with answer to form a synthetic feature
    df['skill_with_answer'] = df['skill'] * 2 + df['correct']

    # Step 4 - Convert to a sequence per user id and shift features 1 timestep
    seq = df.groupby('user_id').apply(
        lambda r: (
            r['skill_with_answer'].values[:-1],
            r['skill'].values[1:],
            r['correct'].values[1:],
        )
    )
    nb_users = len(seq)

    # Step 5 - Get Tensorflow Dataset
    dataset = tf.data.Dataset.from_generator(
        generator=lambda: seq,
        output_types=(tf.int32, tf.int32, tf.float32)
    )

    if shuffle:
        dataset = dataset.shuffle(buffer_size=nb_users)

    # Step 6 - Encode categorical features and merge skills with labels to compute target loss.
    # More info: https://github.com/tensorflow/tensorflow/issues/32142
    features_depth = df['skill_with_answer'].max() + 1
    skill_depth = df['skill'].max() + 1

    dataset = dataset.map(
        lambda feat, skill, label: (
            tf.one_hot(feat, depth=features_depth),
            tf.concat(
                values=[
                    tf.one_hot(skill, depth=skill_depth),
                    tf.expand_dims(label, -1)
                ],
                axis=-1
            )
        )
    )

    # Step 7 - Pad sequences per batch
    dataset = dataset.padded_batch(
        batch_size=batch_size,
        padding_values=(MASK_VALUE, MASK_VALUE),
        padded_shapes=([None, None], [None, None]),
        drop_remainder=True
    )

    length = nb_users // batch_size
    return dataset, length, features_depth, skill_depth


def split_dataset(dataset, total_size, test_fraction, val_fraction=None):
    #basically, just splits the dataset in train, validation and test according to the input fractions
    def split(dataset, split_size):
        split_set = dataset.take(split_size)
        dataset = dataset.skip(split_size)
        return dataset, split_set

    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between (0, 1)")

    if val_fraction is not None and not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between (0, 1)")

    test_size = np.ceil(test_fraction * total_size)
    train_size = total_size - test_size

    if test_size == 0 or train_size == 0:
        raise ValueError(
            "The train and test datasets must have at least 1 element. Reduce the split fraction or get more data.")

    train_set, test_set = split(dataset, test_size)

    val_set = None
    if val_fraction:
        val_size = np.ceil(train_size * val_fraction)
        train_set, val_set = split(train_set, val_size)

    return train_set, test_set, val_set


def get_target(y_true, y_pred):
    # Get skills and labels from y_true
    # I got rid of this function after i changed the inputs and target
    # maybe it wasn't the best idea, but that needs to be tested

    mask = 1. - tf.cast(tf.equal(y_true, MASK_VALUE), y_true.dtype)
    y_true = y_true * mask

    skills, y_true = tf.split(y_true, num_or_size_splits=[-1, 1], axis=-1)

    # Get predictions for each skill
    y_pred = tf.reduce_sum(y_pred * skills, axis=-1, keepdims=True)

    return y_true, y_pred

#-------| GRAVEYARD |--------#

'''def load_dataset_criolla_by_levels(fn, fn2, batch_size=32, shuffle=True, level='nivel 1 prueba de transición'):
    df = pd.read_csv(fn) #should load [demo_dkt] Respuestas.csv
    df2 = pd.read_csv(fn2)#should load [demo_dkt] Clasificaciones.csv

    #Just checking that all the needed things are there
    if "pregunta_id" not in df.columns:
        raise KeyError(f"The column 'pregunta_id' was not found on {fn}")
    if "correcta" not in df.columns:
        raise KeyError(f"The column 'correct' was not found on {fn}")
    if "usuario_id" not in df.columns:
        raise KeyError(f"The column 'usuario_id' was not found on {fn}")
    if "pregunta_id" not in df2.columns:
        raise KeyError(f"The column 'pregunta_id' was not found on {fn2}")
    if "pregunta_id" not in df2.columns:
        raise KeyError(f"The column 'clasificacion_tipo' was not found on {fn2}")
    if "pregunta_id" not in df2.columns:
        raise KeyError(f"The column 'clasificacion' was not found on {fn2}")

    # Right or wrong must be coded as 1s or 0s respectively
    if not (df['correcta'].isin([0, 1])).all():
        raise KeyError(f"The values of the column 'correcta' must be 0 or 1.")

    #build dictionaries with the labels from the 3 cathegories
    n1_dict = {}
    n2_dict = {}
    n3_dict = {}
    for i in range( len( df2['pregunta_id'] )  ):
        if df2['clasificacion_tipo'][i]=='nivel 1 prueba de transición' :
            n1_dict.update({df2['pregunta_id'][i] : df2['clasificacion'][i]})
        elif df2['clasificacion_tipo'][i]=='nivel 2 prueba de transición' :
            n2_dict.update({df2['pregunta_id'][i] : df2['clasificacion'][i]})
        elif df2['clasificacion_tipo'][i]=='nivel 3 prueba de transición' :
            n3_dict.update({df2['pregunta_id'][i] : df2['clasificacion'][i]})

    #Apply the dictionaties to have a straightforward way to a question's cathegory
    df['nivel 1 prueba de transición'] = df['pregunta_id'].map(n1_dict)
    df['nivel 2 prueba de transición'] = df['pregunta_id'].map(n2_dict)
    df['nivel 3 prueba de transición'] = df['pregunta_id'].map(n3_dict)

    # Step 1 - Remove users with a single answer
    df = df.groupby('usuario_id').filter(lambda q: len(q) > 1).copy()

    # Step 2 - Enumerate skill id
    df['pregunta'], label_key = pd.factorize(df[level], sort=True)
    # Lets clasify the nivel 1's to color the nodes according to this
    df['colors'],color_label = pd.factorize(df['nivel 1 prueba de transición'], sort=True)
    hierarchy_key = df.groupby('pregunta').apply( lambda r: r['colors'].values[0] ) # this finaly relates the current clasification to the flavors from Nivel 1
    # Step 3 - Cross skill id with answer to form a synthetic feature
    df['pregunta+correcta'] = df['pregunta'] * 2 + df['correcta']


    # Step 4 - Convert to a sequence per user id and shift features 1 timestep
    seq = df.groupby('usuario_id').apply(
        lambda r: (
            r['pregunta+correcta'].values[:-1],
            r['pregunta'].values[1:],
            r['correcta'].values[1:],
        )
    )
    nb_users = len(seq)

    # Step 5 - Get Tensorflow Dataset
    dataset = tf.data.Dataset.from_generator(
        generator=lambda: seq,
        output_types=(tf.int32,tf.int32, tf.float32)  #
    )

    #if u want to shuffle, let's shuffle
    if shuffle:
        dataset = dataset.shuffle(buffer_size=nb_users)

    #prepares things to build inputs and outputs
    skill_depth = df['pregunta'].max() + 1
    features_depth = int(df['pregunta+correcta'].max() + 1)

    #Building inputs and targets (Input_[n by 2*skill_depth] , output_[n by skill_depth] )
    #the first half of the input is a one_hot encoding of the question, the second half is a one_hot encoding if that question was answered right or not.
    dataset = dataset.map(
        lambda feat, skill, label: (
            tf.one_hot(feat, depth=features_depth),
            tf.concat(
                values=[
                    tf.one_hot(skill, depth=skill_depth),
                    tf.expand_dims(label, -1)
                ],
                axis=-1
            )
        )
    )

    # Step 7 - Pad sequences per batch
    dataset = dataset.padded_batch(
        batch_size=batch_size,
        padding_values=(MASK_VALUE, MASK_VALUE),
        padded_shapes=([None, None], [None, None]),
        drop_remainder=True
    )

    length = nb_users // batch_size

    return dataset, length, skill_depth, label_key, hierarchy_key'''


'''def load_testset_criolla(fn, fn2, valid_features, batch_size=32, shuffle=True, level='nivel 1 prueba de transición'):
    df=pd.read_csv(fn)
    df2 = pd.read_csv(fn2)

    if "pregunta_id" not in df.columns:
        raise KeyError(f"The column 'pregunta_id' was not found on {fn}")
    if "correcta" not in df.columns:
        raise KeyError(f"The column 'correct' was not found on {fn}")
    if "usuario_id" not in df.columns:
        raise KeyError(f"The column 'usuario_id' was not found on {fn}")
    if "pregunta_id" not in df2.columns:
        raise KeyError(f"The column 'pregunta_id' was not found on {fn2}")
    if "pregunta_id" not in df2.columns:
        raise KeyError(f"The column 'clasificacion_tipo' was not found on {fn2}")
    if "pregunta_id" not in df2.columns:
        raise KeyError(f"The column 'clasificacion' was not found on {fn2}")

    # Right or wrong must be coded as 1s or 0s respectively
    if not (df['correcta'].isin([0, 1])).all():
        raise KeyError(f"The values of the column 'correcta' must be 0 or 1.")

    n1_dict = {}
    n2_dict = {}
    n3_dict = {}
    data_to_keep = []
    for i in range( len( df2['pregunta_id'] )  ):
        if df2['clasificacion_tipo'][i]=='nivel 1 prueba de transición' :
            n1_dict.update({df2['pregunta_id'][i] : df2['clasificacion'][i]})
        elif df2['clasificacion_tipo'][i]=='nivel 2 prueba de transición' :
            n2_dict.update({df2['pregunta_id'][i] : df2['clasificacion'][i]})
        elif df2['clasificacion_tipo'][i]=='nivel 3 prueba de transición' :
            n3_dict.update({df2['pregunta_id'][i] : df2['clasificacion'][i]})


    # Apply the dictionaties to have a straightforward way to a question's cathegory
    df['nivel 1 prueba de transición'] = df['pregunta_id'].map(n1_dict)
    df['nivel 2 prueba de transición'] = df['pregunta_id'].map(n2_dict)
    df['nivel 3 prueba de transición'] = df['pregunta_id'].map(n3_dict)

    #df['colador'] = df['nivel 2 prueba de transición'] in valid_features#any(x in paid[j] for x in d)
    data_to_keep=[]
    for i in range(df.shape[0]):
        local_search=[]
        data_to_keep.append( any(x in valid_features for x in [df[level][i]]) )

        #for j in valid_features:
        #    local_search.append( df['nivel 2 prueba de transición'][i]==j )
        #data_to_keep.append( any(local_search) )
        #if i%1000 == 0:
        #    print(i)

    df['colador'] = data_to_keep
    df=df.loc[df['colador']]

    # Step 1 - Remove users with a single answer
    df = df.groupby('usuario_id').filter(lambda q: len(q) > 1).copy()

    features_dict={}
    for i in range(len(valid_features)):
        features_dict.update({i:valid_features[i]})
    inv_features = {v: k for k, v in features_dict.items()}
    # Step 2 - Enumerate skill id
    #df['pregunta'], label_key = pd.factorize(df[level], sort=True)
    df['pregunta'] = df[level].map(inv_features)
    #df.replace({"col_name": dict})

    # Lets clasify the nivel 1's to color the nodes according to this
    df['colors'], color_label = pd.factorize(df['nivel 1 prueba de transición'], sort=True)
    hierarchy_key = df.groupby('pregunta').apply(
        lambda r: r['colors'].values[0])  # this finaly relates the current clasification to the flavors from Nivel 1
    # Step 3 - Cross skill id with answer to form a synthetic feature
    df['pregunta+correcta'] = df['pregunta'] * 2 + df['correcta']

    # Step 4 - Convert to a sequence per user id and shift features 1 timestep
    seq = df.groupby('usuario_id').apply(
        lambda r: (
            r['pregunta+correcta'].values[:-1],
            r['pregunta'].values[1:],
            r['correcta'].values[1:],
        )
    )
    nb_users = len(seq)

    # Step 5 - Get Tensorflow Dataset
    dataset = tf.data.Dataset.from_generator(
        generator=lambda: seq,
        output_types=(tf.int32, tf.int32, tf.float32)  #
    )

    # if u want to shuffle, let's shuffle
    if shuffle:
        dataset = dataset.shuffle(buffer_size=nb_users)

    # prepares things to build inputs and outputs
    skill_depth = df['pregunta'].max() + 1
    features_depth = int(df['pregunta+correcta'].max() + 1)

    # Building inputs and targets (Input_[n by 2*skill_depth] , output_[n by skill_depth] )
    # the first half of the input is a one_hot encoding of the question, the second half is a one_hot encoding if that question was answered right or not.
    dataset = dataset.map(
        lambda feat, skill, label: (
            tf.one_hot(feat, depth=features_depth),
            tf.concat(
                values=[
                    tf.one_hot(skill, depth=skill_depth),
                    tf.expand_dims(label, -1)
                ],
                axis=-1
            )
        )
    )

    # Step 7 - Pad sequences per batch
    dataset = dataset.padded_batch(
        batch_size=batch_size,
        padding_values=(MASK_VALUE, MASK_VALUE),
        padded_shapes=([None, None], [None, None]),
        drop_remainder=True
    )

    #length = nb_users // batch_size

    return dataset, nb_users, skill_depth, hierarchy_key'''
