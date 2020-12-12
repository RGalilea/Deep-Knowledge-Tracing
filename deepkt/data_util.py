import pandas as pd
import tensorflow as tf
import numpy as np


MASK_VALUE = -1.  # The masking value cannot be zero.

def load_dataset_criolla(fn, batch_size=32, shuffle=True):
    df = pd.read_csv(fn)
    if "pregunta_id" not in df.columns:
        raise KeyError(f"The column 'skill_id' was not found on {fn}")
    if "correcta" not in df.columns:
        raise KeyError(f"The column 'correct' was not found on {fn}")
    if "usuario_id" not in df.columns:
        raise KeyError(f"The column 'user_id' was not found on {fn}")

    if not (df['correcta'].isin([0, 1])).all():
        raise KeyError(f"The values of the column 'correcta' must be 0 or 1.")

    # Step 1 - Remove users with a single answer
    df = df.groupby('usuario_id').filter(lambda q: len(q) > 1).copy()

    # Step 2 - Enumerate skill id
    df['pregunta'], _ = pd.factorize(df['pregunta_id'], sort=True)
    #key = df.groupby('pregunta').apply(lambda r: r['skill_name'].values[0:1]) # there are no labels in this dataset

    # Step 4 - Convert to a sequence per user id and shift features 1 timestep
    seq = df.groupby('usuario_id').apply(
        lambda r: (  # r['skill_with_answer'].values[:-1],
            r['pregunta'].values[1:],
            r['correcta'].values[1:]
        )
    )
    nb_users = len(seq)

    # Step 5 - Get Tensorflow Dataset
    dataset = tf.data.Dataset.from_generator(
        generator=lambda: seq,
        output_types=(tf.int32, tf.float32)  # tf.int32,
    )

    # for value in dataset.take(3):
    # print('debug 0:')

    if shuffle:
        dataset = dataset.shuffle(buffer_size=nb_users)

    skill_depth = df['pregunta'].max() + 1
    lower_triangle_gen = lambda size: tf.linalg.LinearOperatorLowerTriangular((tf.ones(shape=(size, size)))).to_dense()

    dataset = dataset.map(  # (  #feat, #tf.one_hot(feat, depth=features_depth),
        lambda skill, label: (
            tf.concat(values=[tf.one_hot(skill, depth=skill_depth),
                              tf.math.multiply(tf.one_hot(skill, skill_depth),
                                               tf.tensordot(a=tf.expand_dims(label, 1), b=tf.ones((1, skill_depth)),
                                                            axes=1))],
                      axis=-1),
            tf.clip_by_value(tf.transpose(tf.tensordot(tf.transpose(tf.math.multiply(tf.one_hot(skill, skill_depth),
                                                                                     tf.tensordot(
                                                                                         a=tf.expand_dims(label, 1),
                                                                                         b=tf.ones((1, skill_depth)),
                                                                                         axes=1))),
                                                       lower_triangle_gen(tf.shape(label)[0]), axes=1)), 0, 1)
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
    return dataset, length, skill_depth


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
    #print(df)
    # Step 2 - Enumerate skill id
    df['skill'], _ = pd.factorize(df['skill_id'], sort=True)
    key=df.groupby('skill').apply(lambda r: r['skill_name'].values[0:1] )
    #key=df_clone[:]
    #df['problem'], key = pd.factorize(df['problem_id'], sort=True)
    #print('The key that relates the Ids is size:',key.size)
    #print(df)
    # Step 3 - Cross skill id with answer to form a synthetic feature
    #df['skill_with_answer'] = df['skill'] * 2 + df['correct'] #I've found this kinda weird
    #df['correct_answer'] = df['skill_id']/df['skill_id'] * df['correct']

    # Step 4 - Convert to a sequence per user id and shift features 1 timestep
    seq = df.groupby('user_id').apply(
        lambda r: (            #r['skill_with_answer'].values[:-1],
            r['skill'].values[1:],
            r['correct'].values[1:]
        )
    )
    nb_users = len(seq)

    # Step 5 - Get Tensorflow Dataset
    dataset = tf.data.Dataset.from_generator(
        generator=lambda: seq,
        output_types=( tf.int32, tf.float32)#tf.int32,
    )

    #for value in dataset.take(3):
    #print('debug 0:')

    if shuffle:
        dataset = dataset.shuffle(buffer_size=nb_users)

    # Step 6 - Encode categorical features and merge skills with labels to compute target loss.
    # More info: https://github.com/tensorflow/tensorflow/issues/32142
    #features_depth = df['skill_with_answer'].max() + 1
    skill_depth = df['skill'].max() + 1
    lower_triangle_gen = lambda size: tf.linalg.LinearOperatorLowerTriangular((tf.ones(shape=(size, size)))).to_dense()#[[float(int(j >= i)) for j in range(size)] for i in range(size)]
    #problem_depth = df['problem'].max() + 1
    #for value in dataset.take(3):
    #    print('debug 0:')
    dataset = dataset.map( #(  #feat, #tf.one_hot(feat, depth=features_depth),
        lambda skill, label: (
            tf.concat(values=[tf.one_hot(skill, depth=skill_depth),
                              tf.math.multiply(tf.one_hot(skill, skill_depth),tf.tensordot(a=tf.expand_dims(label,1),b=tf.ones((1,skill_depth)), axes=1))],
                              axis=-1),
            tf.clip_by_value(tf.transpose(tf.tensordot(tf.transpose(tf.math.multiply(tf.one_hot(skill, skill_depth),tf.tensordot(a=tf.expand_dims(label,1),b=tf.ones((1,skill_depth)), axes=1))),lower_triangle_gen(tf.shape(label)[0]),axes=1)),0,1)
        )
    )
    #for value in dataset.take(1):
    #   print('debug 1:',value)
    # Step 7 - Pad sequences per batch
    dataset = dataset.padded_batch(
        batch_size=batch_size,
        padding_values=(MASK_VALUE, MASK_VALUE),
        padded_shapes=([None, None],[None, None]),
        drop_remainder=True
    ) #I probably have to change this
    #print('debug 2')
    #for value3 in dataset.take(3):
    #    print('debug 3:',value3)

    length = nb_users // batch_size
    return dataset, length, skill_depth, key


def split_dataset(dataset, total_size, test_fraction, val_fraction=None):
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
    '''
    mask = 1. - tf.cast(tf.equal(y_true, MASK_VALUE), y_true.dtype)
    y_true = y_true * mask

    skills, y_true = tf.split(y_true, num_or_size_splits=[-1, 1], axis=-1)

    # Get predictions for each skill
    y_pred = tf.reduce_sum(y_pred * skills, axis=-1, keepdims=True)'''

    return y_true, y_pred
