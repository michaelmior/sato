import argparse
import collections
import csv
import itertools
import io
import json
import random
import sys

import numpy as np
import pandas as pd
import tqdm


def sufficient_samples(columns, sample_columns, min_count):
    column_counts = collections.defaultdict(lambda: 0)
    for s in sample_columns:
        for c in s:
            column_counts[c] += 1

    return all(column_counts[c] > min_count for c in columns)


def sample(df, max_columns, min_count):
    columns = df.columns.tolist()
    sample_columns = set()
    while not sufficient_samples(columns, sample_columns, min_count):
        new_sample = columns[:]
        random.shuffle(new_sample)
        while tuple(new_sample[:max_columns]) in sample_columns:
            random.shuffle(new_sample)
        sample_columns.add(tuple(new_sample[:max_columns]))
    return sample_columns


def sliding_window(iterable, n=2):
    iterables = itertools.tee(iterable, n)

    for iterable, num_skipped in list(zip(iterables, itertools.count())):
        for _ in range(num_skipped):
            next(iterable, None)

    return list(zip(*iterables))


def load_from_json():
    values = collections.defaultdict(lambda: [])
    for line in sys.stdin:
        # Load the value of the JSON object on this line
        # (which consists of single key-value pair)
        obj = next(iter(json.loads(line).values()))

        # Find keys which have a _label suffix to skip
        skip_keys = set()
        for key in obj.keys():
            if key.endswith('_label'):
                skip_keys.add(key[:-len('_label')])

        stripped_obj = {}
        for (key, value) in obj.items():
            # Convert the string "NULL" to None
            if value == 'NULL':
                value = None

            if key not in skip_keys:
                # Take the last part of the key after / or # and remove _label suffix
                key = key.split('/')[-1].split('_label')[0].split('#')[-1]

                if value is not None:
                    values[key].append(value)

    # Stack collected values together into a DataFrame
    # This loses row-wise association of values, but this does not affect the model
    objs = list(map(dict, map(lambda x: zip(*x), zip(itertools.repeat(values.keys()), zip(*(values.values()))))))
    return pd.DataFrame.from_dict(objs[:100])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--format', type=str, choices=('json', 'csv'), default='json')
    args = parser.parse_args()

    if args.format == 'json':
        df = load_from_json()
    elif args.format == 'csv':
        df = pd.read_csv(sys.stdin, nrows=100)

    sample_columns = sample(df, 6, 5)
    #sample_columns = sliding_window(df.columns.tolist(), 6)

    sys.stderr.write('Loading model...\n')
    import predict

    sys.stderr.write(f'Extracting features from {len(df.columns)}..\n')
    feature_dict, sherlock_features = predict.extract(df)

    sys.stderr.write(f'Running {len(sample_columns)} predictions...\n')
    predictions = []
    for cols in tqdm.tqdm(sample_columns):
        predictions.append(predict.evaluate(df, list(cols), feature_dict, sherlock_features))


    col_predictions = collections.defaultdict(lambda: [])
    for i, cols in enumerate(sample_columns):
        for j, col in enumerate(cols):
            col_predictions[col].append(predictions[i][j])

    for c, p in col_predictions.items():
        print(c, collections.Counter(p).most_common(1)[0][0])


if __name__ == '__main__':
    main()
