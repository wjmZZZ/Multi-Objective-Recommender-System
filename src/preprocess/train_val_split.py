import argparse
import json
import random
from copy import deepcopy
from pathlib import Path

# import cudf as pd
import pandas as pd
from beartype import beartype
from pandas.io.json._json import JsonReader
from tqdm import tqdm


class setEncoder(json.JSONEncoder):
    def default(self, obj):
        return list(obj)
    

def ground_truth(events: list):
    prev_labels = {"clicks": None, "carts": set(), "orders": set()}

    for event in reversed(events):
        event["labels"] = {}

        for label in ['clicks', 'carts', 'orders']:
            if prev_labels[label]:
                if label != 'clicks':
                    event["labels"][label] = prev_labels[label].copy()
                else:
                    event["labels"][label] = prev_labels[label]

        if event["type"] == "clicks":
            prev_labels['clicks'] = event["aid"]
        if event["type"] == "carts":
            prev_labels['carts'].add(event["aid"])
        elif event["type"] == "orders":
            prev_labels['orders'].add(event["aid"])

    return events[:-1]


@beartype
def split_events(events: list, split_idx=None):
    test_events = ground_truth(deepcopy(events))
    if not split_idx:
        split_idx = random.randint(1, len(test_events))
    test_events = test_events[:split_idx]
    labels = test_events[-1]['labels']
    for event in test_events:
        del event['labels']
    return test_events, labels


@beartype
def create_kaggle_testset(sessions: pd.DataFrame, sessions_output: Path, labels_output: Path):
    last_labels = []
    splitted_sessions = []

    for _, session in tqdm(sessions.iterrows(), desc="Creating trimmed testset", total=len(sessions)):
        session = session.to_dict()
        splitted_events, labels = split_events(session['events'])
        last_labels.append({'session': session['session'], 'labels': labels})
        splitted_sessions.append({'session': session['session'], 'events': splitted_events})

    with open(sessions_output, 'w') as f:
        for session in splitted_sessions:
            f.write(json.dumps(session) + '\n')

    with open(labels_output, 'w') as f:
        for label in last_labels:
            f.write(json.dumps(label, cls=setEncoder) + '\n')


@beartype
def trim_session(session: dict, max_ts: int) -> dict:
    session['events'] = [event for event in session['events'] if event['ts'] < max_ts]
    return session


@beartype
def get_max_ts(sessions_path: Path) -> int:
    max_ts = float('-inf')
    with open(sessions_path) as f:
        for line in tqdm(f, desc="Finding max timestamp"):
            session = json.loads(line)
            max_ts = max(max_ts, session['events'][-1]['ts'])
    return max_ts


@beartype
def filter_unknown_items(session_path: Path, known_items: set):
    filtered_sessions = []
    with open(session_path) as f:
        for line in tqdm(f, desc="Filtering unknown items"):
            session = json.loads(line)
            session['events'] = [event for event in session['events'] if event['aid'] in known_items]
            if len(session['events']) >= 2:
                filtered_sessions.append(session)
    with open(session_path, 'w') as f:
        for session in filtered_sessions:
            f.write(json.dumps(session) + '\n')


@beartype
def train_test_split(session_chunks: JsonReader, train_path: Path, test_path: Path, max_ts: int, test_days: int):
    split_millis = test_days * 24 * 60 * 60 * 1000
    split_ts = max_ts - split_millis
    train_items = set()
    Path(train_path).parent.mkdir(parents=True, exist_ok=True)
    train_file = open(train_path, "w")
    Path(test_path).parent.mkdir(parents=True, exist_ok=True)
    test_file = open(test_path, "w")
    for chunk in tqdm(session_chunks, desc="Splitting sessions"):
        for _, session in chunk.iterrows():
            session = session.to_dict()
            if session['events'][0]['ts'] > split_ts:
                test_file.write(json.dumps(session) + "\n")
            else:
                session = trim_session(session, split_ts)
                if len(session['events']) >= 2:
                    train_items.update([event['aid'] for event in session['events']])
                    train_file.write(json.dumps(session) + "\n")
    train_file.close()
    test_file.close()
    filter_unknown_items(test_path, train_items)


@beartype
def cv_split(train_set: Path, output_path: Path, days: int, seed: int, train_name: str, valid_name: str):
    random.seed(seed)
    max_ts = get_max_ts(train_set)

    session_chunks = pd.read_json(train_set, lines=True, chunksize=100000)
    train_file = output_path / f'{train_name}.jsonl'
    test_file_full = output_path / f'{valid_name}_full.jsonl'
    train_test_split(session_chunks, train_file, test_file_full, max_ts, days)

    test_sessions = pd.read_json(test_file_full, lines=True)
    test_sessions_file = output_path / f'{valid_name}.jsonl'
    test_labels_file = output_path / f'{valid_name}_labels.jsonl'
    create_kaggle_testset(test_sessions, test_sessions_file, test_labels_file)

    
    
    

