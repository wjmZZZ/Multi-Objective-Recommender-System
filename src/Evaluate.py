import json
import logging
import pandas as pd

from beartype import beartype



def calc_recall(pred, gt):
    pred = pd.read_csv(pred)
    pred[['session', 'type']] = pred.session_type.str.split('_', expand=True)
    pred = pred.drop(columns=['session_type'])
    gt = pd.read_parquet(gt)

    score = 0
    recalls = {}
    weights = {'clicks': 0.10, 'carts': 0.30, 'orders': 0.60}

    for type_id, type_str in enumerate(['clicks','carts','orders']):
        pred_type = pred.loc[pred['type'] == type_str].copy()
        pred_type['labels'] = pred_type['labels'].apply(lambda x: x.split())
        pred_type['session'] = pred_type['session'].astype('int32')
        gt_type = gt.loc[gt['type'] == type_id]
        # gt['aid'] = gt['aid'].astype('object')
        gt_type['aid'] = gt_type['aid'].apply(lambda x: str(x).split())

        gt_type = gt_type.merge(pred_type, how='left', on=['session'])
        # 对于每一行，将该行的ground truth和labels[:20]取交集，并计算交集的长度，即为该样本的命中数
        gt_type['hits'] = gt_type.apply(lambda df: len(set(df.aid).intersection(set(df.labels[:20]))), axis=1)
        gt_type['gt_count'] = gt_type.aid.str.len().clip(0,20)
        type_recall = gt_type['hits'].sum() / gt_type['gt_count'].sum()
        
        recalls[type_str] = type_recall
        # score += weights[type_str] * type_recall

        # print(f'{type_str} recall = {type_recall}')

    result = 0
    for type_, recall in recalls.items():
        result += recall * weights[type_]

    recalls['total'] = result

    return recalls



@beartype
def prepare_predictions(predictions: list):
    prepared_predictions = dict()
    for prediction in predictions:
        sid_type, preds = prediction.strip().split(",")
        sid, event_type = sid_type.split("_")
        preds = [int(aid) for aid in preds.split(" ")] if preds != "" else []
        if not int(sid) in prepared_predictions:
            prepared_predictions[int(sid)] = dict()
        prepared_predictions[int(sid)][event_type] = preds
    return prepared_predictions


@beartype
def prepare_labels(labels: list):
    final_labels = dict()
    for label in labels:
        label = json.loads(label)
        final_labels[label["session"]] = {
                                        "clicks": label["labels"].get("clicks", None),
                                        "carts": set(label["labels"].get("carts", [])),
                                        "orders": set(label["labels"].get("orders", []))
                                        }
    return final_labels


@beartype
def evaluate_session(labels: dict, prediction: dict, k: int):
    if 'clicks' in labels and labels['clicks']:
        clicks_hit = float(labels['clicks'] in prediction['clicks'][:k])
    else:
        clicks_hit = None

    if 'carts' in labels and labels['carts']:
        cart_hits = len(
            set(prediction['carts'][:k]).intersection(labels['carts']))
    else:
        cart_hits = None

    if 'orders' in labels and labels['orders']:
        order_hits = len(
            set(prediction['orders'][:k]).intersection(labels['orders']))
    else:
        order_hits = None

    return {'clicks': clicks_hit, 'carts': cart_hits, 'orders': order_hits}


@beartype
def evaluate_sessions(labels: dict, predictions: dict, k: int):
    result = {}
    for session_id, session_labels in labels.items():
        if session_id in predictions:
            result[session_id] = evaluate_session(session_labels, predictions[session_id], k)
        else:
            result[session_id] = {k: 0. if v else None for k, v in session_labels.items()}
    return result


@beartype
def num_events(labels: dict, k: int):
    num_clicks = 0
    num_carts = 0
    num_orders = 0
    for event in labels.values():
        if 'clicks' in event and event['clicks']:
            num_clicks += 1
        if 'carts' in event and event['carts']:
            num_carts += min(len(event["carts"]), k)
        if 'orders' in event and event['orders']:
            num_orders += min(len(event["orders"]), k)
    return {'clicks': num_clicks, 'carts': num_carts, 'orders': num_orders}


@beartype
def recall_by_event_type(evalutated_events: dict, total_number_events: dict):
    clicks = 0
    carts = 0
    orders = 0
    for event in evalutated_events.values():
        if 'clicks' in event and event['clicks']:
            clicks += event['clicks']
        if 'carts' in event and event['carts']:
            carts += event['carts']
        if 'orders' in event and event['orders']:
            orders += event['orders']

    return {
        'clicks': clicks / total_number_events['clicks'],
        'carts': carts / total_number_events['carts'],
        'orders': orders / total_number_events['orders']
    }


@beartype
def weighted_recalls(recalls: dict, weights: dict):
    result = 0.0
    for event, recall in recalls.items():
        result += recall * weights[event]
    return result


@beartype
def get_scores(labels: dict,
               predictions: dict,
               k=20,
               weights={
                   'clicks': 0.10,
                   'carts': 0.30,
                   'orders': 0.60
}):
    '''
    Calculates the weighted recall for the given predictions and labels.
    Args:
        labels: dict of labels for each session
        predictions: dict of predictions for each session
        k: cutoff for the recall calculation
        weights: weights for the different event types
    Returns:
        recalls for each event type and the weighted recall
    '''
    total_number_events = num_events(labels, k)
    evaluated_events = evaluate_sessions(labels, predictions, k)
    recalls = recall_by_event_type(evaluated_events, total_number_events)
    recalls["total"] = weighted_recalls(recalls, weights)
    return recalls


@beartype
def evaluate(labels_path, predictions_path):
    with open(labels_path, "r") as f:
        # logging.info(f"Reading labels from {labels_path}")
        labels = f.readlines()
        labels = prepare_labels(labels)
        # logging.info(f"Read {len(labels)} labels")
        
    with open(predictions_path, "r") as f:
        # logging.info(f"Reading predictions from {predictions_path}")
        predictions = f.readlines()[1:]
        predictions = prepare_predictions(predictions)
    #     logging.info(f"Read {len(predictions)} predictions")
    # logging.info("Calculating scores")
    scores = get_scores(labels, predictions)
    return scores
