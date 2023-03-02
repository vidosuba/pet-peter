from seqeval.metrics import accuracy_score
from seqeval.metrics import classification_report
from seqeval.metrics import f1_score

from evaluate_ner import evaluate_ner
import jsonlines


def evaluate(y_true, y_pred):
    y_true = [[f'B-{tag}' if tag != "O" else tag for tag in s] for s in y_true]
    y_pred = [[f'B-{tag}' if tag != "O" else tag for tag in s] for s in y_pred]
    # y_true = [['O', 'O', 'B-MISC', 'B-MISC', 'B-MISC', 'B-PER', 'O'], ['B-PER', 'B-PER', 'O']]
    # y_pred = [['O', 'O', 'B-MISC', 'B-MISC', 'B-MISC', 'B-MISC', 'O'], ['B-PER', 'B-PER', 'O']]
    print(f1_score(y_true, y_pred))
    print(classification_report(y_true, y_pred, digits=4))


def prepare_pet(test_path, predictions_path):
    with jsonlines.open(test_path, 'r') as f:
        test = list(f)

    with jsonlines.open(predictions_path, 'r') as f:
        predictions = list(f)

    tags_true = [line['tags'] for line in test]

    idxs = set(map(lambda x: x['idx'], predictions))
    tags_pred = [[y['label'] for y in predictions if y['idx'] == x] for x in idxs]

    return tags_true, tags_pred


def main():
    #test_path = 'data/ner/test-all.jsonl'
    test_path = 'data/ner/test.jsonl'
    #predictions_path = 'output/ner_04/final/p0-i0/predictions.jsonl'
    #predictions_path = 'output/nerr_04_50/final/p0-i0/predictions.jsonl'
    predictions_path = 'output/nerr_06/predictions-10.jsonl'
    true, pred = prepare_pet(test_path, predictions_path)
    #evaluate_ner(true, pred)
    evaluate(true, pred)


main()
