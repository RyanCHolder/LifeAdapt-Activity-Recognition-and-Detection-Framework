import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

"""Calculates performance metrics for activity detection based off of activity recognition models.

    Expects input of predicted and true labels at each time stamp as arrays of integer numbers of equal length.
    The code expects an npz dictionary with labels 'y_pred' for prediction and 'y_true' for the ground truth labels.
    Specify the directory of this file in the label_dir variable below

    mAP score or mean average precision score calculates the average precision at different IoU thresholds, then takes the mean of these averages
    IoU is a measure of the overlap between the predicted windows of time for a certain activity and the true activity windows"""

label_dir = "./output/labels.npz"

def calculate_iou(true_intervals, pred_intervals):
    """
    Calculate Intersection over Union (IoU) between ground truth and predicted activity intervals.

    Args:
        true_intervals: List of tuples representing ground truth intervals for an activity (start, end).
        pred_intervals: List of tuples representing predicted intervals for the same activity (start, end).

    Returns:
        iou: Intersection over Union score.
    """
    # If either ground truth or prediction is empty, IoU is 0
    if not true_intervals or not pred_intervals:
        return 0.0

    # Calculate the union and intersection of all intervals
    union_duration = 0
    intersection_duration = 0

    # Iterate over true and predicted intervals
    for true_start, true_end in true_intervals:
        for pred_start, pred_end in pred_intervals:
            # Calculate intersection (overlap)
            overlap_start = max(true_start, pred_start)
            overlap_end = min(true_end, pred_end)
            overlap = max(0, overlap_end - overlap_start)  # Ensure non-negative overlap
            intersection_duration += overlap

            # Calculate union
            union_start = min(true_start, pred_start)
            union_end = max(true_end, pred_end)
            union_duration += (union_end - union_start)

    # Avoid division by zero
    if union_duration == 0:
        return 0.0

    # Calculate IoU
    iou = intersection_duration / union_duration
    return iou


def get_activity_intervals(labels):
    """
    Converts a list of labels into intervals of continuous activities.

    Args:
        labels: List of activity labels.

    Returns:
        intervals_dict: Dictionary with activity labels as keys and lists of (start, end) intervals.
    """
    intervals_dict = {}
    current_label = labels[0]
    start_idx = 0

    for i in range(1, len(labels)):
        if labels[i] != current_label:
            # Create an interval for the previous label
            end_idx = i-1
            if current_label not in intervals_dict:
                intervals_dict[current_label] = []
            intervals_dict[current_label].append((start_idx, end_idx))

            # Update the current label and start index
            current_label = labels[i]
            start_idx = i

    # Append the final interval
    end_idx = len(labels) - 1
    if current_label not in intervals_dict:
        intervals_dict[current_label] = []
    intervals_dict[current_label].append((start_idx, end_idx))

    return intervals_dict


def evaluate_metrics(y_true, y_pred, activity_classes):
    """
    Evaluate metrics such as Precision, Recall, F1-score, and mAP for activity detection.
    Args:
        y_true: List of true activity labels.
        y_pred: List of predicted activity labels.
        activity_classes: List of possible activity classes.
    Returns:
        Dictionary of evaluation results.
    """
    results = {}

    # Precision, Recall, and F1-score for each class
    precision = precision_score(y_true, y_pred, labels=activity_classes, average='macro')
    recall = recall_score(y_true, y_pred, labels=activity_classes, average='macro')
    f1 = f1_score(y_true, y_pred, labels=activity_classes, average='macro')
    accuracy = accuracy_score(y_true, y_pred)

    # Mean Average Precision (mAP) based on precision/recall using macro average
    mAP = (precision + recall) / 2

    results['precision'] = precision
    results['recall'] = recall
    results['f1'] = f1
    results['accuracy'] = accuracy
    results['mAP'] = mAP

    return results


def main():
    labels = np.load(label_dir)
    y_pred = labels['y_pred']
    y_true = labels['y_test']
    n = len(y_true)
    activity_classes = [0, 1, 2, 3, 4, 5]

    results = evaluate_metrics(y_true, y_pred, activity_classes)
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")

    true_intervals = get_activity_intervals(y_true)
    pred_intervals = get_activity_intervals(y_pred)

    # Calculate IoU for each activity class. Also compute "perfect" IoU for normalization.
    iou_scores = {}
    iou_pscores = {}
    overall_score = 0

    for activity in activity_classes:
        true_activity_intervals = true_intervals.get(activity, [])
        pred_activity_intervals = pred_intervals.get(activity, [])

        iou = calculate_iou(true_activity_intervals, pred_activity_intervals)
        iou_perfect = calculate_iou(true_activity_intervals, true_activity_intervals)
        activity_count = np.count_nonzero(y_true == activity)
        norm_iou = iou / iou_perfect

        iou_scores[activity] = iou
        iou_pscores[activity] = iou_perfect
        overall_score += norm_iou * activity_count
    overall_score /= len(y_true)

    # Display IoU scores for each activity
    for activity, iou in iou_scores.items():
        print(f"IoU for {activity}: {iou:.4f}")
    for activity, iou in iou_pscores.items():
        print(f"Perfect IoU for {activity}: {iou:.4f}")
    print(f"Overall score: {overall_score:.4f}")

if __name__ == '__main__':
    main()