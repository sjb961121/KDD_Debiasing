# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/Val.ipynb (unless otherwise specified).

__all__ = ['evaluate_each_phase', 'evaluate', 'create_underexpose_test_qtime_with_answer', 'main']

# Cell

# coding=utf-8
# from __future__ import division
# from __future__ import print_function

import datetime
import json
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd
from loguru import logger
import os

# Cell

# the higher scores, the better performance
def evaluate_each_phase(predictions: dict, answers: dict):
    """

    predictions: dict
        key是user_id, value是list [pred_item_id_1,pred_item_id_2...]
    answers: dict
        key是user_id, value是(item_id, item_degree)
    """
    list_item_degress = []
    for user_id in answers:
        item_id, item_degree = answers[user_id]  # item_degree 应该是内置的表示新颖性的指标，越小，越新颖
        list_item_degress.append(item_degree)
    list_item_degress.sort()
    median_item_degree = list_item_degress[len(list_item_degress) // 2]

    num_cases_full = 0.0
    ndcg_50_full = 0.0
    ndcg_50_half = 0.0
    num_cases_half = 0.0
    hitrate_50_full = 0.0
    hitrate_50_half = 0.0
    for user_id in answers:
        item_id, item_degree = answers[user_id]
        rank = 0
        # 把正确的item_id和 预测值 依次 比对，
        while rank < 50 and predictions[user_id][rank] != item_id:
            rank += 1
        num_cases_full += 1.0
        if rank < 50:
            ndcg_50_full += 1.0 / np.log2(rank + 2.0)
            hitrate_50_full += 1.0
        if item_degree <= median_item_degree:
            num_cases_half += 1.0
            if rank < 50:
                ndcg_50_half += 1.0 / np.log2(rank + 2.0)
                hitrate_50_half += 1.0
    ndcg_50_full /= num_cases_full
    hitrate_50_full /= num_cases_full
    ndcg_50_half /= num_cases_half
    hitrate_50_half /= num_cases_half
    return np.array([hitrate_50_full, ndcg_50_full,hitrate_50_half, ndcg_50_half], dtype=np.float32)

# Cell

# submit_fname is the path to the file submitted by the participants.
# debias_track_answer.csv is the standard answer, which is not released.
def evaluate(submit_fname,
             answer_fname='../data/debias_track_answer.csv', current_time=None):
    schedule_in_unix_time = [
        0,  # ........ 1970-01-01 08:00:00 (T=0)
        1586534399,  # 2020-04-10 23:59:59 (T=1)
        1587139199,  # 2020-04-17 23:59:59 (T=2)
        1587743999,  # 2020-04-24 23:59:59 (T=3)
        1588348799,  # 2020-05-01 23:59:59 (T=4)
        1588953599,  # 2020-05-08 23:59:59 (T=5)
        1589558399,  # 2020-05-15 23:59:59 (T=6)
        1590163199,  # 2020-05-22 23:59:59 (T=7)
        1590767999,  # 2020-05-29 23:59:59 (T=8)
        1591372799  # .2020-06-05 23:59:59 (T=9)
    ]
    assert len(schedule_in_unix_time) == 10
    for i in range(1, len(schedule_in_unix_time) - 1):
        # 604800 == one week
        assert schedule_in_unix_time[i] + 604800 == schedule_in_unix_time[i + 1]

    if current_time is None:
        current_time = int(time.time())
    print('current_time:', current_time)
    print('date_time:', datetime.datetime.fromtimestamp(current_time))
    current_phase = 6
    # while (current_phase < 9) and (
    #         current_time > schedule_in_unix_time[current_phase + 1]):
    #     current_phase += 1
    # print('current_phase:', current_phase)
#     current_phase = 4
    try:
        answers = [{} for _ in range(10)]
        with open(answer_fname, 'r') as fin:
            for line in fin:
                line = [int(x) for x in line.split(',')]
                phase_id, user_id, item_id, item_degree = line
                # assert user_id % 11 == phase_id
                # exactly one test case for each user_id
                answers[phase_id][user_id] = (item_id, item_degree)
    except Exception as _:
        print('server-side error: answer file incorrect')

    try:
        predictions = {}
        with open(submit_fname, 'r') as fin:
            for line in fin:
                line = line.strip()
                if line == '':
                    continue
                line = line.split(',')
                user_id = int(line[0])
                if user_id in predictions:
                    print('submitted duplicate user_ids')
                item_ids = [int(i) for i in line[1:]]
                if len(item_ids) != 50:
                    print('each row need have 50 items')
                if len(set(item_ids)) != 50:
                    return report_error(
                        stdout, 'each row need have 50 DISTINCT items')
                predictions[user_id] = item_ids
    except Exception as _:
        print('submission not in correct format')

    scores = np.zeros(4, dtype=np.float32)

    # The final winning teams will be decided based on phase T=7,8,9 only.
    # We thus fix the scores to 1.0 for phase 0,1,2,...,6 at the final stage.
    if current_phase >= 7:  # if at the final stage, i.e., T=7,8,9
        scores += 7.0  # then fix the scores to 1.0 for phase 0,1,2,...,6
    phase_beg = (7 if (current_phase >= 7) else 0)
    phase_end = current_phase +1
    r_dict = {}  # 每阶段成绩
    for phase_id in range(phase_beg, phase_end):
        for user_id in answers[phase_id]:
            if user_id not in predictions:
                print('user_id %d of phase %d not in submission' % (
                        user_id, phase_id))
        try:
            # We sum the scores from all the phases, instead of averaging them.
            rs = evaluate_each_phase(predictions, answers[phase_id])
            print(f'phase_id: {phase_id}, score: {rs}')
            r_dict[phase_id] = rs
            scores += rs
        except Exception as _:
            print('error occurred during evaluation')

#     print(score=float(scores[0]))
#     print(ndcg_50_full=float(scores[0]), ndcg_50_half=float(scores[1]),
#         hitrate_50_full=float(scores[2]), hitrate_50_half=float(scores[3]))
#     print(float(scores[1]))
#     print(float(scores[0]), float(scores[1]), float(scores[2]), float(scores[3]))
    return (float(scores[0]), float(scores[1]), float(scores[2]), float(scores[3])), r_dict

# Cell

# FYI. You can create a fake answer file for validation based on this. For example,
# you can mask the latest ONE click made by each user in underexpose_test_click-T.csv,
# and use those masked clicks to create your own validation set, i.e.,
# a fake underexpose_test_qtime_with_answer-T.csv for validation.
def _create_answer_file_for_evaluation(answer_fname='../data/debias_track_answer.csv'):
    train = '../data/underexpose_train/underexpose_train_click-%d.csv'
    test = '../data/underexpose_test/underexpose_test_click-%d.csv'

    # underexpose_test_qtime-T.csv contains only <user_id, item_id>
    # underexpose_test_qtime_with_answer-T.csv contains <user_id, item_id, time>
    answer = '../data/underexpose_test_qtime_with_answer-%d.csv'  # not released

    item_deg = defaultdict(lambda: 0)
    now_phase = 6
    with open(answer_fname, 'w') as fout:
        for phase_id in range(now_phase+1):
#             print(phase_id)
            with open(train % phase_id) as fin:
                for line in fin:
                    user_id, item_id, timestamp = line.split(',')
                    user_id, item_id, timestamp = (
                        int(user_id), int(item_id), float(timestamp))
                    item_deg[item_id] += 1
            with open(test % (phase_id)) as fin:
                for line in fin:
                    user_id, item_id, timestamp = line.split(',')
                    user_id, item_id, timestamp = (
                        int(user_id), int(item_id), float(timestamp))
                    item_deg[item_id] += 1
            with open(answer % phase_id) as fin:
                for line in fin:
                    user_id, item_id, timestamp = line.split(',')
                    user_id, item_id, timestamp = (
                        int(user_id), int(item_id), float(timestamp))
                    # assert user_id % 11 == phase_id
                    print(phase_id, user_id, item_id, item_deg[item_id],
                          sep=',', file=fout)

# Cell

def create_underexpose_test_qtime_with_answer(now_phase):
    test_path = '../data/underexpose_test'
    for c in range(now_phase + 1):
        logger.info(f'phase: {c}')
        ff = f'../data/underexpose_test_qtime_with_answer-{c}.csv'
        if os.path.exists(ff):
            logger.info(f'{ff} exists!')
            continue
        click_test1 = pd.read_csv(test_path + '/underexpose_test_click-{}.csv'.format(c), header=None,  names=['user_id', 'item_id', 'time'])
        click_test1.sort_values(['user_id', 'time']).drop_duplicates(subset=['user_id','item_id'],keep='last').reset_index(drop=True).to_csv(ff, index=None, header=None)

# Cell

def main(now_phase, submit_fname):
    create_underexpose_test_qtime_with_answer(now_phase)
    _create_answer_file_for_evaluation()
    r = evaluate(submit_fname)
    # print(r)

if __name__ == '__main__':
    main(6,'./underexpose_submit-2000-9-8.csv')