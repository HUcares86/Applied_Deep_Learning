import torch.utils.data as data
import torch
import os
import numpy as np
import pandas as pd
from config import Config
import random
import math

cfg = Config()

GENDER_NUM = cfg.gender_num
INTERESTS_NUM = cfg.interests_num
COURSES_NUM = cfg.course_num

pd.options.mode.chained_assignment = None  # default='warn'


class HaHowTrainData(data.Dataset):
    def __init__(self, path):
        data = pd.read_csv(path, usecols=['gender', 'interests', 'course_id'])

        for i in range(len(data)):
            data['course_id'][i] = [int(x) for x in data['course_id'][i].split(' ')
                                    ] if data['course_id'][i] != 'x' else []
            data['interests'][i] = [int(x) for x in data['interests'][i].split(' ')]

        self.data = data.to_numpy()

    def __getitem__(self, index):

        gender, interests, courses = self.data[index]
        while len(courses) == 0:
            index = random.randint(0, self.__len__()-1)
            gender, interests, courses = self.data[index]
        feature = torch.zeros(cfg.in_features)
        pivot = random.randint(0, len(courses)-1)
        # random.shuffle(courses) # time series data, shuffle get worse result
        former, latter = courses[:pivot], courses[pivot:]
        assert len(latter) != 0

        feature[gender] = 1
        for interest in interests:
            feature[GENDER_NUM + interest] = 1
        if len(former) == 1:
            feature[GENDER_NUM + INTERESTS_NUM + former[0]] = 1
        elif len(former) > 1:
            chosen_course = former[:]
            for i in range(int(random.uniform(0, 0.4)*len(chosen_course))):
                chosen_course.pop(random.randint(0, len(chosen_course)-1))
            # random.shuffle(chosen_course)
            # chosen_course = sorted(former[:random.randint(1, len(former)-1)], key=lambda x: former.index(x))
            for i, course in enumerate(chosen_course):
                feature[GENDER_NUM + INTERESTS_NUM + course] = 0.5 + 0.5*(i+1) / len(chosen_course)

        output = torch.zeros(COURSES_NUM)
        for course in latter:
            output[course] = 1

        return feature, output

    def __len__(self):
        return len(self.data)


class HaHowValidData(data.Dataset):
    def __init__(self, path):
        data = pd.read_csv(path, usecols=['user_id', 'gender', 'interests', 'course_id'])

        for i in range(len(data)):
            data['course_id'][i] = [int(x) for x in data['course_id'][i].split(' ')
                                    ] if data['course_id'][i] != 'x' else []
            data['interests'][i] = [int(x) for x in data['interests'][i].split(' ')]

        train_data = pd.read_csv(cfg.train_file, usecols=['user_id', 'course_id'])
        # for i in range(len(data)):
        #     train_data['course_id'][i] = [int(x) for x in train_data['course_id'][i].split(' ')]
        train_data.rename(columns={"course_id": "train_course_id"}, inplace=True)
        merged = pd.merge(data, train_data, on='user_id', how='left')
        merged = merged.fillna({'train_course_id': ''})
        for i in range(len(data)):
            merged['train_course_id'][i] = [int(x) for x in merged['train_course_id'][i].split(
                ' ')] if merged['train_course_id'][i] != 'x' else []
        merged.drop(columns=['user_id'], inplace=True)
        # print(merged)
        # exit()

        self.data = merged.to_numpy()

    def __getitem__(self, index):

        gender, interests, courses, train_courses = self.data[index]
        feature = torch.zeros(cfg.in_features)

        feature[gender] = 1
        for interest in interests:
            feature[GENDER_NUM + interest] = 1
        if train_courses:
            for i, course in enumerate(train_courses):
                feature[GENDER_NUM + INTERESTS_NUM + course] = 0.5 + 0.5*(i+1) / len(train_courses)

        output = torch.zeros(COURSES_NUM)
        for course in courses:
            if course == 'x':
                break
            output[course] = 1

        avoid_courses = torch.zeros(COURSES_NUM)
        for course in train_courses:
            if course == 'x':
                break
            avoid_courses[course] = 1

        return feature, output, avoid_courses

    def __len__(self):
        return len(self.data)


class HaHowTestData(data.Dataset):
    def __init__(self, path):
        data = pd.read_csv(path, usecols=['user_id'])

        train_data = pd.read_csv(cfg.train_file, usecols=['user_id', 'gender', 'interests', 'course_id'])
        # train_data.rename(columns={"course_id": "train_course_id"}, inplace=True)
        merged = pd.merge(data, train_data, on='user_id', how='left')
        merged = merged.fillna({'course_id': '', 'interests': ''})
        for i in range(len(data)):
            merged['interests'][i] = [int(x) for x in merged['interests'][i].split(' ')
                                      ] if merged['interests'][i] != '' else []

            merged['course_id'][i] = [int(x) for x in merged['course_id'][i].split(' ')
                                      ] if merged['course_id'][i] != 'x' else []
        # merged.drop(columns=['user_id'], inplace=True)
        # print(merged)
        # exit()

        self.data = merged.to_numpy()

    def __getitem__(self, index):

        user_ids, gender, interests, train_courses = self.data[index]
        # print(user_ids, gender, interests, train_courses)
        feature = torch.zeros(cfg.in_features)

        feature[gender] = 1
        for interest in interests:
            feature[GENDER_NUM + interest] = 1
        if train_courses:
            for i, course in enumerate(train_courses):
                feature[GENDER_NUM + INTERESTS_NUM + course] = 0.5 + 0.5*(i+1) / len(train_courses)

        avoid_courses = torch.zeros(COURSES_NUM)
        for course in train_courses:
            if course == 'x':
                break
            avoid_courses[course] = 1

        return feature, avoid_courses, user_ids

    def __len__(self):
        return len(self.data)
