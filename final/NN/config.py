
class Config():
    def __init__(self):
        self.batch_size = 128
        self.init_lr = 1e-4
        self.grad_clip = 0.1
        self.num_epochs = 30
        self.challenge = 'unseen'
        self.select_threshold = -3
        self.num_workers = 8
        self.scheduler = 'exp'
        self.gender_num = 4  # female, male, other, not specified
        self.interests_num = 96
        self.course_num = 100 if self.challenge == 'seen' else 50
        self.model_path = './model.pt'
        self.output_path = './output.csv'
        self.train_file = f'./data/train_preprocessed_{self.challenge}.csv'
        self.users_file = './data/users_preprocessed.csv'
        self.valid_file = f'./data/val_{self.challenge}_preprocessed.csv'
        self.test_file = f'./data/test_{self.challenge}.csv'
        self.id2cid_file = f'./data/id2cid_{self.challenge}.csv'
        self.seed = 42
        self.device = 'cuda'
        self.benchmark = 0.1

        if self.challenge == 'seen':
            self.in_features = self.gender_num + self.interests_num + self.course_num
        else:
            self.in_features = self.gender_num + self.interests_num

        assert self.challenge in ['seen', 'unseen']
