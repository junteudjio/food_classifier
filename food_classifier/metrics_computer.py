import inference

from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

__author__ = 'Junior Teudjio'


class MetricsComputer(object):
    def __init__(self):
        self.inference_model = inference.Model()
        self.default_metrics_dict = {
            'accuracy' : accuracy_score,
            'f1' : f1_score,
            'recall' : recall_score,
            'precision' : precision_score,
        }

    def get(self, folder_path, model_type='mobilenet_model', metrics_dict=None):
        if metrics_dict is None:
            metrics_dict = self.default_metrics_dict

        true_labels, pred_labels = self.inference_model.predict_on_batch(folder_path, model_type)
        scores = dict()
        for metric, metric_func in metrics_dict.iteritems():
            scores[metric] = metric_func(true_labels, pred_labels)

        return scores

    def get_train_val_test_scores(self, model_type='mobilenet_model'):
        train_scores = self.get('data/train', model_type=model_type)
        val_scores = self.get('data/validation', model_type=model_type)
        test_scores = self.get('data/test', model_type=model_type)

        return train_scores, val_scores, test_scores

    def get_base_model_scores(self):
        return self.get_train_val_test_scores(model_type='base_model')

    def get_mobilenet_model_scores(self):
        return self.get_train_val_test_scores(model_type='mobilenet_model')



if __name__ == '__main__':
    metrics_computer = MetricsComputer()

    print metrics_computer.get_base_model_scores()
    print metrics_computer.get_mobilenet_model_scores()