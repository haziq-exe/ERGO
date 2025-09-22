# This file is deprecated and no longer in use


# from evaluation.evaluator import CodeEvaluator, GSM8KEvaluator, DatabaseEvaluator, ActionsEvaluator, DataToTextEvaluator

# class RunEvaluation:
#     def __init__(self, dataset_path):
#         self.dataset_path = dataset_path
#         pass

#     def Code_evaluate(self, numQ, num_runs, input_path):
#         evaluator = CodeEvaluator(input_path, self.dataset_path)
#         scores, average_accuracy = evaluator.evaluate(num_runs, numQ)
#         print(f"Code: Average Accuracy over {num_runs} runs: {average_accuracy}")
#         return scores, average_accuracy

#     def GSM8K_evaluate(self, numQ, num_runs, input_path):
#         evaluator = GSM8KEvaluator(input_path, self.dataset_path)
#         scores, average_accuracy = evaluator.evaluate(num_runs, numQ)
#         print(f"GSM8K: Average Accuracy over {num_runs} runs: {average_accuracy}")
#         return scores, average_accuracy
    
#     def Database_evaluate(self, numQ, num_runs, input_path, spider_DB_path):
#         evaluator = DatabaseEvaluator(input_path, self.dataset_path)
#         scores, average_accuracy = evaluator.evaluate(num_runs, numQ, spider_DB_path=spider_DB_path)
#         print(f"Database: Average Accuracy over {num_runs} runs: {average_accuracy}")
#         return scores, average_accuracy

#     def Actions_evaluate(self, numQ, num_runs, input_path):
#         evaluator = ActionsEvaluator(input_path, self.dataset_path)
#         scores, average_accuracy = evaluator.evaluate(num_runs, numQ)
#         print(f"Actions: Average Accuracy over {num_runs} runs: {average_accuracy}")
#         return scores, average_accuracy

#     def DataToText_evaluate(self, numQ, num_runs, input_path):
#         evaluator = DataToTextEvaluator(input_path, self.dataset_path)
#         scores, average_accuracy = evaluator.evaluate(num_runs, numQ)
#         print(f"DataToText: Average Accuracy over {num_runs} runs: {average_accuracy}")
#         return scores, average_accuracy