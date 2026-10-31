import os
import json
import re
import string
import logging
import pandas as pd
import csv
import io
from pathlib import Path
from collections import Counter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("evaluate")

class Evaluator:
    
    def parse_multiple_answers(self, answer_str):
        answer_str = answer_str.strip()
        
        if "|" in answer_str:
            return [ans.strip() for ans in answer_str.split("|")]
        
        if '"' in answer_str:
            try:
                csv_reader = csv.reader(io.StringIO(answer_str))
                answers = next(csv_reader)
                return [ans.strip() for ans in answers if ans.strip()]
            except Exception:
                pass
        
        if "," in answer_str:
            return [ans.strip() for ans in answer_str.split(",")]
        
        return [answer_str]
    
    def get_label_pred_list(self, df, pred_col, label_col):
        pred_list = df[pred_col].tolist()
        label_list = df[label_col].tolist()
        return label_list, pred_list
    
    def normalize_answer(self, s):
        def remove_articles(text):
            return re.sub(r'\b(a|an|the|An|The)\b', ' ', text)
        
        def white_space_fix(text):
            return ' '.join(text.split())
        
        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)
        
        def lower(text):
            return text.lower()
        
        return white_space_fix(remove_articles(remove_punc(lower(s))))
    
    def eval_accuracy(self, prediction, ground_truth):
        prediction = prediction.replace("Yes", "yes").replace("No", "no")
        ground_truth = ground_truth.replace("Yes", "yes").replace("No", "no")
        
        normalized_pred = self.normalize_answer(prediction)
        normalized_gt = self.normalize_answer(ground_truth)
        
        if normalized_gt in normalized_pred:
            return 1
        else:
            return 0
    
    def f1_score(self, prediction, ground_truth):
        prediction = prediction.replace("Yes", "yes").replace("No", "no")
        ground_truth = ground_truth.replace("Yes", "yes").replace("No", "no")
        
        prediction_tokens = self.normalize_answer(prediction).split()
        ground_truth_tokens = self.normalize_answer(ground_truth).split()
        
        if len(prediction_tokens) == 0 or len(ground_truth_tokens) == 0:
            return 0, 0, 0
            
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        
        if num_same == 0:
            return 0, 0, 0
            
        precision = num_same / len(prediction_tokens)
        recall = num_same / len(ground_truth_tokens)
        f1 = 2 * precision * recall / (precision + recall)
        
        return f1, precision, recall
    
    def exact_match_score(self, prediction, ground_truth):
        prediction = prediction.replace("Yes", "yes").replace("No", "no")
        ground_truth = ground_truth.replace("Yes", "yes").replace("No", "no")
        
        return int(self.normalize_answer(prediction) == self.normalize_answer(ground_truth))
    
    def eval_results(self, results_file):
        with open(results_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        results = data.get('results', [])
        
        successful_results = [r for r in results if r.get('status') == 'success']
        
        avg_query_time = 0
        if successful_results:
            total_query_time = sum(r.get('query_time', 0) for r in successful_results)
            avg_query_time = total_query_time / len(successful_results)
        
        df = pd.DataFrame(successful_results)
        
        if 'answer' not in df.columns or 'golden_answer' not in df.columns:
            logger.error("Results file missing required columns: 'answer' or 'golden_answer'")
            return None
            
        df = df.rename(columns={'answer': 'output', 'golden_answer': 'answer'})
        
        res_dict, evaluated_df = self.short_eval(df)
        
        res_dict['avg_query_time'] = avg_query_time
        
        output_eval_file = results_file.replace('.json', '_eval.json')
        
        eval_data = {
            "metadata": data.get('metadata', {}),
            "eval_metrics": {
                "accuracy": res_dict["accuracy"],
                "f1": res_dict["f1"]
            },
            "per_question_metrics": evaluated_df[['question_id', 'question', 'output', 'answer', 'accuracy', 'f1']].to_dict('records')
        }
        
        with open(output_eval_file, 'w', encoding='utf-8') as f:
            json.dump(eval_data, f, ensure_ascii=False, indent=2)
            
        logger.info(f"Evaluation completed! Results saved to: {output_eval_file}")
        
        return res_dict, evaluated_df
        
    def short_eval(self, df: pd.DataFrame):
        accuracy_list = []
        f1_list = []
        precision_list = []
        recall_list = []
        em_list = []

        label_list, pred_list = self.get_label_pred_list(df, "output", "answer")

        for prediction, answer in zip(pred_list, label_list):
            prediction = str(prediction) if prediction is not None else ""
            answer = str(answer) if answer is not None else ""
            
            prediction = prediction.replace("|", "\n")
            prediction = prediction.split("\n")
            prediction_str = " ".join(prediction)

            answer_list = self.parse_multiple_answers(answer)
            
            if len(answer_list) == 1:
                single_answer = answer_list[0].strip()
                accuracy = self.eval_accuracy(prediction_str, single_answer)
                f1, prec, recall = self.f1_score(prediction_str, single_answer)
                em = self.exact_match_score(prediction_str, single_answer)
            else:
                all_correct = True
                covered_answers = 0
                total_answers = 0
                
                for single_answer in answer_list:
                    single_answer = single_answer.strip()
                    if not single_answer:
                        continue
                    
                    total_answers += 1
                    
                    single_accuracy = self.eval_accuracy(prediction_str, single_answer)
                    if single_accuracy == 0:
                        all_correct = False
                    else:
                        covered_answers += 1
                
                accuracy = 1 if all_correct else 0
                
                if total_answers == 0:
                    f1, prec, recall = 0, 0, 0
                else:
                    recall = covered_answers / total_answers
                    prec = recall
                    f1 = recall
                
                exact_matches = 0
                for single_answer in answer_list:
                    single_answer = single_answer.strip()
                    if not single_answer:
                        continue
                    
                    single_em = self.exact_match_score(prediction_str, single_answer)
                    if single_em == 1:
                        exact_matches += 1
                
                em = 1 if exact_matches == total_answers and total_answers > 0 else 0
            
            em_list.append(em)
            f1_list.append(f1)
            precision_list.append(prec)
            recall_list.append(recall)
            accuracy_list.append(accuracy)

        accuracy = sum(accuracy_list) * 100 / len(accuracy_list) if accuracy_list else 0
        f1 = sum(f1_list) * 100 / len(f1_list) if f1_list else 0
        pre = sum(precision_list) * 100 / len(precision_list) if precision_list else 0
        recall = sum(recall_list) * 100 / len(recall_list) if recall_list else 0
        em = sum(em_list) * 100 / len(em_list) if em_list else 0

        df["accuracy"] = accuracy_list
        df["f1"] = f1_list
        df["precision"] = precision_list
        df["recall"] = recall_list
        df["em"] = em_list

        res_dict = {
            "accuracy": accuracy,
            "f1": f1,
            "precision": pre,
            "recall": recall,
            "em": em,
        }

        return res_dict, df

def run_evaluation(results_file):
    evaluator = Evaluator()
    
    if not Path(results_file).exists():
        logger.error(f"Results file {results_file} does not exist!")
        return None
    
    logger.info(f"Starting evaluation of results file: {results_file}")
    
    metrics, _ = evaluator.eval_results(results_file)
    
    return metrics

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate GraphRAG query results")
    parser.add_argument("--results-file", type=str, required=True, help="Query results file path")
    
    args = parser.parse_args()
    
    evaluation_results = run_evaluation(args.results_file)
    
    if evaluation_results:
        print("Evaluation metrics:")
        print(f"accuracy: {evaluation_results['accuracy']:.4f}")
        print(f"f1: {evaluation_results['f1']:.4f}") 