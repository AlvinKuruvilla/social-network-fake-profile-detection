from classifiers.template_generator import all_ids
from experiments import print_k_table
from performance_evaluation.heatmap import VerifierType, HeatMap


heatmap = HeatMap(VerifierType.COSINE)

matrix = heatmap.combined_keystroke_matrix(1, 2, None, None, 1)
ids = all_ids()
print()
print("FI")
print_k_table(matrix=matrix, ids=ids)
