from akkadian_bert.train_bert import tokens_stats
from akkadian_bert.evaluate_bert import generate_evaluations
import logging
# tokens_stats("models/mbert_with_hyphens_with_pseudowords_all_projects/bert_akk_train_dataset.txt",
#              'models/mbert_with_hyphens_with_pseudowords_all_projects')

logger = logging.getLogger()
logger.setLevel(logging.INFO)
handler = logging.FileHandler(filename="./models/mt/mt-va-15/eval.log", mode='w', encoding='utf-8')
handler.setFormatter(logging.Formatter('%(name)s %(message)s'))
logger.addHandler(handler)
generate_evaluations('models/mt/test.txt', 'models/mt/mt-va-15', 'models/mt/vocab-mbert-5000.txt', k=30)

