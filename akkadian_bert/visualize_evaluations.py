import seaborn as sns
import matplotlib.pyplot as plt
from pandas import DataFrame

# nono_k_1 = {'eval_loss': 1.1600912602742512, 'mrr': 0.8397491129573743, 'hit_k': 0.7854}
# nono_k_5 = {'eval_loss': 1.1600913035869598, 'mrr': 0.8397491129573743, 'hit_k': 0.8935}
# nono_k_10 = {'eval_loss': 1.1600912602742512, 'mrr': 0.8397491129573743, 'hit_k': 0.9274}
#
# wino_k_1 = {'eval_loss': 0.8353283347686132, 'mrr': 0.8614783528744009, 'hit_k': 0.81465}
# wino_k_5 = {'eval_loss': 0.8353283347686132, 'mrr': 0.8614783528744009, 'hit_k': 0.908}
# wino_k_10 = {'eval_loss': 0.8353283347686132, 'mrr': 0.8614783528744009, 'hit_k': 0.945}
#
# wiwi_k_1 = {'eval_loss': 0.7624893327554066, 'mrr': 0.867258836512274, 'hit_k': 0.825}
# wiwi_k_5 = {'eval_loss': 0.7624893327554066, 'mrr': 0.867258836512274, 'hit_k': 0.9137}
# wiwi_k_10 = {'eval_loss': 0.7599415466189384, 'mrr': 0.8677999036424576, 'hit_k': 0.9425}
#
# df = DataFrame({
#     'Preprocessing': ['- pseudo\n- hyphens', '- pseudo\n+ hyphens', '+ pseudo\n+ hyphens'],
#     'mrr': [nono_k_1['mrr'], wino_k_10['mrr'], wiwi_k_1['mrr']],
#     'hit_k=1': [nono_k_1['hit_k'], wino_k_1['hit_k'], wiwi_k_1['hit_k']],
#     'hit_k=5': [nono_k_5['hit_k'], wino_k_5['hit_k'], wiwi_k_5['hit_k']],
#     'hit_k=10': [nono_k_10['hit_k'], wino_k_10['hit_k'], wiwi_k_10['hit_k']],
#
# })
# sns.set_theme(style="whitegrid")
#
# tidy = df.melt(id_vars='Preprocessing').rename(columns=str.title)
# ax = sns.barplot(x='Preprocessing', y='Value', hue='Variable', data=tidy)
# ax.set(ylim=(0.75, 1.0))
#
#
# # df_results = DataFrame(data=data, columns=['preprocessing', 'mrr', 'hit_k=1', 'hit_k=5', 'hit_k=10'])
# # tips = sns.load_dataset("tips")
# # ax = sns.barplot(x='Preprocessing', y='mrr', data=df)
# plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
# plt.savefig('current_results.jpg')
# plt.show()

# Qs = [1609, 2143, 2047, 2550, 1438]
sign_1_hit_ks = [0.3430702299564947, 0.454319453076445, 0.4934742075823493, 0.5301429459291486, 0.545680546923555]
sign_2_hit_ks = [0.08866075594960336, 0.14325711619225384, 0.15865608959402708, 0.17638824078394774, 0.1843210452636491]
sign_3_hit_ks = [0.04738641914997557, 0.08353688324377137, 0.0903761602344895, 0.09721543722520762, 0.10014655593551539]
sign_4_hit_ks = [0.016862745098039214, 0.02431372549019608, 0.027058823529411764, 0.02784313725490196,
                 0.02823529411764706]
sign_5_hit_ks = [0.006258692628650904, 0.016689847009735744, 0.016689847009735744, 0.016937669376693765,
                 0.016937669376693765]
signs_hit_ks = [sign_1_hit_ks, sign_2_hit_ks, sign_3_hit_ks, sign_4_hit_ks, sign_5_hit_ks]
akk_one_token_hit_ks = [0.8063504508036065, 0.853390827126617, 0.8749509996079968, 0.8902391219129753, 0.8992551940415523]

word_1_hit_ks = [0.496, 0.571, 0.603, 0.627, 0.637]
word_2_hit_ks = [0.207, 0.259, 0.286, 0.296, 0.306]
word_3_hit_ks = [0.069, 0.086, 0.093, 0.1, 0.109]
word_4_hit_ks = [0.027, 0.038, 0.046, 0.05, 0.05]
word_5_hit_ks = [0.009, 0.011, 0.012, 0.012, 0.013]
words_hit_ks = [word_1_hit_ks, word_2_hit_ks, word_3_hit_ks, word_4_hit_ks, word_5_hit_ks]
eng_one_token_hit_ks = [0.5718818751057708, 0.6474022677271958, 0.6842528346589948, 0.7062531731257404, 0.7240649856151633]

plt.plot([hit_k * 100 for hit_k in akk_one_token_hit_ks], label='1 token')
for i in range(len(signs_hit_ks)):
    plt.plot([hit_k * 100 for hit_k in signs_hit_ks[i]], label=f"{i + 1} sign{'' if i == 0 else 's'}")
plt.ylabel("Precision in %")
plt.xticks(range(len(signs_hit_ks)), [f"hit@{i + 1}" for i in range(len(signs_hit_ks))])
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.ylim(0, 100)
# plt.title("Hit@ks prediction as function of number of signs to predict")
plt.show()

plt.plot([hit_k * 100 for hit_k in eng_one_token_hit_ks], label='1 token')
for i in range(len(signs_hit_ks)):
    plt.plot([hit_k * 100 for hit_k in words_hit_ks[i]], label=f"{i + 1} word{'' if i == 0 else 's'}")
plt.ylabel("Precision in %")
plt.xticks(range(len(words_hit_ks)), [f"hit@{i + 1}" for i in range(len(words_hit_ks))])
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.ylim(0, 100)
# plt.title("Hit@ks prediction as function of number of signs to predict")
plt.show()

# mrrs = [0.42402113113735274, 0.12711152589827343, 0.07003745318352059, 0.021777777777777778, 0.011348238482384823]
# plt.title("MRR as a function of number of signs to predict")
# plt.ylabel("Prediction")
# plt.plot(range(1, len(mrrs) + 1), mrrs)
# plt.xticks(range(1, len(mrrs) + 1))
# plt.show()
