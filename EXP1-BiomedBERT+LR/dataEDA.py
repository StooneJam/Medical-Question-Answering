import json
import matplotlib.pyplot as plt
import numpy as np

with open('data\ori_pqal.json', 'r') as f:
    data = json.load(f)

labels = [v['final_decision'] for v in data.values()]
counts = {'yes': labels.count('yes'),
          'no': labels.count('no'),
          'maybe': labels.count('maybe')}

plt.figure(figsize=(6, 4))
plt.bar(counts.keys(), counts.values(), color=['steelblue', 'tomato', 'gray'])
plt.title('Label Distribution in PQA-L')
plt.xlabel('Label')
plt.ylabel('Count')
for i, (k, v) in enumerate(counts.items()):
    plt.text(i, v + 5, f'{v}\n({v/len(labels)*100:.1f}%)', ha='center')
plt.tight_layout()
plt.savefig('label_distribution.png', dpi=150)
plt.show()

lengths = []
for v in data.values():
    abstract = ' '.join(v['CONTEXTS'])
    lengths.append(len(abstract.split()))

plt.figure(figsize=(6, 4))
plt.hist(lengths, bins=30, color='steelblue', edgecolor='white')
plt.axvline(np.mean(lengths), color='red', linestyle='--',
            label=f'Mean: {np.mean(lengths):.0f} words')
plt.title('Context Length Distribution in PQA-L')
plt.xlabel('Number of Words')
plt.ylabel('Count')
plt.legend()
plt.tight_layout()
plt.savefig('Context_length.png', dpi=150)
plt.show()

print(f"Total samples: {len(data)}")
print(f"Label counts: {counts}")
print(f"Mean abstract length: {np.mean(lengths):.1f} words")
print(f"Max abstract length: {max(lengths)} words")