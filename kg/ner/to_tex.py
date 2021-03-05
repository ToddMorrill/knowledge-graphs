import pandas as pd
import numpy as np
from sklearn.metrics import classification_report

preds = np.random.choice(a=[False, True], size=(100, 1))
ground_truth = np.random.choice(a=[False, True], size=(100, 1))
report = classification_report(preds, ground_truth, output_dict=True)
accuracy = report.pop('accuracy')
df = pd.DataFrame(report).T
df.index.name = 'Class'
df = df.reset_index()
df.columns = [x.title() for x in df.columns]
df['Class'] = df['Class'].apply(lambda x: x.title())
df['Support'] = df['Support'].astype(int)

table_string = df.to_latex(
    index=False,
    # column_format='c | c | c',
    # caption=('test caption', 'test'),
    float_format="%.2f"
    # label='tab:test'
)

# if you add a caption, it will enclose everything in table environment
table_split = table_string.split('\n')
table_split[0] = table_split[0] + '[ht]'
table = '\n'.join(table_split)

# save file
save_file_path = '../../reports/entity_detection/test.tex'

breakpoint()