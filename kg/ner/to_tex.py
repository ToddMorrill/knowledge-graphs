import pandas as pd

df = pd.DataFrame(
    dict(name=['Raphael', 'Donatello'],
         mask=['red', 'purple'],
         weapon=['sai', 'bo staff']))

# longtable=False
df.to_latex(
    '../../reports/entity_detection/test.tex',
    # index=False,
    # column_format='c | c | c',
    caption=('test caption', 'test'),
    # label='tab:test'
)
