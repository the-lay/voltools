import voltools as vt
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt

size = (50, 50, 50)
data = np.random.random(size).astype(np.float32)

rows = [
    [
        {
            'interpolation': 'linear',
            'device': 'cpu'
        },
        {
            'interpolation': 'bspline',
            'device': 'cpu'
        },
        {
            'interpolation': 'filt_bspline',
            'device': 'cpu'
        },
    ],
    [
        {
            'interpolation': 'linear',
            'device': 'gpu:0'
        },
        {
            'interpolation': 'bspline',
            'device': 'gpu:0'
        },
        {
            'interpolation': 'filt_bspline',
            'device': 'gpu:0'
        },
    ]
]

fig, ax = plt.subplots(len(rows), len(rows[0]), sharex=True, sharey=True)
for i, r in enumerate(rows):
    for j, case in enumerate(r):
        print(f'Test case: {case["interpolation"]} / {case["device"]}')
        tf = vt.transform(data, rotation=(0, 30, 0), interpolation=case['interpolation'],
                          profile=True, device=case['device'])

        ax[i][j].set_title(f'{case["interpolation"]} / {case["device"]}')

        if isinstance(tf, cp.ndarray):
            ax[i][j].imshow(tf[size[0] // 2].get())
        else:
            ax[i][j].imshow(tf[size[0] // 2])

plt.show()


