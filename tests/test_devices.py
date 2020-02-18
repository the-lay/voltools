import voltools as vt
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
plt.ioff()

size = (50, 50, 50)
data = np.random.random(size).astype(np.float32)
rotation = np.random.uniform(0, 180, 3)
scale = np.random.uniform(0.7, 1.3, 3)

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
            'device': 'gpu'
        },
        {
            'interpolation': 'bspline',
            'device': 'gpu'
        },
        {
            'interpolation': 'filt_bspline',
            'device': 'gpu'
        },
    ]
]

### testing transforms methods
fig, ax = plt.subplots(len(rows), len(rows[0]), sharex=True, sharey=True)
for i, r in enumerate(rows):
    for j, case in enumerate(r):
        print(f'Test case: {case["interpolation"]} / {case["device"]}')
        tf = vt.transform(data, rotation=rotation, scale=scale, interpolation=case['interpolation'],
                          profile=True, device=case['device'])

        ax[i][j].set_title(f'{case["interpolation"]} / {case["device"]}')
        ax[i][j].imshow(tf[size[0] // 2])

plt.show()


### testing static volume methods
print('\n\n\n')
st_volumes = [
    vt.StaticVolume(data, interpolation='linear', device='cpu'),
    vt.StaticVolume(data, interpolation='bspline', device='cpu'),
    vt.StaticVolume(data, interpolation='filt_bspline', device='cpu'),
    vt.StaticVolume(data, interpolation='linear', device='gpu'),
    vt.StaticVolume(data, interpolation='bspline', device='gpu'),
    vt.StaticVolume(data, interpolation='filt_bspline', device='gpu')
]

fig, ax = plt.subplots(2, 3, sharex=True, sharey=True)

for n, v in enumerate(st_volumes):
    print(f'Test case: {v.interpolation} / {v.device}')
    tf = v.transform(scale=scale, rotation=rotation, profile=True)
    i, j = int(n / 3), n % 3
    ax[i][j].set_title(f'{v.interpolation} / {v.device}')
    ax[i][j].imshow(tf[size[0] // 2])

plt.show()

