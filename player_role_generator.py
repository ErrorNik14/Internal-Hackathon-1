import pandas as pd
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

DATASET_DIR = './Datasets/IPL DELIVERIES/'

data = pd.concat(
    [pd.read_csv(f'{DATASET_DIR}/ipl_{y}_deliveries.csv') for y in [2022, 2023, 2024, 2025]],
    ignore_index=True
)
data['date'] = pd.to_datetime(data['date'], format='%b %d, %Y')

player_runs  = defaultdict(int)   # actual runs
player_balls = defaultdict(int)   # balls faced
player_dels  = defaultdict(int)   # legal deliveries bowled
player_wicket = defaultdict(int)
player_field = defaultdict(int)

for row in data.itertuples(index=False):
    player_runs[row.striker] += 1
    if not row.wide:
        player_balls[row.striker] += 1
    if not row.wide and not row.noballs:
        player_dels[row.bowler] += 1
    if row.wicket_type=='stumped':
        player_wicket[row.fielder] += 1
    # if row.wicket_type in []

combined_data = []
defaults = (0, 0, 0)
all_names = list(player_runs.keys() | player_dels.keys())

for name in all_names:
    runs = player_balls.get(name, defaults[0])
    balls = player_dels.get(name, defaults[1])
    isWicket = 1 if player_wicket.get(name, defaults[2]) else 0
    
    total = runs + balls

    if total==0:
        runs = -1
        balls = -9
        total = 10

    print(name, runs, balls, total)

    combined_data.append([int(runs/total*100), int(balls/total*100), isWicket])

combined_np = np.array(combined_data)

to_store = pd.DataFrame({'Name':all_names, 'Perc_ of balls faced':combined_np[:,0], 'Perc_ of balls delivered': combined_np[:,1], 'is Wicket keeper': combined_np[:,2]})

to_store.to_csv('./csv files/player_behaviour_data.csv')

def get_roles(behaviour_csv):
    b = pd.read_csv(behaviour_csv, index_col=0)
    b.columns = ['name', 'pct_faced', 'pct_bowled', 'is_wk']

    def role(r):
        if r.is_wk:            return 'WK'
        if r.pct_faced >= 75:  return 'BAT'
        if r.pct_faced <= 25:  return 'BOWL'
        return 'AR'

    b['role'] = b.apply(role, axis=1)
    b.to_csv('./csv files/player_roles.csv', index=False)
    return dict(zip(b['name'], b['role']))

roles = get_roles('./csv files/player_behaviour_data.csv')

if __name__=='__main__':
    plt.hist(combined_np[:,0], bins=11)
    plt.title('Distribution of % of balls_batted-to-balls_delivered')
    plt.xlabel('% of balls_batted-to-balls_delivered')
    plt.ylabel('Frequency')
    plt.show()


    plt.hist(combined_np[:,2], bins=[-0.5, 0.5, 1.5], rwidth=0.8, color='skyblue', edgecolor='black')
    plt.xticks([0, 1], ['False', 'True'])
    plt.title('No. of wicket-keepers')
    plt.xlabel('Is Wicket Keeper?')
    plt.ylabel('Frequency')
    plt.show()