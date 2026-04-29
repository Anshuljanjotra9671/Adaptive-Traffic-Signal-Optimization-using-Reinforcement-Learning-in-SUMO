# ================== IMPORTS ==================
import os
import sys
import time
import random
import pickle
import matplotlib.pyplot as plt

# ================== FOLDER SETUP ==================
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

# ================== SUMO SETUP ==================
if 'SUMO_HOME' not in os.environ:
    os.environ['SUMO_HOME'] = r"C:\Program Files (x86)\Eclipse\Sumo"

tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
sys.path.append(tools)

import traci

PORT = 8813

# ================== Q-LEARNING PARAMS ==================
ALPHA = 0.1
GAMMA = 0.9
EPSILON = 1.0
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.05

SWITCH_TIME = 20
SWITCH_PENALTY = -2

q_table = {}

# ================== STATE ==================
def bucket(x):
    if x <= 5:
        return 0
    elif x <= 10:
        return 1
    else:
        return 2

def state_to_tuple(state):
    return tuple(bucket(x) for x in state)

def get_state(tls_id):
    lanes = traci.trafficlight.getControlledLanes(tls_id)
    return tuple(traci.lane.getLastStepVehicleNumber(lane) for lane in lanes[:4])

# ================== REWARD ==================
def get_reward(state, action, prev_action):
    waiting = sum(state)
    imbalance = abs((state[0] + state[1]) - (state[2] + state[3]))

    reward = -(waiting + 0.5 * imbalance)

    if action != prev_action:
        reward += SWITCH_PENALTY

    return reward

# ================== ACTION ==================
def choose_action(state):
    global EPSILON

    state_t = state_to_tuple(state)

    if random.random() < EPSILON:
        return random.choice([0, 1])

    q0 = q_table.get((state_t, 0), 0)
    q1 = q_table.get((state_t, 1), 0)

    return 0 if q0 >= q1 else 1

# ================== UPDATE ==================
def update_q(state, action, reward, next_state):
    state_t = state_to_tuple(state)
    next_t = state_to_tuple(next_state)

    old_q = q_table.get((state_t, action), 0)

    next_max = max(
        q_table.get((next_t, 0), 0),
        q_table.get((next_t, 1), 0)
    )

    new_q = old_q + ALPHA * (reward + GAMMA * next_max - old_q)
    q_table[(state_t, action)] = new_q

# ================== SIGNAL ==================
def set_phase(tls_id, action):
    if action == 0:
        traci.trafficlight.setPhase(tls_id, 0)
    else:
        traci.trafficlight.setPhase(tls_id, 2)

# ================== CONNECTION ==================
def connect():
    for _ in range(10):
        try:
            traci.init(PORT)
            print("Connected to SUMO ✅")
            return
        except:
            print("Waiting for SUMO...")
            time.sleep(1)

    raise Exception("Connection failed")

# ================== TRAIN ==================
def train():
    global EPSILON

    connect()

    tls_list = traci.trafficlight.getIDList()

    if not tls_list:
        print("❌ No traffic lights found")
        return

    tls_id = tls_list[0]
    print("Using TLS:", tls_id)

    EPISODES = 50
    STEPS = 500

    rewards_history = []

    for ep in range(EPISODES):
        total_reward = 0
        switch_timer = 0
        action = 0
        prev_action = 0

        print(f"\nEpisode {ep+1}")

        for step in range(STEPS):
            traci.simulationStep()
            time.sleep(0.05)

            state = get_state(tls_id)

            if switch_timer == 0:
                action = choose_action(state)

            set_phase(tls_id, action)

            switch_timer += 1
            if switch_timer >= SWITCH_TIME:
                switch_timer = 0

            traci.simulationStep()

            next_state = get_state(tls_id)

            reward = get_reward(next_state, action, prev_action)

            update_q(state, action, reward, next_state)

            total_reward += reward
            prev_action = action

        EPSILON = max(EPSILON * EPSILON_DECAY, MIN_EPSILON)

        rewards_history.append(total_reward)

        print(f"Reward: {total_reward} | Epsilon: {round(EPSILON,3)}")

    # ================== SAVE MODEL ==================
    with open("models/q_table.pkl", "wb") as f:
        pickle.dump(q_table, f)

    print("Model saved ✅")

    # ================== SAVE GRAPH ==================
    plt.plot(rewards_history)
    plt.title("Training Performance")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")

    plt.savefig("results/training_plot.png")
    plt.show()

    # ================== SAVE METRICS ==================
    with open("results/metrics.txt", "w") as f:
        f.write("Training Summary\n")
        f.write(f"Episodes: {EPISODES}\n")
        f.write(f"Final Epsilon: {EPSILON}\n")
        f.write(f"Max Reward: {max(rewards_history)}\n")
        f.write(f"Min Reward: {min(rewards_history)}\n")

    print("Results saved ✅")

    traci.close()

# ================== MAIN ==================
if __name__ == "__main__":
    train()