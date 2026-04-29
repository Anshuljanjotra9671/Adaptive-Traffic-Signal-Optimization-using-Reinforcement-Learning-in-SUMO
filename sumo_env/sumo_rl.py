# ================== IMPORTS ==================
import os
import sys
import time
import pickle

# ================== SUMO SETUP ==================
if 'SUMO_HOME' not in os.environ:
    os.environ['SUMO_HOME'] = r"C:\Program Files (x86)\Eclipse\Sumo"

tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
sys.path.append(tools)

import traci

PORT = 8813

# ================== CONFIG ==================
MODEL_PATH = "models/q_table.pkl"
SWITCH_TIME = 20  # must match training


# ================== LOAD MODEL ==================
def load_q_table():
    try:
        with open(MODEL_PATH, "rb") as f:
            print("Q-table loaded ✅")
            return pickle.load(f)
    except FileNotFoundError:
        print("❌ Model not found at:", MODEL_PATH)
        return {}
    except Exception as e:
        print("❌ Error loading model:", e)
        return {}


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

    counts = []
    for lane in lanes[:4]:
        counts.append(traci.lane.getLastStepVehicleNumber(lane))

    return tuple(counts)


# ================== ACTION ==================
def choose_action(state, q_table):
    state_t = state_to_tuple(state)

    q0 = q_table.get((state_t, 0), 0)
    q1 = q_table.get((state_t, 1), 0)

    return 0 if q0 >= q1 else 1


# ================== SIGNAL CONTROL ==================
def set_phase(tls_id, action):
    if action == 0:
        traci.trafficlight.setPhase(tls_id, 0)  # NS green
    else:
        traci.trafficlight.setPhase(tls_id, 2)  # EW green


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

    raise Exception("❌ Could not connect to SUMO")


# ================== MAIN RUN ==================
def run():
    connect()

    q_table = load_q_table()

    if not q_table:
        print("⚠️ Running without trained model (random-like behavior)")

    tls_list = traci.trafficlight.getIDList()

    if not tls_list:
        print("❌ No traffic lights found")
        return

    tls_id = tls_list[0]
    print("Using Traffic Light:", tls_id)

    step = 0
    switch_timer = 0
    current_action = 0

    while step < 1000:
        traci.simulationStep()
        time.sleep(0.05)

        state = get_state(tls_id)

        # change action only when timer resets
        if switch_timer == 0:
            current_action = choose_action(state, q_table)

        set_phase(tls_id, current_action)

        # timer update
        switch_timer += 1
        if switch_timer >= SWITCH_TIME:
            switch_timer = 0

        # print useful debug info
        total_cars = sum(state)
        print(f"Step {step} | Cars: {total_cars} | State: {state} | Action: {current_action}")

        step += 1

    traci.close()
    print("Simulation ended ✅")


# ================== ENTRY ==================
if __name__ == "__main__":
    run()