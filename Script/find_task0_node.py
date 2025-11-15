"""Find which node task 0 is on and its clock speed"""
import json

sol = json.load(open('solution/T2_var_001_ga.json'))
t0 = [t for t in sol if t['task_id'] == 0][0]
print(f"Task 0 assigned to node: {t0['node_id']}")

plat = json.load(open('Path_Information/3_Tier_Platform.json'))
node = [n for n in plat['platform']['nodes'] if n['id'] == t0['node_id']]

if node:
    print(f"Node type: {node[0]['type_of_processor']}")
    print(f"Clocking speed: {node[0]['clocking_speed']}")
    
    # Now verify the timing calculation
    clk_str = node[0]['clocking_speed']
    if 'GHz' in clk_str:
        clk_hz = float(clk_str.split()[0]) * 1e9
    elif 'MHz' in clk_str:
        clk_hz = float(clk_str.split()[0]) * 1e6
    else:
        clk_hz = None
        
    print(f"Clock in Hz: {clk_hz:,.0f}")
    
    # Load application
    app = json.load(open('Application/T2_var_001.json'))
    task0_app = [j for j in app['application']['jobs'] if j['id'] == 0][0]
    
    proc_time = task0_app['processing_times']
    wcet = task0_app['wcet_fullspeed']
    
    calc_duration_from_proc = proc_time / clk_hz
    calc_duration_from_wcet = wcet / clk_hz
    actual_duration = t0['end_time'] - t0['start_time']
    
    print(f"\nTiming Analysis:")
    print(f"  processing_times: {proc_time}")
    print(f"  wcet_fullspeed: {wcet}")
    print(f"  Calculated from proc_time: {calc_duration_from_proc:.10e}")
    print(f"  Calculated from wcet: {calc_duration_from_wcet:.10e}")
    print(f"  Actual duration: {actual_duration:.10e}")
    print(f"  Match with proc_time? {abs(calc_duration_from_proc - actual_duration) < 1e-9}")
    print(f"  Match with wcet? {abs(calc_duration_from_wcet - actual_duration) < 1e-9}")
else:
    print(f"Node {t0['node_id']} not found in platform!")
